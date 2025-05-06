import json
import os
import logging
import sqlite3
import numpy as np
from typing import Dict, Optional, List, Tuple
from base_classes import Retriever
from embedders.text_embedders import TextEmbedder
from scorers import DefaultScorer
from query_utils import route_query

class TextRetriever(Retriever):
    """Default retriever for semantic search using embeddings."""
    def __init__(self, dir_to_idx: Optional[str] = None, embedder: Optional[TextEmbedder] = None):
        """Initialize the retriever.
        
        Args:
            dir_to_idx: Path to directory containing documents
            embedder: Embedder instance to use for computing embeddings
        """
        # Initialize database path
        if dir_to_idx:
            dir_name = os.path.basename(os.path.normpath(dir_to_idx))
            self.db_path = f"{dir_name}_embeddings.db.sqlite"
        else:
            self.db_path = "embeddings.db.sqlite"
            
        self.db_params = {'dbname': self.db_path}
        self.embedder = embedder or TextEmbedder()
        super().__init__(self.db_params, embedder=self.embedder)
        
    def _compute_cosine_similarity(self, query_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        
    def _get_table_chunks(self, table: str, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Get top-k chunks from a table based on embedding similarity.
        
        Args:
            table: Table name
            query_embedding: Query embedding vector
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, similarity) tuples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Get all chunks and embeddings
            cur.execute(f"SELECT ngram, embedding FROM {table}")
            results = cur.fetchall()
            
            if not results:
                return []
                
            # Compute similarities
            chunks_with_scores = []
            for ngram, embedding_bytes in results:
                try:
                    # Convert bytes to numpy array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    similarity = self._compute_cosine_similarity(query_embedding, embedding)
                    chunks_with_scores.append((ngram, similarity))
                except Exception as e:
                    logging.error(f"Error processing chunk {ngram}: {e}")
                    continue
                    
            # Sort by similarity and return top k
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            return chunks_with_scores[:k]
            
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return []
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
                
    def _get_available_tables(self) -> List[str]:
        """Get list of available tables in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tables
        except sqlite3.Error as e:
            logging.error(f"Error getting available tables: {e}")
            return []

    def _map_tag_to_table(self, tag: str, available_tables: List[str]) -> Optional[str]:
        """Map a tag to an existing table name.
        
        Args:
            tag: Tag from query routing
            available_tables: List of available tables in the database
            
        Returns:
            Mapped table name if found, None otherwise
        """
        # Try exact match first
        if tag in available_tables:
            return tag
            
        # Try different variations of the table name
        if '.' in tag:  # ArXiv category
            possible_names = [
                tag.replace('.', '_'),  # Replace dots with underscores
                tag.split('.')[-1],  # Last component
                '_'.join(tag.split('.')[:-1]),  # Without last component
                '_'.join(tag.split('.')[-2:])  # Last two components
            ]
        else:  # Regular tag
            possible_names = [
                tag,  # Original tag
                tag.split('/')[-1],  # Last component
                '_'.join(tag.split('/')[:-1]),  # Without last component
                '_'.join(tag.split('/')[-2:])  # Last two components
            ]
            
        # Check each possible name
        for name in possible_names:
            if name in available_tables:
                return name
                
        return None

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks from the database.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            # 1. Compute query embedding
            query_embedding = self.embedder.embed(query)
            
            # 2. Get available tables
            available_tables = self._get_available_tables()
            if not available_tables:
                logging.warning("No tables found in database")
                return []
            
            # 3. Route query to appropriate tables
            with open('tag_hierarchy.json', 'r') as f:
                tag_hierarchy = json.load(f)
            tags = route_query(query, tag_hierarchy)
            
            if not tags:
                logging.warning(f"No relevant tags found for query: {query}")
                return []
                
            # 4. Map tags to existing tables
            tables = []
            for tag in tags:
                table = self._map_tag_to_table(tag, available_tables)
                if table:
                    tables.append(table)
                    
            if not tables:
                logging.warning(f"No matching tables found for tags: {tags}")
                return []
                
            # 5. Get chunks from each table
            all_chunks = []
            for table in tables:
                chunks = self._get_table_chunks(table, query_embedding, top_k)
                all_chunks.extend(chunks)
                
            # 6. Sort all chunks by similarity and return top k
            all_chunks.sort(key=lambda x: x[1], reverse=True)
            return all_chunks[:top_k]
            
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []

    def route_query(self, query, tag_hierarchy):
        """Route query to relevant tags using query_utils."""
        return route_query(query, tag_hierarchy)
        
    def fetch_top_k(self, query, k=3, debug=False, max_length=512):
        """Fetch top-k most relevant chunks for query using semantic search"""
        # Load tag hierarchy
        with open('tag_hierarchy.json', 'r') as f:
            tag_hierarchy = json.load(f)
            
        # Route query to get relevant tags
        tags = self.route_query(query, tag_hierarchy)
        if not tags:
            if debug:
                print("No relevant tags found for query.")
            return []
            
        if debug:
            print(f"Found relevant tags: {tags}")
        
        # Compute query embedding
        query_embedding = self.embedder.embed(query)
        
        try:
            # Connect to database
            if 'dbname' in self.db_params:
                db_path = self.db_params['dbname']
                if not os.path.isabs(db_path):
                    db_path = os.path.abspath(db_path)
                if debug:
                    print(f"Connecting to database at: {db_path}")
                
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                
                # Get all available tables
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                available_tables = [row[0] for row in cur.fetchall()]
                # Filter out system tables
                available_tables = [t for t in available_tables if not t.startswith('sqlite_')]
                if debug:
                    print(f"Available tables: {available_tables}")
                
                # Find matching tables based on tags
                matching_tables = []
                for tag in tags:
                    # If tag is already a table name, use it directly
                    if tag in available_tables:
                        matching_tables.append(tag)
                        continue
                        
                    # Try different variations of the table name based on whether it's an ArXiv category
                    if '.' in tag:  # ArXiv category
                        possible_names = [
                            tag.replace('.', '_'),  # Replace dots with underscores
                            tag.split('.')[-1],  # Last component
                            '_'.join(tag.split('.')[:-1]),  # Without last component
                            '_'.join(tag.split('.')[-2:])  # Last two components
                        ]
                    else:  # Regular tag
                        possible_names = [
                            tag,  # Original tag
                            tag.split('/')[-1],  # Last component
                            '_'.join(tag.split('/')[:-1]),  # Without last component
                            '_'.join(tag.split('/')[-2:])  # Last two components
                        ]
                    
                    # Check each possible name
                    for name in possible_names:
                        if name in available_tables:
                            matching_tables.append(name)
                            if debug:
                                print(f"Found matching table: {name} for tag: {tag}")
                            break
                    else:
                        if debug:
                            print(f"Warning: No matching table found for tag: {tag}")
                            print(f"Tried names: {possible_names}")
                            print(f"Available tables: {available_tables}")
                
                if not matching_tables:
                    if debug:
                        print(f"No matching tables found for tags: {tags}")
                    return []
                
                # Build query to search across all matching tables
                queries = []
                params = []
                for table in matching_tables:
                    queries.append(f"""
                        SELECT ngram, filepath, embedding
                        FROM {table}
                        ORDER BY id
                        LIMIT ?
                    """)
                    params.append(k)
                
                # Execute queries and collect results
                all_results = []
                for query, limit, table_name in zip(queries, params, matching_tables):
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    if results:
                        all_results.extend(results)
                        if debug:
                            print(f"Found {len(results)} results in table {table_name}")
                    else:
                        if debug:
                            print(f"No results found in table {table_name}")
                
                cur.close()
                conn.close()
                
                if not all_results:
                    if debug:
                        print("No matching documents found in database.")
                    return []
                
                # Compute similarities and return top k
                chunks = []
                seen_chunks = set()
                
                for ngram, filepath, embedding in all_results:
                    try:
                        if isinstance(embedding, str):
                            embedding = np.array(eval(embedding))
                        if not isinstance(embedding, np.ndarray):
                            if debug:
                                print(f"Warning: Invalid embedding format for ngram: {ngram}")
                            continue
                            
                        # Normalize the chunk text for comparison
                        normalized_chunk = ' '.join(ngram.split())
                        
                        # Skip if we've seen this chunk before
                        if normalized_chunk in seen_chunks:
                            continue
                        seen_chunks.add(normalized_chunk)
                        
                        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                        chunks.append((ngram, similarity, filepath))
                    except Exception as e:
                        if debug:
                            print(f"Error computing similarity for ngram: {ngram}, error: {e}")
                        continue
                    
                # Sort by similarity and return top k unique chunks
                chunks.sort(key=lambda x: x[1], reverse=True)
                return [chunk[0] for chunk in chunks[:k]]
                
            else:
                if debug:
                    print("No database name provided in db_params")
                return []
                
        except sqlite3.Error as e:
            if debug:
                print(f"SQLite error: {str(e)}")
            return []
        except Exception as e:
            if debug:
                print(f"Error fetching chunks: {str(e)}")
            return [] 
