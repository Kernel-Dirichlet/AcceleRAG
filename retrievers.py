import json
import os
import sqlite3
import numpy as np
from base_classes import Retriever

class DefaultRetriever(Retriever):
    """Default retriever using fetch_top_k for document retrieval."""
    def __init__(self, dir_to_idx=None, embedder=None):
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
        super().__init__(self.db_params, embedder=embedder)
        
    def route_query(self, query, tag_hierarchy):
        """Route query to relevant tags in the hierarchy."""
        # Simple routing based on exact matches
        tags = []
        for tag, children in tag_hierarchy.items():
            if query.lower() in tag.lower():
                tags.append(tag)
            if isinstance(children, dict):
                tags.extend(self.route_query(query, children))
        return tags
        
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
        
        # Compute query embedding using provided embedder or default
        if self.embedder:
            query_embedding = self.embedder.embed(query)
        else:
            # Default embedding computation if no embedder provided
            query_embedding = np.random.rand(768)  # Placeholder for actual embedding
        
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
            
    def retrieve(self, query, top_k=5, **kwargs):
        """Retrieve relevant chunks from the database.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional arguments to pass to the retrieval function
        """
        return self.fetch_top_k(query, k=top_k, **kwargs) 
