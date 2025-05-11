import os
import re
import logging
import sqlite3
import numpy as np
from embedders import * 
from base_classes import Indexer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class TextIndexer(Indexer):
    """Default indexer using transformer models for document indexing."""
    def __init__(
        self,
        model_name ='huawei-noah/TinyBERT_General_4L_312D',
        ngram_size = 16,
        device = 'cpu',
        embedder = None
    ):
        super().__init__(
            model_name = model_name,
            ngram_size = ngram_size,
            embedder = embedder
        )
        self.device = device
        
        # Initialize embedder if not provided
        if self.embedder is None:
            self.embedder = TextEmbedder(
                model_name = model_name,
                device = device
            )
            
    def _get_all_files(self, directory):
        """Get all files in directory recursively"""
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)
                all_files.append((full_path, rel_path))
        return all_files
        
    def _get_tag_from_path(self, rel_path, tag_hierarchy):
        """Extract tag from file path based on tag hierarchy.
        
        Args:
            rel_path: Relative path of the file
            tag_hierarchy: Whether to use directory structure as tag hierarchy
            
        Returns:
            The tag (directory path) if tag_hierarchy is True, None otherwise
        """
        if not tag_hierarchy:
            return None
            
        path_parts = rel_path.split(os.sep)
        if len(path_parts) < 2:  # Need at least a directory and a file
            return None
            
        # The tag is the directory path
        tag = '/'.join(path_parts[:-1])
        return tag
        
    def _sanitize_table_name(self, name):
        """Sanitize table name by replacing invalid characters with underscores"""
        # Replace dots, spaces, and other special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure the name starts with a letter or underscore
        if not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        return sanitized
        
    def _get_ngrams(self, text, n):
        """Extract ngrams from text using non-overlapping chunks of size n"""
        # Remove URLs and normalize whitespace
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Split text into words
        words = text.split()
        if len(words) < n:
            return [text]  # Return full text if shorter than ngram size
            
        # Create non-overlapping ngrams
        ngrams = []
        for i in range(0, len(words), n):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
            
        return ngrams
        
    def _create_embeddings_table(self, conn, table_name):
        """Create table for storing embeddings"""
        cur = conn.cursor()
        try:
            # Drop table if it exists to ensure clean state
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table with proper schema
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB,
                    ngram TEXT,
                    filepath TEXT
                )
            """)
            conn.commit()
            logging.info(f"Successfully created table: {table_name}")
        except Exception as e:
            logging.error(f"Error creating table {table_name}: {e}")
            conn.rollback()
        finally:
            cur.close()
            
    def _batch_insert_sqlite(self, conn, table_name, batch_data):
        """Insert batch of data into sqlite"""
        cur = conn.cursor()
        try:
            for data in batch_data:
                ngram, embedding, _, filepath = data
                if not isinstance(embedding, np.ndarray):
                    logging.error(f"Invalid embedding type for ngram: {ngram}")
                    continue
                # Store embedding as BLOB
                embedding_bytes = embedding.tobytes()
                cur.execute(
                    f"INSERT INTO {table_name} (embedding, ngram, filepath) VALUES (?, ?, ?)",
                    (embedding_bytes, ngram, filepath)
                )
            conn.commit()
            logging.info(f"Successfully inserted {len(batch_data)} records into {table_name}")
        except Exception as e:
            logging.error(f"Error inserting into {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            
    def index(
        self,
        corpus_dir,
        tag_hierarchy=None,
        db_params=None,
        batch_size=32,
        ngram_size=16,
        **kwargs
    ):
        """Index documents in the specified directory.
        
        Args:
            corpus_dir: Directory containing documents to index
            tag_hierarchy: Optional tag hierarchy for organizing documents
            db_params: Database parameters
            batch_size: Batch size for processing
            ngram_size: Size of ngrams to extract
            **kwargs: Additional parameters
        """
        try:
            # Ensure SQLite database is created in the correct location
            db_path = db_params.get('dbname', 'embeddings.db.sqlite')
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(db_path)
            print(f"Creating SQLite database at: {db_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            logging.info(f"Using SQLite database at: {db_path}")

            all_files = self._get_all_files(corpus_dir)
            if not all_files:
                logging.warning(f"No files found in {corpus_dir}")
                return

            # Create tables for each directory in the corpus
            created_tables = set()
            for full_path, rel_path in all_files:
                # Get directory from relative path
                dir_path = os.path.dirname(rel_path)
                if not dir_path:  # Files in the root directory
                    dir_path = os.path.basename(os.path.normpath(corpus_dir))
                
                table_name = self._sanitize_table_name(dir_path)
                if table_name not in created_tables:
                    self._create_embeddings_table(conn, table_name)
                    created_tables.add(table_name)
                    logging.info(f"Created table for directory: {table_name}")

            logging.info(f"Processing {len(all_files)} files with {ngram_size}-grams...")
            logging.info(f"Using model: {self.model_name}")

            completed = 0
            with tqdm(total=len(all_files), desc="Processing files") as pbar:
                for full_path, rel_path in all_files:
                    try:
                        logging.info(f"Computing {ngram_size}-gram embeddings for {rel_path}")
                        
                        # Determine table name from directory
                        dir_path = os.path.dirname(rel_path)
                        if not dir_path:  # Files in the root directory
                            dir_path = os.path.basename(os.path.normpath(corpus_dir))
                        table_name = self._sanitize_table_name(dir_path)
                        
                        with open(full_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        paragraphs = text.split('\n\n')
                        results = []
                        
                        for paragraph in paragraphs:
                            if not paragraph.strip():
                                continue
                                
                            ngrams = self._get_ngrams(paragraph, ngram_size)
                            if not ngrams:
                                continue
                                
                            # Process ngrams in batches
                            for i in range(0, len(ngrams), batch_size):
                                batch = ngrams[i:i+batch_size]
                                try:
                                    embeddings = self.embedder.embed_batch(batch)
                                    if embeddings is None or len(embeddings) == 0:
                                        logging.error(f"Failed to generate embeddings for batch in {rel_path}")
                                        continue
                                        
                                    for ngram, embedding in zip(batch, embeddings):
                                        results.append((ngram, embedding, 0, rel_path))
                                except Exception as e:
                                    logging.error(f"Error processing batch in {rel_path}: {e}")
                                    continue
                        
                        if not results:
                            logging.warning(f"No embeddings generated for {rel_path}")
                            continue
                            
                        # Insert into the table corresponding to the file's directory
                        self._batch_insert_sqlite(conn, table_name, results)
                            
                        completed += 1
                        pbar.update(1)
                        logging.info(f"Completed {completed}/{len(all_files)} files")
                        
                    except Exception as e:
                        logging.error(f"Error processing {rel_path}: {e}")
                        continue  # Continue with next file even if one fails

            logging.info(f"Successfully processed {completed} files")
            
            # Verify tables were created and populated
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            tables = cur.fetchall()
            logging.info(f"Final tables in database: {[t[0] for t in tables]}")
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cur.fetchone()[0]
                logging.info(f"Table {table[0]} has {count} records")
            cur.close()

        except Exception as e:
            logging.error(f"Error in indexing: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                logging.info("Database connection closed") 

