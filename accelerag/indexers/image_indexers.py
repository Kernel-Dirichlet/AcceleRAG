import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
import logging
import sqlite3
import numpy as np
from typing import Dict, Optional, List
from ..base_classes import Indexer
from tqdm import tqdm
from ..embedders.image_embedders import ImageEmbedder


class ImageIndexer(Indexer):
    """Default indexer using MobileNet for image indexing."""
    def __init__(
        self,
        model_name='mobilenet_v2',
        device='cpu',
        embedder=None
    ):
        super().__init__(
            model_name=model_name,
            embedder=embedder
        )
        self.device = device
        
        # Initialize embedder if not provided
        if self.embedder is None:
            self.embedder = ImageEmbedder(
                model_name=model_name,
                device=device
            )
            
    def _get_all_files(self, directory):
        """Get all image files in directory recursively"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    all_files.append((full_path, rel_path))
        return all_files
        
    def _get_tag_from_path(self, rel_path, tag_hierarchy):
        """Extract tag from file path based on tag hierarchy."""
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
                    label TEXT,
                    metadata TEXT,
                    filepath TEXT
                )
            """)
            
            # Create centroid table
            cur.execute(f"DROP TABLE IF EXISTS {table_name}_centroid")
            cur.execute(f"""
                CREATE TABLE {table_name}_centroid (
                    id INTEGER PRIMARY KEY,
                    centroid BLOB
                )
            """)
            
            conn.commit()
            logging.info(f"Successfully created tables: {table_name} and {table_name}_centroid")
        except Exception as e:
            logging.error(f"Error creating tables: {e}")
            conn.rollback()
        finally:
            cur.close()
            
    def _compute_and_store_centroid(self, conn, table_name):
        """Compute and store centroid for a table's embeddings."""
        cur = conn.cursor()
        try:
            # Get all embeddings
            cur.execute(f"SELECT embedding FROM {table_name}")
            results = cur.fetchall()
            
            if not results:
                logging.warning(f"No embeddings found in table {table_name}")
                return
                
            # Convert bytes to numpy arrays
            embeddings = []
            for embedding_bytes, in results:
                try:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Error processing embedding: {e}")
                    continue
                    
            if not embeddings:
                logging.warning(f"No valid embeddings found in table {table_name}")
                return
                
            # Stack embeddings and compute centroid
            embeddings_array = np.stack(embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            
            # Store centroid
            centroid_bytes = centroid.tobytes()
            cur.execute(f"DELETE FROM {table_name}_centroid")
            cur.execute(f"INSERT INTO {table_name}_centroid (centroid) VALUES (?)", (centroid_bytes,))
            conn.commit()
            logging.info(f"Successfully computed and stored centroid for table {table_name}")
            
        except Exception as e:
            logging.error(f"Error computing centroid for table {table_name}: {e}")
            conn.rollback()
        finally:
            cur.close()
            
    def _batch_insert_sqlite(self, conn, table_name, batch_data):
        """Insert batch of data into sqlite"""
        cur = conn.cursor()
        try:
            for data in batch_data:
                embedding, label, metadata, filepath = data
                if not isinstance(embedding, np.ndarray):
                    logging.error(f"Invalid embedding type for file: {filepath}")
                    continue
                # Store embedding as BLOB
                embedding_bytes = embedding.tobytes()
                cur.execute(
                    f"INSERT INTO {table_name} (embedding, label, metadata, filepath) VALUES (?, ?, ?, ?)",
                    (embedding_bytes, label, metadata, filepath)
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
        **kwargs
    ):
        """Index images in the specified directory."""
        try:
            # Ensure SQLite database is created in the correct location
            dir_name = os.path.basename(os.path.normpath(corpus_dir)) 
            db_name = f"{dir_name}_embeddings.db.sqlite"
            if db_params is None:
                db_params = {}
            db_path = db_params.get('dbname',db_name)
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(db_path)
            print(f"Creating SQLite database at {db_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            logging.info(f"Using SQLite database at: {db_path}")

            all_files = self._get_all_files(corpus_dir)
            if not all_files:
                logging.warning(f"No image files found in {corpus_dir}")
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

            logging.info(f"Processing {len(all_files)} images...")
            logging.info(f"Using model: {self.model_name}")

            completed = 0
            with tqdm(total=len(all_files), desc="Processing images") as pbar:
                for full_path, rel_path in all_files:
                    try:
                        logging.info(f"Computing embeddings for {rel_path}")
                        
                        # Determine table name from directory
                        dir_path = os.path.dirname(rel_path)
                        if not dir_path:  # Files in the root directory
                            dir_path = os.path.basename(os.path.normpath(corpus_dir))
                        table_name = self._sanitize_table_name(dir_path)
                        
                        # Get label from directory name
                        label = os.path.basename(dir_path)
                        
                        # Generate embedding
                        embedding = self.embedder.embed(full_path)
                        if embedding is None:
                            logging.error(f"Failed to generate embedding for {rel_path}")
                            continue
                            
                        # Insert into the table corresponding to the file's directory
                        self._batch_insert_sqlite(
                            conn,
                            table_name,
                            [(embedding, label, None, rel_path)]
                        )
                            
                        completed += 1
                        pbar.update(1)
                        logging.info(f"Completed {completed}/{len(all_files)} images")
                        
                    except Exception as e:
                        logging.error(f"Error processing {rel_path}: {e}")
                        continue  # Continue with next file even if one fails

            logging.info(f"Successfully processed {completed} images")
            
            # Compute centroids for each table
            logging.info("Computing centroids for each table...")
            for table_name in created_tables:
                self._compute_and_store_centroid(conn, table_name)
            
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
