import sqlite3
import numpy as np
from typing import Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from ..embedders import TextEmbedder
from ..base_classes import PromptCache

class DefaultCache(PromptCache):
    """Default implementation of prompt caching using SQLite."""
    def __init__(self, 
                 model_name = 'huawei-noah/TinyBERT_General_4L_312D'):
        self.model_name = model_name
        self.embedder = TextEmbedder(model_name = model_name)
        
    def verify_cache_schema(self,
                            db_path):
        """Verify that the database has the correct response_cache table schema."""
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Get table info
            cur.execute("PRAGMA table_info(response_cache)")
            columns = cur.fetchall()
            
            # Expected schema
            expected_columns = {
                'query': 'TEXT',
                'query_embedding': 'BLOB',
                'response': 'TEXT',
                'quality_score': 'FLOAT',
                'timestamp': 'DATETIME'
            }
            
            # Check if all required columns exist with correct types
            found_columns = {col[1]: col[2] for col in columns}
            if not all(col in found_columns for col in expected_columns):
                return False
                
            # Check primary key constraint
            cur.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name='response_cache'
            """)
            table_sql = cur.fetchone()[0]
            if 'PRIMARY KEY' not in table_sql or 'query' not in table_sql:
                return False
                
            return True
            
        except sqlite3.Error:
            return False
        finally:
            if 'conn' in locals():
                conn.close()
                
    def init_cache(self,
                   db_path):
        """Initialize the cache table in the SQLite database."""
        try:
            # Use prompt_cache.db by default
            cache_path = 'prompt_cache.db'
            if db_path:
                cache_path = db_path
                
            conn = sqlite3.connect(cache_path)
            cur = conn.cursor()
            
            # Check if table exists and has correct schema
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='response_cache'")
            if cur.fetchone():
                if not self.verify_cache_schema(cache_path):
                    raise ValueError("Existing response_cache table has incorrect schema")
                return
                
            # Create cache table with correct schema
            cur.execute("""
                CREATE TABLE response_cache (
                    query TEXT PRIMARY KEY,
                    query_embedding BLOB,
                    response TEXT,
                    quality_score FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            logging.info(f"Cache table initialized in {cache_path}")
        except sqlite3.Error as e:
            logging.error(f"Error initializing cache: {e}")
            raise
            
    def _get_query_embedding(self, query):
        """Get embedding for a query using the specified model."""
        try:
            embeddings = self.embedder.embed_batch([query])
            return embeddings[0]  # Return first (and only) embedding
        except Exception as e:
            logging.error(f"Error computing query embedding: {e}")
            
    def _extract_score_from_response(self, 
                                     score_text):
        """Extract numerical score from LLM's evaluation response."""
        try:
            # Look for various score patterns
            patterns = [
                r'#\s*Evaluation Score:\s*(\d+)\s*/\s*100',  # "# Evaluation Score: XX/100"
                r'Evaluation Score:\s*(\d+)\s*/\s*100',  # "Evaluation Score: XX/100"
                r'Score:\s*(\d+)\s*/\s*100',  # "Score: XX/100"
                r'#\s*Score:\s*(\d+)\s*/\s*100',  # "# Score: XX/100"
                r'(\d+)\s*/\s*100',  # "XX/100"
                r'(\d+)\s*out of\s*100',  # "XX out of 100"
                r'#\s*Evaluation Score:\s*(\d+)',  # "# Evaluation Score: XX"
                r'Evaluation Score:\s*(\d+)',  # "Evaluation Score: XX"
                r'Score:\s*(\d+)',  # "Score: XX"
                r'#\s*Score:\s*(\d+)',  # "# Score: XX"
            ]
            
            # First try to find a fraction pattern
            for pattern in patterns:
                match = re.search(pattern, score_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    score = float(match.group(1))
                    # Ensure score is between 0 and 100
                    return max(0.0, min(100.0, score))
                    
            logging.warning(f"No score found in response: {score_text[:100]}...")
            return 0.0
        except Exception as e:
            logging.error(f"Error extracting score: {e}")
            return 0.0
            
    def cache_response(self,
                       db_path,
                       query,
                       response, 
                       quality_score = None,
                       quality_thresh = 80.0, 
                       cache_db = None):
        """Cache a query and its response if quality score meets threshold."""
        try:
            # Extract numerical score if quality_score is a string
            if isinstance(quality_score, str):
                quality_score = self._extract_score_from_response(quality_score)
                
            # Skip caching if quality score is below threshold
            if quality_score is not None:
                quality_score = float(quality_score)  # Ensure quality_score is float
                if quality_score < quality_thresh:
                    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                        logging.info(f"Skipping cache for query due to low quality score: {quality_score} < {quality_thresh}")
                    return
                
            # Use specified cache db or default to prompt_cache.db
            target_db = cache_db if cache_db else 'prompt_cache.db'
            
            # Initialize cache if it doesn't exist
            self.init_cache(target_db)
                
            conn = sqlite3.connect(target_db)
            cur = conn.cursor()
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Convert numpy array to bytes for storage
            embedding_bytes = query_embedding.tobytes()
            
            # Insert or replace existing entry
            cur.execute("""
                INSERT OR REPLACE INTO response_cache (query, query_embedding, response, quality_score)
                VALUES (?, ?, ?, ?)
            """, (query, embedding_bytes, response, quality_score))
            
            conn.commit()
            cur.close()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error caching response: {e}")
            raise
            
    def get_cached_response(self,
                            db_path,
                            query,
                            threshold, 
                            cache_db = None):
        """Retrieve a cached response if a similar query exists."""
        try:
            # Use specified cache db or default to prompt_cache.db
            target_db = cache_db if cache_db else 'prompt_cache.db'
            
            # Verify schema if using external cache
            if cache_db and not self.verify_cache_schema(cache_db):
                raise ValueError("External cache database has incorrect schema")
                
            return self._get_cached_response_from_db(target_db, query, threshold)
            
        except sqlite3.Error as e:
            logging.error(f"Error retrieving cached response: {e}")
            return None
            
    def _get_cached_response_from_db(self, 
                                     db_path,
                                     query,
                                     threshold):
        """Internal function to retrieve cached response from a specific database."""
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Get all cached embeddings
            cur.execute("SELECT query, query_embedding, response, quality_score FROM response_cache")
            cached_entries = cur.fetchall()
            
            if not cached_entries:
                return None
                
            # Find the most similar cached query
            best_similarity = -1
            best_response = None
            best_query = None
            
            for cached_query, cached_embedding_bytes, cached_response, quality_score in cached_entries:
                cached_embedding = np.frombuffer(cached_embedding_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = cached_response
                    best_query = cached_query
            
            # Return response if similarity exceeds threshold
            if best_similarity >= threshold:
                return best_response, best_similarity
                
            return None
            
        except sqlite3.Error as e:
            logging.error(f"Error retrieving cached response: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    #TODO - this needs fixing
    def clear_cache(self, db_path: str) -> None:
        """Clear all entries from the cache table by dropping it."""
        try:
            # Use prompt_cache.db by default
            cache_path = 'prompt_cache.db'
            if db_path:
                cache_path = db_path
                
            conn = sqlite3.connect(cache_path)
            cur = conn.cursor()
            
            # Drop the response_cache table
            cur.execute("DROP TABLE IF EXISTS response_cache")
            
            conn.commit()
            cur.close()
            conn.close()
            logging.info(f"Cache table dropped from {cache_path}")
        except sqlite3.Error as e:
            logging.error(f"Error clearing cache: {e}")
            raise 
