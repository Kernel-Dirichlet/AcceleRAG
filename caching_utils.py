import sqlite3
import numpy as np
from typing import Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from indexing_utils import compute_embeddings
import torch
import re

def init_cache(db_path: str) -> None:
    """Initialize the cache table in the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Drop existing table if it exists to ensure clean schema
        cur.execute("DROP TABLE IF EXISTS response_cache")
        
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
        logging.info("Cache table initialized with correct schema")
    except sqlite3.Error as e:
        logging.error(f"Error initializing cache: {e}")
        raise

def get_query_embedding(query: str) -> np.ndarray:
    """Get embedding for a query using the compute_embeddings function."""
    # Use compute_embeddings with transformer type and TinyBERT model
    # Force CPU device to avoid CUDA issues
    embeddings = compute_embeddings([query], embedding_type='transformer', device='cpu')
    return embeddings[0]  # Return first (and only) embedding

def extract_score_from_response(score_text: str) -> float:
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

def cache_response(db_path: str, query: str, response: str, quality_score: str = None, quality_thresh: float = 80.0) -> None:
    """Cache a query and its response if quality score meets threshold."""
    try:
        # Extract numerical score if quality_score is a string
        if isinstance(quality_score, str):
            quality_score = extract_score_from_response(quality_score)
            
        # Skip caching if quality score is below threshold
        if quality_score is not None:
            quality_score = float(quality_score)  # Ensure quality_score is float
            if quality_score < quality_thresh:
                if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                    logging.info(f"Skipping cache for query due to low quality score: {quality_score} < {quality_thresh}")
                return
            
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get query embedding
        query_embedding = get_query_embedding(query)
        
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

def get_cached_response(db_path: str, query: str, threshold: float) -> Optional[Tuple[str, float]]:
    """Retrieve a cached response if a similar query exists."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get query embedding
        query_embedding = get_query_embedding(query)
        
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
