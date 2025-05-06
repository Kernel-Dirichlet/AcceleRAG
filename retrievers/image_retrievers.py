import os
import logging
import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from base_classes import Retriever
from embedders.image_embedders import ImageEmbedder

class ImageRetriever(Retriever):
    """Retriever for finding similar images using embeddings."""
    def __init__(self, dir_to_idx: Optional[str] = None, embedder: Optional[ImageEmbedder] = None):
        """Initialize the image retriever.
        
        Args:
            dir_to_idx: Path to directory containing images
            embedder: Embedder instance to use for computing embeddings
        """
        # Initialize database path
        if dir_to_idx:
            dir_name = os.path.basename(os.path.normpath(dir_to_idx))
            self.db_path = f"{dir_name}_image_embeddings.db.sqlite"
        else:
            self.db_path = "image_embeddings.db.sqlite"
            
        self.db_params = {'dbname': self.db_path}
        self.embedder = embedder or ImageEmbedder()
        super().__init__(self.db_params, embedder=self.embedder)
        
    def _compute_cosine_similarity(self, query_embedding: np.ndarray, image_embedding: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(query_embedding, image_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(image_embedding)
        )
        
    def _get_table_images(self, table: str, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Get top-k images from a table based on embedding similarity.
        
        Args:
            table: Table name
            query_embedding: Query embedding vector
            k: Number of images to retrieve
            
        Returns:
            List of (image_path, similarity) tuples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Get all images and embeddings
            cur.execute(f"SELECT image_path, embedding FROM {table}")
            results = cur.fetchall()
            
            if not results:
                return []
                
            # Compute similarities
            images_with_scores = []
            for image_path, embedding_bytes in results:
                try:
                    # Convert bytes to numpy array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    similarity = self._compute_cosine_similarity(query_embedding, embedding)
                    images_with_scores.append((image_path, similarity))
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")
                    continue
                    
            # Sort by similarity and return top k
            images_with_scores.sort(key=lambda x: x[1], reverse=True)
            return images_with_scores[:k]
            
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return []
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
                
    def retrieve(self, query_image: str, top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Retrieve similar images from the database.
        
        Args:
            query_image: Path to query image
            top_k: Number of images to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of (image_path, similarity_score) tuples
        """
        try:
            # 1. Compute query embedding
            query_embedding = self.embedder.embed(query_image)
            
            # 2. Get available tables
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            
            if not tables:
                logging.warning("No tables found in database")
                return []
                
            # 3. Get images from each table
            all_images = []
            for table in tables:
                images = self._get_table_images(table, query_embedding, top_k)
                all_images.extend(images)
                
            # 4. Sort all images by similarity and return top k
            all_images.sort(key=lambda x: x[1], reverse=True)
            return all_images[:top_k]
            
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []
            
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_path='{self.db_path}')" 
