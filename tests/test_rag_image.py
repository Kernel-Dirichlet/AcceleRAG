import os
import unittest
import tempfile
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sqlite3
import warnings
from sklearn.datasets import load_digits

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accelerag.managers import RAGManager
from accelerag.embedders import ImageEmbedder
from accelerag.indexers import ImageIndexer
from accelerag.retrievers import ImageRetriever


def generate_digits_dataset(output_dir):
    """Generate MNIST digits dataset for testing.
    
    Args:
        output_dir: Directory to save the generated images
    """
    # Load digits dataset
    digits = load_digits()
    images = digits.images
    labels = digits.target
    
    # Create directories for each digit
    for digit in range(10):
        digit_dir = os.path.join(output_dir, f'digit_{digit}')
        os.makedirs(digit_dir, exist_ok=True)
    
    # Save images
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to PIL Image
        img = Image.fromarray((image * 16).astype(np.uint8))
        
        # Save to appropriate directory
        digit_dir = os.path.join(output_dir, f'digit_{label}')
        img_path = os.path.join(digit_dir, f'digit_{i}.png')
        img.save(img_path)

class TestImageRAG(unittest.TestCase):
    """Test cases for image RAG functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        
        # Get API key from environment
        cls.api_key = os.environ.get("CLAUDE_API_KEY")
        if not cls.api_key:
            raise EnvironmentError("CLAUDE_API_KEY not found in environment variables")
        
        # Generate digits dataset
        cls.digits_dir = os.path.join(cls.test_dir, 'digits_dataset')
        os.makedirs(cls.digits_dir, exist_ok=True)
        generate_digits_dataset(cls.digits_dir)
        
        cls.db_path = os.path.join(cls.test_dir, 'test_embeddings.db.sqlite')
        
        # Initialize components
        cls.embedder = ImageEmbedder(device='cpu')
        cls.indexer = ImageIndexer(embedder=cls.embedder)
        
        # Index images
        cls.indexer.index(
            corpus_dir=cls.digits_dir,
            db_params={'dbname': cls.db_path}
        )
        
        # Initialize RAG manager (without caching)
        cls.rag = RAGManager(
            api_key=cls.api_key,
            dir_to_idx=cls.digits_dir,
            grounding='soft',
            enable_cache=False,  # Disable caching
            use_cache=False,     # Disable caching
            logging_enabled=True,
            force_reindex=True,
            hard_grounding_prompt='prompts/hard_grounding_prompt.txt',
            soft_grounding_prompt='prompts/soft_grounding_prompt.txt'
        )
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        shutil.rmtree(cls.test_dir)
            
    def test_indexing(self):
        """Test image indexing functionality."""
        print("\n=== Testing Image Indexing ===")
        
        # Verify database exists and has tables
        self.assertTrue(os.path.exists(self.db_path))
        print(f"Database created at: {self.db_path}")
        
        # Verify tables have content
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cur.fetchall()]
        
        # Verify we have tables
        self.assertGreater(len(tables), 0, "No tables created during indexing")
        print("\nIndexing Statistics:")
        print("-" * 50)
        
        # Verify each table has content and correct schema
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, f"Table {table} is empty")
            print(f"Table {table}: {count} records")
            
            # Verify schema based on table type
            cur.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in cur.fetchall()}
            
            if table.endswith('_centroid'):
                # Centroid tables should have id and centroid columns
                expected_columns = {'id', 'centroid'}
            else:
                # Regular tables should have all columns
                expected_columns = {'id', 'embedding', 'label', 'metadata', 'filepath'}
                
            self.assertEqual(columns, expected_columns, f"Table {table} has incorrect schema")
                
        conn.close()
        
    def test_retrieval(self):
        """Test image retrieval functionality."""
        print("\n=== Testing Image Retrieval ===")
        
        # Get a query image from digit_9 directory
        nines_dir = os.path.abspath(os.path.join(self.digits_dir, 'digit_9'))
        if not os.path.exists(nines_dir):
            raise ValueError(f"Directory not found: {nines_dir}")
            
        # Get first image from directory
        query_image = os.path.join(nines_dir, os.listdir(nines_dir)[0])
        print(f"\nQuery Image: {query_image}")
        
        # Get query embedding
        query_embedding = self.embedder.embed(query_image)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get all centroid tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_centroid'")
        centroid_tables = [row[0] for row in cur.fetchall()]
        
        # Find nearest centroid
        nearest_centroid = None
        max_similarity = -1
        best_table = None
        
        for table in centroid_tables:
            cur.execute(f"SELECT centroid FROM {table}")
            for centroid_bytes, in cur.fetchall():
                centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
                similarity = np.dot(query_embedding, centroid) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(centroid)
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    nearest_centroid = centroid
                    best_table = table.replace('_centroid', '')
        
        print(f"\nNearest Centroid:")
        print(f"Table: {best_table}")
        print(f"Similarity: {max_similarity:.4f}")
        
        # Add warning if nearest centroid isn't from digit_9
        if best_table != 'digit_9':
            warnings.warn(f"Warning: Nearest centroid is from table '{best_table}' instead of 'digit_9'")
        
        # Get top-k images from the nearest table
        cur.execute(f"""
            SELECT filepath, embedding 
            FROM {best_table} 
            ORDER BY RANDOM() 
            LIMIT 5
        """)
        
        results = []
        for filepath, embedding_bytes in cur.fetchall():
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            full_path = os.path.abspath(os.path.join(self.digits_dir, filepath))
            results.append((full_path, similarity))
            
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Verify we got exactly 5 chunks
        self.assertEqual(len(results), 5, "Did not retrieve exactly 5 chunks")
        print("\nRetrieved Images:")
        print("-" * 50)
        
        # Create visualization
        plt.figure(figsize=(15, 3))
        plt.subplot(1, 6, 1)
        plt.imshow(Image.open(query_image))
        plt.title('Query Image')
        plt.axis('off')
        
        for i, (img_path, similarity) in enumerate(results, 1):
            print(f"\nImage {i}:")
            print(f"Path: {img_path}")
            print(f"Similarity Score: {similarity:.4f}")
            
            # Add to visualization
            plt.subplot(1, 6, i + 1)
            plt.imshow(Image.open(img_path))
            plt.title(f'Score: {similarity:.4f}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('retrieval_results.png')
        plt.close()
        
        conn.close()

if __name__ == '__main__':
    unittest.main() 

