import os
import unittest
import tempfile
import shutil
import sys
import numpy as np
import sqlite3
import logging
from sklearn.datasets import fetch_20newsgroups

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cotarag.accelerag.managers import RAGManager
from cotarag.accelerag.embedders import TextEmbedder
from cotarag.accelerag.indexers import CentroidIndexer
from cotarag.accelerag.retrievers import CentroidRetriever

def generate_newsgroups_dataset(output_dir):
    """Generate 20 newsgroups dataset for testing.
    
    Args:
        output_dir: Directory to save the generated text files
    """
    # Load newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data
    labels = newsgroups.target
    target_names = newsgroups.target_names
    
    # Create directories for each category
    for category in target_names:
        category_dir = os.path.join(output_dir, category.replace(' ', '_'))
        os.makedirs(category_dir, exist_ok=True)
    
    # Save texts
    doc_count = 0
    for i, (text, label) in enumerate(zip(texts, labels)):
        # Skip empty documents
        if not text.strip():
            continue
            
        # Get category name
        category = target_names[label]
        category_dir = os.path.join(output_dir, category.replace(' ', '_'))
        
        # Save text file
        file_path = os.path.join(category_dir, f'doc_{doc_count}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        doc_count += 1
        
    print(f"Generated {doc_count} non-empty documents across {len(target_names)} categories")

class TestTextRAG(unittest.TestCase):
    """Test cases for text RAG functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        
        # Get API key from environment
        cls.api_key = os.environ.get("CLAUDE_API_KEY")
        if not cls.api_key:
            raise EnvironmentError("CLAUDE_API_KEY not found in environment variables")
        
        # Generate newsgroups dataset
        cls.newsgroups_dir = os.path.join(cls.test_dir, 'newsgroups_dataset')
        os.makedirs(cls.newsgroups_dir, exist_ok=True)
        generate_newsgroups_dataset(cls.newsgroups_dir)
        
        cls.db_path = os.path.join(cls.test_dir, 'test_embeddings.db.sqlite')
        
        # Initialize components
        cls.embedder = TextEmbedder(device='cpu')
        cls.indexer = CentroidIndexer(embedder=cls.embedder)
        
        # Index texts
        cls.indexer.index(
            corpus_dir=cls.newsgroups_dir,
            db_params={'dbname': cls.db_path}
        )
        
        # Initialize RAG manager (without caching)
        cls.rag = RAGManager(
            api_key=cls.api_key,
            dir_to_idx=cls.newsgroups_dir,
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
        """Test text indexing functionality."""
        print("\n=== Testing Text Indexing ===")
        
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
                expected_columns = {'id', 'embedding', 'label', 'metadata', 'filepath', 'content'}
                
            self.assertEqual(columns, expected_columns, f"Table {table} has incorrect schema")
                
        conn.close()
        
    def test_retrieval(self):
        """Test text retrieval functionality."""
        print("\n=== Testing Text Retrieval ===")
        
        # Initialize retriever
        retriever = CentroidRetriever(
            model_name='huawei-noah/TinyBERT_General_4L_312D',
            device='cpu',
            top_c=3,
            top_k=5
        )
        
        # Use a query that matches newsgroup discussion style
        query_text = """I'm trying to learn about 3D modeling and rendering techniques.
        Can someone explain the key differences between ray tracing and rasterization?
        I'm particularly interested in their performance characteristics and visual quality tradeoffs.
        Also, what hardware would I need for real-time ray tracing?"""
            
        print(f"\nQuery Text: {query_text}")
        
        # Perform retrieval
        results = retriever.retrieve(
            query=query_text,
            db_path=self.db_path,
            top_c=3,
            top_k=5
        )
        
        # Verify results
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertLessEqual(len(results), 5, "Should return at most 5 results")
        
        print("\nRetrieved Documents:")
        print("-" * 50)
        
        # Track which centroids we've shown
        shown_centroids = set()
        
        for i, result in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {result['content'][:200]}...")  # Print first 200 chars
            print(f"Similarity Score: {result['similarity']:.4f}")
            print(f"Source Table: {result['table']}")
            print(f"Filepath: {result['filepath']}")
            
            # Show centroid distances if we haven't shown this centroid yet
            if 'centroid_distances' in result and result['table'] not in shown_centroids:
                print("\nTop-C Centroid Distances:")
                for centroid, distance in result['centroid_distances'].items():
                    print(f"  {centroid}: {distance:.4f}")
                shown_centroids.add(result['table'])
            
        # Verify result structure
        for result in results:
            self.assertIn('content', result, "Result should contain content")
            self.assertIn('similarity', result, "Result should contain similarity score")
            self.assertIn('table', result, "Result should contain source table")
            self.assertIn('filepath', result, "Result should contain filepath")
            self.assertIn('query', result, "Result should contain original query")
            self.assertIn('retrieval_method', result, "Result should contain retrieval method")
            self.assertIn('centroid_distances', result, "Result should contain centroid distances")
            
        # Verify similarity scores are in descending order
        similarities = [r['similarity'] for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True), 
                        "Results should be sorted by similarity in descending order")

if __name__ == '__main__':
    unittest.main()
