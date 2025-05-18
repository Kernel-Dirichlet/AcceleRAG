import os
import unittest
import tempfile
import shutil
import sys
import json
import sqlite3

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers import RAGManager
from query_utils import create_tag_hierarchy
from cachers import DefaultCache

class TestRAGManager(unittest.TestCase):
    """Test cases for RAGManager functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        
        # Get API key from environment
        cls.api_key = os.environ.get("CLAUDE_API_KEY")
        if not cls.api_key:
            raise EnvironmentError("CLAUDE_API_KEY not found in environment variables")
        
        # Copy test data to temp directory
        cls.data_dir = os.path.join(cls.test_dir, 'test_data')
        test_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'arxiv_mini')
        shutil.copytree(test_data_path, cls.data_dir)
        
        cls.db_path = os.path.join(cls.test_dir, 'test_embeddings.db.sqlite')
        
        # Create tag hierarchy from directory structure
        cls.tag_hierarchy = create_tag_hierarchy(cls.data_dir)
        
        # Get project root for constructing prompt file paths
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # Initialize RAG manager with cache settings
        cls.rag = RAGManager(
            api_key=cls.api_key,
            dir_to_idx=cls.data_dir,
            grounding='soft',
            enable_cache=True,
            use_cache=True,
            cache_thresh=0.9,
            logging_enabled=True,
            force_reindex=True,
            cache_db=None,  # Let RAGManager create its own cache
            hard_grounding_prompt=os.path.join(project_root, 'prompts', 'hard_grounding_prompt.txt'),
            soft_grounding_prompt=os.path.join(project_root, 'prompts', 'soft_grounding_prompt.txt'),
            template_path=os.path.join(project_root, 'web_rag_template.txt')
        )
        
        # Set database path
        cls.rag.retriever.db_path = cls.db_path
        
        # Index documents with n-gram size 32
        cls.rag.index(
            db_params={'dbname': cls.db_path},
            ngram_size=32  # Set n-gram size for indexing
        )
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        shutil.rmtree(cls.test_dir)
        # Removed deletion of prompt_cache.db to avoid deleting the Claude API key
        # cache_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt_cache.db')
        # if os.path.exists(cache_db_path):
        #     os.remove(cache_db_path)
            
    def test_indexing(self):
        """Test document indexing functionality."""
        print("\n=== Testing Indexing ===")
        
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
        
        # Verify each table has content
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, f"Table {table} is empty")
            print(f"Table {table}: {count} records")
            
        conn.close()
        
    def test_retrieval(self):
        """Test document retrieval functionality."""
        print("\n=== Testing Retrieval ===")
        
        # Test retrieval with top_k = 5
        query = "database design patterns"
        print(f"\nQuery: {query}")
        print(f"Grounding Mode: {self.rag.grounding}")
        
        chunks = self.rag.retrieve(query, top_k = 5)
        
        # Verify we got exactly 3 chunks
        self.assertEqual(len(chunks), 5, "Did not retrieve exactly 5 chunks")
        print("\nRetrieved Chunks:")
        print("-" * 50)
        
        for i, (chunk, similarity) in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Content: {chunk}")
            print(f"Similarity Score: {similarity:.4f}")
            
    def test_caching(self):
        """Test response caching functionality."""
        print("\n=== Testing Caching ===")
        
        # First query - should compute and cache
        query = "what can we infer about database design from this context"
        print(f"\nFirst Query: {query}")
        first_response = self.rag.generate_response(query)
        print("\nFirst Response:")
        print("-" * 50)
        print(first_response)
        
        # Same query - should get cache hit
        print("\nSecond Query (Exact Match):")
        second_response = self.rag.generate_response(query)
        print(second_response)
        
        # Verify exact match cache hit
        self.assertEqual(first_response, second_response, "Cached response does not match original response")
        
        # Similar query - should get cache hit if within threshold
        similar_query = "what can we learn about database design from this context"
        print(f"\nThird Query (Similar): {similar_query}")
        similar_response = self.rag.generate_response(similar_query)
        print(similar_response)
        
        # Verify similar query cache hit
        cached_result = self.rag._get_cached_response(similar_query, self.rag.cache_thresh)
        self.assertIsNotNone(cached_result, "No cache hit for similar query")
        cached_response, similarity = cached_result
        print(f"\nCache Hit Similarity: {similarity:.4f}")
        
        # Verify similarity is above threshold
        self.assertGreaterEqual(similarity, self.rag.cache_thresh, "Similar query similarity below threshold")

if __name__ == '__main__':
    unittest.main() 
