import os
import unittest
import tempfile
import shutil
import sys
import json
import sqlite3
from dotenv import load_dotenv

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
        cls.api_key = os.environ.get("CLAUDE_API_KEY")
        # Copy arxiv_mini to test directory to avoid reindexing prompt
        cls.arxiv_dir = os.path.join(cls.test_dir, 'arxiv_mini')
        shutil.copytree(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'arxiv_mini'), cls.arxiv_dir)
        
        cls.db_path = os.path.join(cls.test_dir, 'test_embeddings.db.sqlite')
        
        # Create tag hierarchy from directory structure
        cls.tag_hierarchy = create_tag_hierarchy(cls.arxiv_dir)
        
        # Remove existing cache database if it exists
        cache_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt_cache.db')
        if os.path.exists(cache_db_path):
            os.remove(cache_db_path)
            
        # Initialize cache using DefaultCache
        cache = DefaultCache()
        cache.init_cache(cache_db_path)
        
        # Initialize RAG manager with explicit cache settings
        cls.rag = RAGManager(
            api_key=cls.api_key,
            dir_to_idx=cls.arxiv_dir,
            grounding='soft',
            enable_cache=True,  # Enable cache writing
            use_cache=True,     # Enable cache reading
            cache_thresh=0.9,   # Set similarity threshold
            logging_enabled=True,  # Enable logging to debug cache issues
            force_reindex=True,
            cache_db=cache_db_path,  # Explicitly set cache database
            template_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web_rag_template.txt'),  # Pass template path
            hard_grounding_prompt=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'hard_grounding_prompt.txt'),  # Pass hard grounding prompt path
            soft_grounding_prompt=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'soft_grounding_prompt.txt')  # Pass soft grounding prompt path
        )
        
        # Set database path
        cls.rag.retriever.db_path = cls.db_path
        
        # Index documents once for all tests
        cls.rag.index(db_params={'dbname': cls.db_path})
        
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
        
        # First query
        query = "what can we infer about database design from this context"
        print(f"\nQuery: {query}")
        print(f"Grounding Mode: {self.rag.grounding}")
        
        # Generate first response
        first_response = self.rag.generate_response(query)
        print("\nFirst Response:")
        print("-" * 50)
        print(first_response)
        
        # Verify cache entry was created
        cache_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt_cache.db')
        conn = sqlite3.connect(cache_db_path)
        cur = conn.cursor()
        
        # Check cache entry
        cur.execute("SELECT query, response, quality_score FROM response_cache")
        result = cur.fetchone()
        
        self.assertIsNotNone(result, "No cache entry found after first query")
        cached_query, cached_response, quality_score = result
        
        print("\nCache Information:")
        print("-" * 50)
        print(f"Cache Hit Quality Score: {quality_score}")
        
        # Verify quality score is above threshold
        self.assertGreaterEqual(float(quality_score), 80.0, "Cache hit quality score is below threshold")
        
        # Second query (should be cached)
        print("\nSecond Query (Cached):")
        second_response = self.rag.generate_response(query)
        
        # Verify cache hit
        cur.execute("SELECT query, response, quality_score FROM response_cache")
        result = cur.fetchone()
        self.assertIsNotNone(result, "No cache entry found after second query")
        
        conn.close()

if __name__ == '__main__':
    unittest.main() 

