import os
import unittest
import tempfile
import shutil
import sys
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers import RAGManager
from query_utils import create_tag_hierarchy

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
            enable_cache=True,  # Enable cache writing
            use_cache=True,     # Enable cache reading
            cache_thresh=0.9,   # Set similarity threshold
            logging_enabled=True,
            force_reindex=True,
            hard_grounding_prompt=os.path.join(project_root, 'prompts', 'hard_grounding_prompt.txt'),
            soft_grounding_prompt=os.path.join(project_root, 'prompts', 'soft_grounding_prompt.txt'),
            template_path=os.path.join(project_root, 'web_rag_template.txt')
        )
        
        # Set database path
        cls.rag.retriever.db_path = cls.db_path
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        shutil.rmtree(cls.test_dir)
        
    def test_rag_flow(self):
        """Test the complete RAG flow: indexing, retrieval, and caching."""
        print("\n" + "="*80)
        print("TESTING COMPLETE RAG FLOW")
        print("="*80)
        
        # Step 1: Index documents
        print("\n1. INDEXING DOCUMENTS")
        print("-"*50)
        self.rag.index(
            db_params={'dbname': self.db_path},
            ngram_size=32
        )
        
        # Step 2: First query - tests retrieval and caches response
        print("\n2. FIRST QUERY (TESTS RETRIEVAL)")
        print("-"*50)
        query = "what can we infer about database design from this context"
        print(f"Query: {query}")
        
        # Get and display retrieved chunks first
        chunks = self.rag.retrieve(query, top_k=5)
        print("\nRetrieved Chunks:")
        print("-"*50)
        for i, (chunk, similarity) in enumerate(chunks, 1):
            print(f"\nChunk {i} (Similarity: {similarity:.4f}):")
            print(f"{chunk[:200]}...")
        
        # Generate response
        first_response = self.rag.generate_response(query)
        print("\nGenerated Response:")
        print("-"*50)
        print(f"{first_response[:60]}...")
        # Verify cache entry exists with correct structure
        self.assertIn(query, self.rag.cache, "Query should be in cache")
        cache_data = self.rag.cache[query]
        self.assertIn('response', cache_data, "Cache should have response")
        self.assertIn('embedding', cache_data, "Cache should have embedding")
        self.assertIn('quality', cache_data, "Cache should have quality score")
        
        # Step 3: Exact match query - tests cache hit
        print("\n3. EXACT MATCH QUERY (TESTS CACHE)")
        exact_response = self.rag.generate_response(query)
        print("\nCached Response:")
        print(f"{exact_response[:60]}...")
        
        # Verify exact match
        self.maxDiff = None  # Show full diff
        self.assertEqual(first_response, exact_response, "Cached response does not match original response")
        
        # Step 4: Similar query - tests cache similarity
        print("\n4. SIMILAR QUERY (TESTS CACHE SIMILARITY)")
        print("-"*50)
        similar_query = "what can we learn about database design from this context"
        print(f"Query: {similar_query}")
        
        similar_response = self.rag.generate_response(similar_query)
        print("\nResponse:")
        print("-"*50)
        print(f"{similar_response[:100]}...")
         
        # Step 5: Check scorer
        print("\n5. VERIFYING SCORER")
        print("-"*50)
        score_result = self.rag.scorer.score_json(similar_response, similar_query, [])
        print(f"Quality Score: {score_result['quality_score']:.4f}")
        print("\nEvaluation:")
        print(score_result['evaluation'])

if __name__ == '__main__':
    unittest.main() 
