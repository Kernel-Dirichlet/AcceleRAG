import os
import unittest
import tempfile
import shutil
import json
import sqlite3
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
import sys
import logging  # Added logging module
sys.path.append('../')
from indexing_utils import get_ngrams, compute_embeddings, get_all_files

class TestBase(unittest.TestCase):
    """Base test class with common setup and teardown"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.mock_data_dir = os.path.join(os.path.dirname(__file__), 'mock_data')
        shutil.copytree(self.mock_data_dir, os.path.join(self.test_dir, 'mock_data'))
        self.db_path = os.path.join(self.test_dir, 'test_embeddings.db')
        
        # Initialize model and tokenizer
        self.model_name = 'prajjwal1/bert-tiny'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        # Force model to CPU
        self.model = self.model.to('cpu')
        
        # Define tag hierarchy
        self.tag_hierarchy = {
            'cats': {},
            'dogs': {},
            'birds': {}
        }

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

class TestNgramGeneration(TestBase):
    """Test n-gram generation functionality"""
    
    def test_ngram_generation(self):
        """Test basic n-gram generation"""
        text = "This is a test sentence for n-gram generation"
        ngrams = get_ngrams(text, n=3)
        # For text with 8 words and non-overlapping 3-grams:
        # "This is a", "test sentence for", "n-gram generation"
        self.assertEqual(len(ngrams), 3)
        self.assertEqual(ngrams[0], "This is a")
        self.assertEqual(ngrams[1], "test sentence for")
        self.assertEqual(ngrams[2], "n-gram generation")
        logging.info("Test for basic n-gram generation passed")

    def test_ngram_edge_cases(self):
        """Test n-gram generation with edge cases"""
        # Test with text shorter than n
        short_text = "Short text"
        ngrams = get_ngrams(short_text, n=5)
        self.assertEqual(len(ngrams), 1)
        self.assertEqual(ngrams[0], short_text)
        
        # Test with empty text
        empty_text = ""
        ngrams = get_ngrams(empty_text, n=3)
        self.assertEqual(len(ngrams), 1)
        self.assertEqual(ngrams[0], empty_text)
        
        # Test with text length not divisible by n
        text = "This is a test sentence"
        ngrams = get_ngrams(text, n=3)
        # Should get: "This is a", "test sentence"
        self.assertEqual(len(ngrams), 2)
        self.assertEqual(ngrams[0], "This is a")
        self.assertEqual(ngrams[1], "test sentence")
        logging.info("Test for n-gram edge cases passed")

class TestEmbeddingComputation(TestBase):
    """Test embedding computation functionality"""
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions"""
        text = "Test sentence for embedding computation"
        ngrams = get_ngrams(text, n=3)
        embeddings = compute_embeddings(ngrams, self.model, self.tokenizer, device='cpu')
        
        # Check embedding dimensions
        self.assertEqual(embeddings.shape[0], len(ngrams))
        self.assertEqual(embeddings.shape[1], 128)  # BERT-tiny embedding size
        logging.info("Test for embedding dimensions passed")  # Logging test pass

    def test_embedding_consistency(self):
        """Test that same text produces same embeddings"""
        text = "Test sentence for consistency"
        ngrams = get_ngrams(text, n=3)
        
        # Compute embeddings twice
        embeddings1 = compute_embeddings(ngrams, self.model, self.tokenizer, device='cpu')
        embeddings2 = compute_embeddings(ngrams, self.model, self.tokenizer, device='cpu')
        
        # Check that embeddings are identical
        np.testing.assert_array_equal(embeddings1, embeddings2)
        logging.info("Test for embedding consistency passed")  # Logging test pass

class TestDatabaseOperations(TestBase):
    """Test database operations"""
    
    def setUp(self):
        """Additional setup for database tests"""
        super().setUp()
        self.create_database_schema()
    
    def create_database_schema(self):
        """Create test database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata TEXT NOT NULL,
            tag_path TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()

    def test_document_indexing(self):
        """Test document indexing and storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Process a single test document
        test_file = os.path.join(self.test_dir, 'mock_data', 'dogs', 'facts.txt')
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ngrams = get_ngrams(content, n=3)
        embeddings = compute_embeddings(ngrams, self.model, self.tokenizer, device='cpu')
        
        # Store in database
        for ngram, embedding in zip(ngrams, embeddings):
            metadata = {
                'source_file': 'facts.txt',
                'tag_path': 'dogs',
                'ngram_size': 3,
                'model': self.model_name
            }
            
            cursor.execute('''
            INSERT INTO embeddings (text, embedding, metadata, tag_path)
            VALUES (?, ?, ?, ?)
            ''', (
                ngram,
                embedding.tobytes(),
                json.dumps(metadata),
                'dogs'
            ))
        
        conn.commit()
        
        # Verify storage
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        total_embeddings = cursor.fetchone()[0]
        self.assertGreater(total_embeddings, 0)
        
        # Verify metadata structure
        cursor.execute('SELECT metadata FROM embeddings LIMIT 1')
        metadata = json.loads(cursor.fetchone()[0])
        self.assertIn('source_file', metadata)
        self.assertIn('tag_path', metadata)
        self.assertIn('ngram_size', metadata)
        self.assertIn('model', metadata)
        logging.info("Test for document indexing and storage passed")  # Logging test pass
        
        conn.close()

    def test_tag_hierarchy_consistency(self):
        """Test tag hierarchy consistency"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert test data with valid tags
        test_data = [
            ('test text 1', b'dummy_embedding', '{"tag": "dogs"}', 'dogs'),
            ('test text 2', b'dummy_embedding', '{"tag": "cats"}', 'cats')
        ]
        
        cursor.executemany('''
        INSERT INTO embeddings (text, embedding, metadata, tag_path)
        VALUES (?, ?, ?, ?)
        ''', test_data)
        
        conn.commit()
        
        # Verify all tags are valid
        cursor.execute('SELECT DISTINCT tag_path FROM embeddings')
        tag_paths = [row[0] for row in cursor.fetchall()]
        for tag_path in tag_paths:
            self.assertIn(tag_path, self.tag_hierarchy)
        logging.info("Test for tag hierarchy consistency passed")  # Logging test pass
        
        conn.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configure logging
    unittest.main()  # End of Selection
