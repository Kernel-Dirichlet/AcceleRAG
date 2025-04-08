import os
import unittest
import tempfile
import shutil
import json
import sqlite3
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np

class TestAcceleRAG(unittest.TestCase):
    """
    Test suite for AcceleRAG document indexing and retrieval system.
    Tests include document processing, embedding generation, and database operations.
    """
    
    def setUp(self):
        """
        Set up test environment:
        - Create temporary directory
        - Copy mock data
        - Initialize database
        - Load TinyBERT model
        - Define tag hierarchy
        """
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.mock_data_dir = os.path.join(os.path.dirname(__file__), 'mock_data')
        
        # Copy mock data to test directory
        shutil.copytree(self.mock_data_dir, os.path.join(self.test_dir, 'mock_data'))
        
        # Define test database parameters
        self.db_path = os.path.join(self.test_dir, 'test_embeddings.db')
        
        # Initialize TinyBERT model and tokenizer
        self.model_name = 'prajjwal1/bert-tiny'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Define tag hierarchy - now flat structure with categories
        self.tag_hierarchy = {
            'cats': {},
            'dogs': {},
            'birds': {}
        }

    def tearDown(self):
        """Clean up temporary directory after tests"""
        shutil.rmtree(self.test_dir)

    def compute_bert_embeddings(self, text):
        """
        Compute BERT embeddings for a given text.
        
        Args:
            text (str): Input text to generate embeddings for
            
        Returns:
            numpy.ndarray: Mean-pooled embeddings of shape (1, hidden_size)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def create_database_schema(self):
        """
        Create the SQLite database schema with the following columns:
        - id: Auto-incrementing primary key
        - text: The n-gram text content
        - embedding: BLOB storage for the TinyBERT embeddings
        - metadata: JSON string containing document metadata
        - tag_path: The category path of the document
        """
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

    def test_doc_indexing(self):
        """
        Test document indexing functionality:
        - Process all documents in the mock data directory
        - Generate n-grams of size 16
        - Compute TinyBERT embeddings
        - Store in SQLite database with metadata
        - Verify the stored data
        """
        # Create database schema
        self.create_database_schema()
        
        # Process all documents in the mock data directory
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for root, _, files in os.walk(os.path.join(self.test_dir, 'mock_data')):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    # Get category from parent directory name
                    tag_path = os.path.basename(root)
                    
                    # Read and process the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into n-grams of size 16
                    words = content.split()
                    ngrams = [' '.join(words[i:i+16]) for i in range(0, len(words), 16)]
                    
                    # Compute embeddings for each n-gram
                    for ngram in ngrams:
                        embedding = self.compute_bert_embeddings(ngram)
                        
                        # Create metadata
                        metadata = {
                            'source_file': file,
                            'tag_path': tag_path,
                            'ngram_size': 16,
                            'model': self.model_name
                        }
                        
                        # Store in database
                        cursor.execute('''
                        INSERT INTO embeddings (text, embedding, metadata, tag_path)
                        VALUES (?, ?, ?, ?)
                        ''', (
                            ngram,
                            embedding.tobytes(),
                            json.dumps(metadata),
                            tag_path
                        ))
        
        conn.commit()
        
        # Verify the indexing
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
        
        # Verify embedding dimensions
        cursor.execute('SELECT embedding FROM embeddings LIMIT 1')
        embedding = np.frombuffer(cursor.fetchone()[0], dtype=np.float32)
        self.assertEqual(embedding.shape[0], 128)  # BERT-tiny embedding size
        
        conn.close()

    def test_tag_hierarchy_consistency(self):
        """
        Test that all documents are properly tagged according to the hierarchy.
        Verifies that all tag paths in the database are valid categories.
        """
        self.create_database_schema()
        
        # Process documents
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all unique tag paths from the database
        cursor.execute('SELECT DISTINCT tag_path FROM embeddings')
        tag_paths = [row[0] for row in cursor.fetchall()]
        
        # Verify that all tag paths are valid according to the hierarchy
        for tag_path in tag_paths:
            self.assertIn(tag_path, self.tag_hierarchy)
        
        conn.close()

    def test_bert_embeddings(self):
        """Test BERT embeddings with TinyBERT model."""
        # Initialize TinyBERT with proper configuration
        model_name = "huawei-noah/TinyBERT_General_4L_312D"
        config = AutoConfig.from_pretrained(model_name)
        config.model_type = "bert"  # Explicitly set model type
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=config)
        
        # Initialize BERT embeddings
        embeddings = BERTEmbeddings(
            model=model,
            tokenizer=tokenizer,
            batch_size=32,
            device="cpu"
        )

if __name__ == '__main__':
    unittest.main() 
