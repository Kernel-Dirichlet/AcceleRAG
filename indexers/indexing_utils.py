import os
import torch
from transformers import AutoTokenizer, AutoModel
import sqlite3
import numpy as np
from itertools import islice
from tqdm import tqdm
from anthropic import Anthropic
from openai import OpenAI
import logging
import re
from typing import Dict, Optional, Union, Tuple

# Model configuration mapping
MODEL_CONFIGS = {
    # OpenAI models
    'gpt-4o': {'provider': 'openai', 'type': 'chat'},
    'gpt-4-turbo': {'provider': 'openai', 'type': 'chat'},
    'gpt-3.5-turbo': {'provider': 'openai', 'type': 'chat'},
    'text-embedding-3-small': {'provider': 'openai', 'type': 'embedding'},
    'text-embedding-3-large': {'provider': 'openai', 'type': 'embedding'},
    
    # Anthropic models
    'claude-3-7-sonnet-20250219': {'provider': 'anthropic', 'type': 'chat'},
    'claude-3-5-sonnet': {'provider': 'anthropic', 'type': 'chat'},
    'claude-3-5-haiku': {'provider': 'anthropic', 'type': 'chat'},
    
    # Default transformer models
    'huawei-noah/TinyBERT_General_4L_312D': {'provider': 'transformer', 'type': 'embedding'},
    'sentence-transformers/all-MiniLM-L6-v2': {'provider': 'transformer', 'type': 'embedding'},
    'BAAI/bge-large-en-v1.5': {'provider': 'transformer', 'type': 'embedding'},
}

# Default embedding model configurations
EMBEDDING_MODELS = {
    'transformer': {
        'huawei-noah/TinyBERT_General_4L_312D': {
            'max_length': 512,
            'pooling': 'cls',
        },
        'sentence-transformers/all-MiniLM-L6-v2': {
            'max_length': 384,
            'pooling': 'mean',
        },
        'BAAI/bge-large-en-v1.5': {
            'max_length': 512,
            'pooling': 'cls',
        },
    }
}

def get_model_provider(model_name):
    """Get provider and type information for a given model name.
    
    Args:
        model_name: Name of the model to look up
        
    Returns:
        Dictionary containing provider and type information
        
    Raises:
        ValueError: If model name is not found in configuration
    """
    if model_name not in MODEL_CONFIGS:
        # Check if it's a valid transformer model
        try:
            AutoTokenizer.from_pretrained(model_name)
            return {'provider': 'transformer', 'type': 'embedding'}
        except Exception:
            raise ValueError(f"Model {model_name} not found in configuration and is not a valid transformer model")
    return MODEL_CONFIGS[model_name]

def get_embedding_model_config(model_name):
    """Get configuration for a specific embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Dictionary containing model configuration
        
    Raises:
        ValueError: If model configuration is not found
    """
    # First check if it's in our default configurations
    for provider, models in EMBEDDING_MODELS.items():
        if model_name in models:
            return models[model_name]
            
    # If not found, try to get default configuration for transformer models
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {
            'max_length': tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512,
            'pooling': 'cls'  # Default pooling strategy
        }
    except Exception as e:
        raise ValueError(f"Could not determine configuration for model {model_name}: {e}")

def load_embedding_model(model_name, device='cpu'):
    """Load tokenizer and model for embedding generation.
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (tokenizer, model, config) or None for LLM models
    """
    try:
        model_config = get_model_provider(model_name)
        if model_config['provider'] == 'transformer':
            config = get_embedding_model_config(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            return tokenizer, model, config
        else:
            # For LLM models, we don't need to load anything
            return None, None, None
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise

def get_ngrams(text, n):
    """Extract ngrams from text using non-overlapping chunks of size n"""
    # Remove URLs and normalize whitespace
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Split text into words
    words = text.split()
    if len(words) < n:
        return [text]  # Return full text if shorter than ngram size
        
    # Create non-overlapping ngrams
    ngrams = []
    for i in range(0, len(words), n):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
        
    return ngrams

def compute_embeddings(ngrams, model_name='huawei-noah/TinyBERT_General_4L_312D', device='cpu'):
    """Compute embeddings for ngrams using specified model.
    
    Args:
        ngrams: List of text chunks to embed
        model_name: Name of the model to use
        device: Device to run model on
        
    Returns:
        Numpy array of embeddings
    """
    model_config = get_model_provider(model_name)
    provider = model_config['provider']
    
    if provider == 'transformer':
        tokenizer, model, config = load_embedding_model(model_name, device)
        embeddings = []
        
        for ngram in ngrams:
            inputs = tokenizer(
                ngram,
                return_tensors='pt',
                max_length=config['max_length'],
                truncation=True,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            if config['pooling'] == 'cls':
                embedding = outputs.last_hidden_state[:,0,:]
            elif config['pooling'] == 'mean':
                embedding = outputs.last_hidden_state.mean(dim=1)
                
            embeddings.append(embedding.cpu().numpy())
            
        return np.vstack(embeddings)
        
    elif provider in ['openai', 'anthropic']:
        embeddings = []
        
        if provider == 'anthropic':
            client = Anthropic()
            for ngram in ngrams:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=1000,
                    system="Return an embedding vector for the provided text.",
                    messages=[{"role": "user", "content": ngram}]
                )
                embeddings.append(np.array(response.content[0].text))
                
        elif provider == 'openai':
            client = OpenAI()
            for ngram in ngrams:
                response = client.embeddings.create(
                    model=model_name,
                    input=ngram
                )
                embeddings.append(np.array(response.data[0].embedding))
                
        return np.stack(embeddings)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def batch_insert_sqlite(conn, table_name, batch_data):
    """Insert batch of data into sqlite"""
    cur = conn.cursor()
    try:
        for data in batch_data:
            ngram, embedding, _, filepath = data
            embedding_list = embedding.tolist()
            cur.execute(
                f"INSERT INTO {table_name} (embedding, ngram, filepath) VALUES (?, ?, ?)",
                (str(embedding_list), ngram, filepath)
            )
        conn.commit()
        logging.info(f"Successfully inserted {len(batch_data)} records into {table_name}")
    except Exception as e:
        logging.error(f"Error inserting into {table_name}: {e}")
        conn.rollback()
    finally:
        cur.close()

def get_all_files(directory):
    """Get all files in directory recursively"""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, directory)
            all_files.append((full_path, rel_path))
    return all_files

def create_embeddings_table(conn, table_name):
    """Create table for storing embeddings"""
    cur = conn.cursor()
    try:
        # Drop table if it exists to ensure clean state
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table with proper schema
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding TEXT,
                ngram TEXT,
                filepath TEXT
            )
        """)
        conn.commit()
        logging.info(f"Successfully created table: {table_name}")
    except Exception as e:
        logging.error(f"Error creating table {table_name}: {e}")
        conn.rollback()
    finally:
        cur.close()

def sanitize_table_name(name):
    """Sanitize table name by replacing invalid characters with underscores"""
    # Replace dots, spaces, and other special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure the name starts with a letter or underscore
    if not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = '_' + sanitized
    return sanitized

def get_tag_from_path(rel_path):
    """Extract tag from file path based on tag hierarchy"""
    path_parts = rel_path.split(os.sep)
    if len(path_parts) < 2:  # Need at least a directory and a file
        return None
        
    # The tag is the directory path
    tag = '/'.join(path_parts[:-1])
    return tag

def process_corpus(
    corpus_dir,
    tag_hierarchy=None,
    ngram_size=3,
    batch_size=32,
    db_params={
        'dbname': 'your_db'
    },
    model_name='huawei-noah/TinyBERT_General_4L_312D',
    device='cpu'
):
    """Process corpus and store embeddings in database.
    
    Args:
        corpus_dir: Directory containing documents
        tag_hierarchy: Optional tag hierarchy for organizing documents
        ngram_size: Size of ngrams to extract
        batch_size: Batch size for processing
        db_params: Database parameters
        model_name: Name of the model to use for embeddings
        device: Device to run model on
    """
    try:
        # Get model configuration
        model_config = get_model_provider(model_name)
        provider = model_config['provider']
        
        # Initialize model based on provider
        if provider == 'transformer':
            tokenizer, model, config = load_embedding_model(model_name, device)
        else:
            tokenizer = model = config = None
            
        # Ensure SQLite database is created in the correct location
        db_path = db_params.get('dbname', 'embeddings.db.sqlite')
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        print(f"Creating SQLite database at: {db_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        batch_insert = batch_insert_sqlite
        logging.info(f"Using SQLite database at: {db_path}")

        all_files = get_all_files(corpus_dir)
        if not all_files:
            logging.warning(f"No files found in {corpus_dir}")
            return

        # Create a default table if no tag hierarchy is provided
        if not tag_hierarchy:
            table_name = 'documents'
            create_embeddings_table(conn, table_name)
            created_tables = {table_name}
        else:
            # Create tables for each tag
            created_tables = set()
            for full_path, rel_path in all_files:
                tag = get_tag_from_path(rel_path, tag_hierarchy)
                if tag:
                    table_name = sanitize_table_name(tag)
                    if table_name not in created_tables:
                        create_embeddings_table(conn, table_name)
                        created_tables.add(table_name)
                        logging.info(f"Created table for tag: {table_name}")

        logging.info(f"Processing {len(all_files)} files with {ngram_size}-grams...")
        logging.info(f"Using model: {model_name} ({provider})")

        completed = 0
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for full_path, rel_path in all_files:
                try:
                    logging.info(f"Computing {ngram_size}-gram embeddings for {rel_path}")
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    paragraphs = text.split('\n\n')
                    results = []
                    
                    for paragraph in paragraphs:
                        if not paragraph.strip():
                            continue
                            
                        ngrams = get_ngrams(paragraph, ngram_size)
                        if not ngrams:
                            continue
                            
                        # Process ngrams in batches
                        for i in range(0, len(ngrams), batch_size):
                            batch = ngrams[i:i+batch_size]
                            embeddings = compute_embeddings(
                                batch,
                                model_name=model_name,
                                device=device
                            )
                            
                            for ngram, embedding in zip(batch, embeddings):
                                results.append((ngram, embedding, 0, rel_path))
                    
                    if tag_hierarchy:
                        tag = get_tag_from_path(rel_path, tag_hierarchy)
                        if tag:
                            table_name = sanitize_table_name(tag)
                            batch_insert(conn, table_name, results)
                    else:
                        # Use default table if no tag hierarchy
                        batch_insert(conn, 'documents', results)
                        
                    completed += 1
                    pbar.update(1)
                    logging.info(f"Completed {completed}/{len(all_files)} files")
                    
                except Exception as e:
                    logging.error(f"Error processing {rel_path}: {e}")

        logging.info(f"Successfully processed {completed} files")
        
        # Verify tables were created and populated
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        logging.info(f"Final tables in database: {[t[0] for t in tables]}")
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cur.fetchone()[0]
            logging.info(f"Table {table[0]} has {count} records")
        cur.close()

    except Exception as e:
        logging.error(f"Error in process_corpus: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            logging.info("Database connection closed") 
