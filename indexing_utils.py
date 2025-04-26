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

def compute_embeddings(ngrams, model=None, tokenizer=None, device='cpu', embedding_type='transformer', llm_provider='anthropic'):
    """Compute embeddings for ngrams using transformers or LLM"""
    if embedding_type == 'transformer':
        if model is None or tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
                model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise
            
        embeddings = []
        for ngram in ngrams:
            try:
                inputs = tokenizer(ngram, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:,0,:].cpu().numpy())
            except Exception as e:
                logging.error(f"Error computing embedding: {e}")
                raise
        return np.vstack(embeddings)
        
    elif embedding_type == 'llm':
        embeddings = []
        
        if llm_provider == 'anthropic':
            client = Anthropic()
            for ngram in ngrams:
                response = client.messages.create(
                    model="claude-3-sonnet-20240320",
                    max_tokens=1000,
                    system="Return an embedding vector for the provided text.",
                    messages=[{"role": "user", "content": ngram}]
                )
                embeddings.append(np.array(response.content[0].text))
                
        elif llm_provider == 'openai':
            client = OpenAI()
            for ngram in ngrams:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=ngram
                )
                embeddings.append(np.array(response.data[0].embedding))
                
        else:
            raise ValueError("llm_provider must be 'anthropic' or 'openai'")
            
        return np.stack(embeddings)
    
    else:
        raise ValueError("embedding_type must be 'transformer' or 'llm'")

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

def get_tag_from_path(rel_path, tag_hierarchy):
    """Extract tag from file path based on tag hierarchy"""
    path_parts = rel_path.split(os.sep)
    if len(path_parts) < 2:  # Need at least a directory and a file
        return None
        
    # The tag is the directory path
    tag = '/'.join(path_parts[:-1])
    return tag

def process_corpus(
    corpus_dir,
    tag_hierarchy,
    ngram_size=3,
    batch_size=32,
    db_params={
        'dbname': 'your_db'
    },
    embedding_type='transformer',
    llm_provider='anthropic'
):
    """Process corpus and store embeddings in database"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)
    
    try:
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

        # Create tables for each tag
        created_tables = set()
        for full_path, rel_path in all_files:
            tag = get_tag_from_path(rel_path, tag_hierarchy)
            if tag:
                # Sanitize the table name
                table_name = sanitize_table_name(tag)
                if table_name not in created_tables:
                    create_embeddings_table(conn, table_name)
                    created_tables.add(table_name)
                    logging.info(f"Created table for tag: {table_name} (original: {tag})")

        logging.info(f"Processing {len(all_files)} files with {ngram_size}-grams...")
        logging.info(f"Created tables: {created_tables}")

        completed = 0
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for full_path, rel_path in all_files:
                try:
                    logging.info(f"Computing {ngram_size}-gram embeddings for {rel_path}")
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Split text into paragraphs and process each paragraph
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
                            embeddings = compute_embeddings(batch, model, tokenizer, device, embedding_type, llm_provider)
                            
                            for ngram, embedding in zip(batch, embeddings):
                                results.append((ngram, embedding, 0, rel_path))
                    
                    tag = get_tag_from_path(rel_path, tag_hierarchy)
                    if tag:
                        table_name = sanitize_table_name(tag)
                        batch_insert(conn, table_name, results)
                        completed += 1
                        pbar.update(1)
                        logging.info(f"Completed {completed}/{len(all_files)} files")
                    else:
                        logging.warning(f"No matching tag found for {rel_path}")
                    
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
