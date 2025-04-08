import os
import torch
from transformers import AutoTokenizer, AutoModel
import psycopg2
import sqlite3
import numpy as np
from itertools import islice
from tqdm import tqdm
from anthropic import Anthropic
from openai import OpenAI
import logging
import re

def get_ngrams(text, n):
    """Extract n-grams from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    
    words = text.split()
    if len(words) < n:
        return [text]
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def compute_embeddings(ngrams, model=None, tokenizer=None, device='cuda', embedding_type='transformer', llm_provider='anthropic'):
    """Compute embeddings for ngrams using transformers or LLM"""
    if embedding_type == 'transformer':
        if model is None or tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
            model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)
            
        embeddings = []
        for ngram in ngrams:
            inputs = tokenizer(ngram, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:,0,:].cpu().numpy())
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

def batch_insert_postgres(conn, table_name, batch_data):
    """Insert batch of data into postgres"""
    cur = conn.cursor()
    
    for data in batch_data:
        ngram, embedding, idx, filepath = data
        cur.execute(
            f"INSERT INTO {table_name} (embedding, ngram, filepath) VALUES (%s, %s, %s)",
            (embedding.tobytes(), ngram, filepath)
        )
    
    conn.commit()
    cur.close()

def batch_insert_sqlite(conn, table_name, batch_data):
    """Insert batch of data into sqlite"""
    cur = conn.cursor()
    
    for data in batch_data:
        ngram, embedding, idx, filepath = data
        embedding_list = embedding.tolist()
        cur.execute(
            f"INSERT INTO {table_name} (embedding, ngram, filepath) VALUES (?, ?, ?)",
            (str(embedding_list), ngram, filepath)
        )
    
    conn.commit()
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

def create_embeddings_table(conn, table_name, embedding_type):
    """Create table for storing embeddings"""
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding {embedding_type},
            ngram TEXT,
            filepath TEXT
        )
    """)
    conn.commit()
    cur.close()

def process_corpus(
    corpus_dir,
    tag_hierarchy,
    ngram_size=3,
    batch_size=32,
    db_type='postgres',
    db_params={
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost'
    },
    embedding_type='transformer',
    llm_provider='anthropic'
):
    """Process corpus and store embeddings in database"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)
    
    if db_type == 'postgres':
        conn = psycopg2.connect(**db_params)
        batch_insert = batch_insert_postgres
        param_placeholder = '%s'
        db_embedding_type = 'BYTEA'
    else:
        conn = sqlite3.connect(db_params.get('dbname', 'embeddings.db'))
        batch_insert = batch_insert_sqlite
        param_placeholder = '?'
        db_embedding_type = 'TEXT'

    all_files = get_all_files(corpus_dir)
    if not all_files:
        logging.warning(f"No files found in {corpus_dir}")
        return

    table_name = "document_embeddings"
    create_embeddings_table(conn, table_name, db_embedding_type)

    logging.info(f"Processing {len(all_files)} files with {ngram_size}-grams...")

    completed = 0
    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        for full_path, rel_path in all_files:
            try:
                logging.info(f"Computing {ngram_size}-gram embeddings for {rel_path}")
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Ensure ngram_size is an integer
                ngram_size = int(ngram_size)
                ngrams = get_ngrams(text, ngram_size)
                results = []
                
                for i in range(0, len(ngrams), batch_size):
                    batch = ngrams[i:i+batch_size]
                    embeddings = compute_embeddings(batch, model, tokenizer, device, embedding_type, llm_provider)
                    
                    for j, (ngram, embedding) in enumerate(zip(batch, embeddings)):
                        results.append((ngram, embedding, i+j, rel_path))
                        
                batch_insert(conn, table_name, results)
                completed += 1
                pbar.update(1)
                logging.info(f"Completed {completed}/{len(all_files)} files")
            except Exception as e:
                logging.error(f"Error processing {rel_path}: {e}")

    logging.info(f"Successfully processed {completed} files")
    conn.close()
