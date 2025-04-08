import torch
from transformers import AutoTokenizer, AutoModel
import psycopg2
import sqlite3
import numpy as np
from query_utils import route_query
import anthropic
import os
from openai import OpenAI

def compute_query_embedding(query, embedding_type='transformer', model_name='prajjwal1/bert-tiny', llm_provider='anthropic'):
    """Compute embedding for query using transformer or LLM"""
    if embedding_type == 'transformer':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        inputs = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].cpu().numpy()[0]
        
    elif embedding_type == 'llm':
        if llm_provider == 'anthropic':
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-3-sonnet-20240320",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": f"Generate an embedding for: {query}"}]
                }]
            )
            return np.array(response.content[0].text)
            
        elif llm_provider == 'openai':
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return np.array(response.data[0].embedding)
            
        else:
            raise ValueError("llm_provider must be 'anthropic' or 'openai'")
            
    else:
        raise ValueError("embedding_type must be 'transformer' or 'llm'")

def fetch_top_k(query, db_params, tag_hierarchy, k=5):
    """Fetch top-k most relevant chunks for query using semantic search"""
    # Route query to get relevant tags
    tags = route_query(query, tag_hierarchy)
    if not tags:
        return []
        
    # Compute query embedding
    query_embedding = compute_query_embedding(query)
    
    # Connect to database
    if 'dbname' in db_params and isinstance(db_params['dbname'], str) and db_params['dbname'].endswith('.sqlite'):
        conn = sqlite3.connect(db_params['dbname'])
        cur = conn.cursor()
        
        # Get embeddings from SQLite
        cur.execute("""
            SELECT ngram, filepath, embedding
            FROM document_embeddings
            WHERE filepath LIKE ?
            ORDER BY ROWID
            LIMIT ?
        """, (f"%{tags[0]}%", k))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
    else:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Get embeddings from PostgreSQL
        cur.execute("""
            SELECT ngram, filepath, embedding
            FROM document_embeddings
            WHERE filepath LIKE %s
            ORDER BY id
            LIMIT %s
        """, (f"%{tags[0]}%", k))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
    
    # Compute similarities and return top k
    chunks = []
    for ngram, filepath, embedding in results:
        if isinstance(embedding, str):
            embedding = np.array(eval(embedding))
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        chunks.append((ngram, similarity))
        
    chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk[0] for chunk in chunks[:k]]
