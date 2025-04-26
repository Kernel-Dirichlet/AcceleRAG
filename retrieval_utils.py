import torch
from transformers import AutoTokenizer, AutoModel
import psycopg2
import sqlite3
import numpy as np
from query_utils import route_query
from arxiv_category_mapper import get_category_description, get_related_categories, suggest_tables_for_query
from anthropic import Anthropic
import os
from openai import OpenAI

def compute_query_embedding(query, embedding_type='transformer', model_name='huawei-noah/TinyBERT_General_4L_312D', llm_provider='anthropic', max_length=512):
    """Compute embedding for query using transformer or LLM"""
    if embedding_type == 'transformer':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        inputs = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].cpu().numpy()[0]
        
    elif embedding_type == 'llm':
        if llm_provider == 'anthropic':
            client = Anthropic()
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Generate an embedding for: {query[:max_length]}"
                }]
            )
            return np.array(response.content[0].text)
            
        elif llm_provider == 'openai':
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query[:max_length]
            )
            return np.array(response.data[0].embedding)
            
        else:
            raise ValueError("llm_provider must be 'anthropic' or 'openai'")
            
    else:
        raise ValueError("embedding_type must be 'transformer' or 'llm'")

def fetch_top_k(query, db_params, tag_hierarchy, k=3, debug=False, max_length=512):
    """Fetch top-k most relevant chunks for query using semantic search"""
    # Route query to get relevant tags
    tags = route_query(query, tag_hierarchy)
    if not tags:
        if debug:
            print("No relevant tags found for query.")
        return []
        
    if debug:
        print(f"Found relevant tags: {tags}")
    
    # Compute query embedding
    query_embedding = compute_query_embedding(query, max_length=max_length)
    
    try:
        # Connect to database
        if 'dbname' in db_params:
            db_path = db_params['dbname']
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(db_path)
            if debug:
                print(f"Connecting to database at: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Get all available tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            available_tables = [row[0] for row in cur.fetchall()]
            # Filter out system tables
            available_tables = [t for t in available_tables if not t.startswith('sqlite_')]
            if debug:
                print(f"Available tables: {available_tables}")
            
            # Find matching tables based on tags
            matching_tables = []
            for tag in tags:
                # If tag is already a table name, use it directly
                if tag in available_tables:
                    matching_tables.append(tag)
                    continue
                    
                # Try different variations of the table name based on whether it's an ArXiv category
                if '.' in tag:  # ArXiv category
                    possible_names = [
                        tag.replace('.', '_'),  # Replace dots with underscores
                        tag.split('.')[-1],  # Last component
                        '_'.join(tag.split('.')[:-1]),  # Without last component
                        '_'.join(tag.split('.')[-2:])  # Last two components
                    ]
                else:  # Regular tag
                    possible_names = [
                        tag,  # Original tag
                        tag.split('/')[-1],  # Last component
                        '_'.join(tag.split('/')[:-1]),  # Without last component
                        '_'.join(tag.split('/')[-2:])  # Last two components
                    ]
                
                # Check each possible name
                for name in possible_names:
                    if name in available_tables:
                        matching_tables.append(name)
                        if debug:
                            print(f"Found matching table: {name} for tag: {tag}")
                        break
                else:
                    if debug:
                        print(f"Warning: No matching table found for tag: {tag}")
                        print(f"Tried names: {possible_names}")
                        print(f"Available tables: {available_tables}")
            
            if not matching_tables:
                if debug:
                    print(f"No matching tables found for tags: {tags}")
                return []
            
            # Build query to search across all matching tables
            queries = []
            params = []
            for table in matching_tables:
                queries.append(f"""
                    SELECT ngram, filepath, embedding
                    FROM {table}
                    ORDER BY id
                    LIMIT ?
                """)
                params.append(k)
            
            # Execute queries and collect results
            all_results = []
            for query, limit, table_name in zip(queries, params, matching_tables):
                cur.execute(query, (limit,))
                results = cur.fetchall()
                if results:
                    all_results.extend(results)
                    if debug:
                        print(f"Found {len(results)} results in table {table_name}")
                else:
                    if debug:
                        print(f"No results found in table {table_name}")
            
            cur.close()
            conn.close()
            
            if not all_results:
                if debug:
                    print("No matching documents found in database.")
                return []
            
            # Compute similarities and return top k
            chunks = []
            seen_chunks = set()
            
            for ngram, filepath, embedding in all_results:
                try:
                    if isinstance(embedding, str):
                        embedding = np.array(eval(embedding))
                    if not isinstance(embedding, np.ndarray):
                        if debug:
                            print(f"Warning: Invalid embedding format for ngram: {ngram}")
                        continue
                        
                    # Normalize the chunk text for comparison
                    normalized_chunk = ' '.join(ngram.split())
                    
                    # Skip if we've seen this chunk before
                    if normalized_chunk in seen_chunks:
                        continue
                    seen_chunks.add(normalized_chunk)
                    
                    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                    chunks.append((ngram, similarity, filepath))
                except Exception as e:
                    if debug:
                        print(f"Error computing similarity for ngram: {ngram}, error: {e}")
                    continue
                
            # Sort by similarity and return top k unique chunks
            chunks.sort(key=lambda x: x[1], reverse=True)
            return [chunk[0] for chunk in chunks[:k]]
            
        else:
            if debug:
                print("No database name provided in db_params")
            return []
            
    except sqlite3.Error as e:
        if debug:
            print(f"SQLite error: {str(e)}")
        return []
    except Exception as e:
        if debug:
            print(f"Error fetching chunks: {str(e)}")
        return []

