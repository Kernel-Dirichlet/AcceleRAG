import argparse
import logging
from query_utils import create_tag_hierarchy
from indexing_utils import process_corpus
import psycopg2
import sqlite3
import os
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Create tag hierarchy and database tables from directory structure')
    parser.add_argument('--dir', required=True, help='Directory containing the tag hierarchy')
    parser.add_argument('--ngram_size', type=int, default=64, help='Size of ngrams to generate (default: 64)')
    
    parser.add_argument('--db_type', choices=['postgres', 'sqlite'], default='sqlite',
                      help='Database type to use (default: sqlite)')
    
    parser.add_argument('--host', default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', default=5432, help='PostgreSQL port (default: 5432)')
    parser.add_argument('--user', default='postgres', help='PostgreSQL user (default: postgres)')
    parser.add_argument('--password', default='postgres', help='PostgreSQL password (default: postgres)')
    parser.add_argument('--dbname', default='postgres', help='Database name (default: postgres)')
    
    parser.add_argument('--sqlite_dbname', default='embeddings.db', help='SQLite database name (default: embeddings.db)')
    
    parser.add_argument('--embedding_type', choices=['transformer', 'llm'], default='transformer',
                      help='Type of embeddings to use (default: transformer)')
    parser.add_argument('--llm_provider', choices=['anthropic', 'openai'], default='anthropic',
                      help='LLM provider when using llm embeddings (default: anthropic)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing (default: 32)')
    
    args = parser.parse_args()

    if args.db_type == 'postgres':
        db_params = {
            'dbname': args.dbname,
            'user': args.user,
            'password': args.password,
            'host': args.host,
            'port': args.port
        }
    else:
        db_params = {
            'dbname': args.sqlite_dbname
        }

    logging.info(f"Creating tag hierarchy from directory: {args.dir}")
    tag_hierarchy = create_tag_hierarchy(args.dir)
    logging.info("Tag hierarchy created successfully")

    try:
        process_corpus(
            corpus_dir=args.dir,
            tag_hierarchy=tag_hierarchy,
            ngram_size=args.ngram_size,
            batch_size=args.batch_size,
            db_type=args.db_type,
            db_params=db_params,
            embedding_type=args.embedding_type,
            llm_provider=args.llm_provider
        )
        logging.info("Corpus processing completed successfully")
    except Exception as e:
        logging.error(f"Error processing corpus: {e}")

if __name__ == "__main__":
    main()

