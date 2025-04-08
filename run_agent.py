import argparse
import logging
from datetime import datetime
from retrieval_utils import fetch_top_k
import json
import anthropic
import os
from openai import OpenAI
from web_search_utils import search_web, score_chunks, interactive_web_rag
import sqlite3

# Set up logging
logging.basicConfig(
    filename='responses.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_db_rag(llm_provider, db_params, score=False):
    """Interactive RAG agent with database retrieval"""
    
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError("llm_provider must be 'anthropic' or 'openai'")
        
    # Connect to SQLite database
    db_path = db_params.get('dbname', 'embeddings.db.sqlite')
    conn = sqlite3.connect(db_path)
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        # Retrieve relevant chunks from database
        tag_hierarchy = {}
        with open('tag_hierarchy.json', 'r') as f:
            tag_hierarchy = json.loads(f.read().strip())
            
        chunks = fetch_top_k(query, db_params, k=5, tag_hierarchy=tag_hierarchy)
        context = "\n\n".join(chunks)
        
        if score:
            chunk_scores, hall_risk = score_chunks(context, llm_provider)
            score_msg = f"\nChunk Scores: {chunk_scores}\nHallucination Risk: {hall_risk}%"
            print(score_msg)
            logging.info(f"Query: {query}{score_msg}")
            
        # Format prompt with retrieved context
        with open('web_rag_template.txt', 'r') as f:
            template = f.read().strip()
            prompt = template.format(
                context=context.strip(),
                query=query
            )

def main():
    parser = argparse.ArgumentParser(description='Run query agent')
    parser.add_argument('--mode', choices=['interactive'], default='interactive',
                      help='Mode to run the agent in')
    parser.add_argument('--score', action='store_true',
                      help='Enable scoring of retrieved chunks')
    parser.add_argument('--llm_provider', choices=['anthropic', 'openai'], default='anthropic',
                      help='LLM provider to use (default: anthropic)')
    parser.add_argument('--rag_mode', choices=['db', 'agentic'], default='db',
                      help='RAG mode to use (default: db)')
    parser.add_argument('--api_key_path', required=True,
                      help='Path to file containing API key')
    
    # Database connection arguments
    parser.add_argument('--host', default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', default=5432, help='PostgreSQL port (default: 5432)')
    parser.add_argument('--user', default='postgres', help='PostgreSQL user (default: postgres)')
    parser.add_argument('--password', default='postgres', help='PostgreSQL password (default: postgres)')
    parser.add_argument('--dbname', default='embeddings.db.sqlite', help='Database name (default: embeddings.db.sqlite)')
    
    args = parser.parse_args()
    
    # Load API key from file and set as environment variable
    with open(args.api_key_path, 'r') as f:
        api_key = f.read().strip()
        
    if args.llm_provider == 'anthropic':
        os.environ['ANTHROPIC_API_KEY'] = api_key
    else:
        os.environ['OPENAI_API_KEY'] = api_key
    
    db_params = {
        'dbname': args.dbname,
        'user': args.user,
        'password': args.password,
        'host': args.host,
        'port': args.port
    }
    
    if args.mode == 'interactive':
        if args.rag_mode == 'agentic':
            interactive_web_rag(args.llm_provider, args.score)
        else:  # db mode
            run_db_rag(args.llm_provider, db_params, args.score)

if __name__ == "__main__":
    main()
