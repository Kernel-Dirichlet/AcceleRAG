import argparse
import logging
from datetime import datetime
from retrieval_utils import fetch_top_k
import json
import anthropic
import os
from openai import OpenAI
from web_search_utils import search_web, score_response, interactive_web_rag
import sqlite3
import time

# Set up logging
logging.basicConfig(
    filename='responses.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_db_rag(llm_provider, db_params, score=False, debug=False):
    """Interactive RAG agent with database retrieval"""
    
    # Initialize scoring log if scoring is enabled
    if score:
        try:
            with open('scoring.log', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n[{timestamp}] Starting new session\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            print(f"Error initializing scoring log: {e}")
            logging.error(f"Error initializing scoring log: {e}")
            score = False  # Disable scoring if log initialization fails
    
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
    if not os.path.isabs(db_path):
        # If path is relative, make it absolute relative to current directory
        db_path = os.path.abspath(db_path)
    
    if debug:
        print(f"Looking for database at: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Verify database has tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        if not tables:
            print("Error: No tables found in database. Please run the indexer first.")
            if debug:
                print(f"Database path: {db_path}")
            return
            
        if debug:
            print(f"Found {len(tables)} tables in database:")
            for table in tables:
                # Count rows in each table
                cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cur.fetchone()[0]
                print(f"- {table[0]}: {count} rows")
            
        cur.close()
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        try:
            # Retrieve relevant chunks from database
            tag_hierarchy = {}
            try:
                with open('tag_hierarchy.json', 'r') as f:
                    tag_hierarchy = json.loads(f.read().strip())
            except FileNotFoundError:
                print("Error: tag_hierarchy.json not found. Please create it first.")
                continue
            except json.JSONDecodeError:
                print("Error: tag_hierarchy.json is not valid JSON.")
                continue
                
            chunks = fetch_top_k(query,
                                 db_params,
                                 tag_hierarchy = tag_hierarchy,
                                 k = 3, 
                                 debug = debug)

            if not chunks:
                print("No relevant information found in the database.")
                continue
                
            context = "\n\n".join(chunks)
            print(f"\nRetrieved information:\n{context}\n")
                
            # Format prompt with retrieved context
            prompt = f"Context:\n{context}\n\nQuery: {query}"
                
            # Send to LLM and get response with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    if llm_provider == 'anthropic':
                        response = client.messages.create(
                            model="claude-3-7-sonnet-20250219",
                            max_tokens=1000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        answer = response.content[0].text
                    else:  # openai
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1000
                        )
                        answer = response.choices[0].message.content
                        
                    print(f"\nAnswer: {answer}\n")
                    logging.info(f"Query: {query}\nAnswer: {answer}")
                    
                    # Score the final response if enabled
                    if score:
                        try:
                            score_result = score_response(answer, query, llm_provider)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open('scoring.log', 'a') as f:
                                f.write(f"\n[{timestamp}] Query: {query}\n")
                                f.write(f"Response: {answer}\n")
                                f.write(f"Score: {score_result}\n")
                                f.write("=" * 80 + "\n")
                        except Exception as e:
                            print(f"Error scoring response: {e}")
                            logging.error(f"Error scoring response: {e}")
                    break
                    
                except anthropic._exceptions.OverloadedError:
                    if attempt < max_retries - 1:
                        print(f"API overloaded, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("API is currently overloaded. Please try again later.")
                        logging.error(f"API overloaded after {max_retries} attempts for query: {query}")
                        
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                    logging.error(f"Error processing query {query}: {str(e)}")
                    break
                    
        except Exception as e:
            print(f"Error: {str(e)}")
            logging.error(f"Error in main loop: {str(e)}")
            continue

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
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with verbose output')
    
    # Database connection arguments
    parser.add_argument('--dbname', default='embeddings.db.sqlite', help='Database name/file (default: embeddings.db.sqlite)')
    parser.add_argument('--db_type', choices=['sqlite', 'postgres'], default='sqlite',
                      help='Database type to use (default: sqlite)')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', default=5432, help='PostgreSQL port (default: 5432)')
    parser.add_argument('--user', default='postgres', help='PostgreSQL user (default: postgres)')
    parser.add_argument('--password', default='postgres', help='PostgreSQL password (default: postgres)')
    
    args = parser.parse_args()
    
    # Load API key from file and set as environment variable
    with open(args.api_key_path, 'r') as f:
        api_key = f.read().strip()
        
    if args.llm_provider == 'anthropic':
        os.environ['ANTHROPIC_API_KEY'] = api_key
    else:
        os.environ['OPENAI_API_KEY'] = api_key
    
    # Set up database parameters based on type
    if args.db_type == 'sqlite':
        db_params = {
            'dbname': args.dbname
        }
    else:  # postgres
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
            run_db_rag(args.llm_provider, db_params, args.score, args.debug)

if __name__ == "__main__":
    main()

