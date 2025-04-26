import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin, urlparse
from indexing_utils import get_ngrams, compute_embeddings
import sqlite3
import psycopg2
from anthropic import Anthropic
from openai import OpenAI
import os
import re
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WebContentIndexer:
    def __init__(
        self,
        db_type: str = 'sqlite',
        db_params: Dict = None,
        llm_provider: str = 'anthropic',
        max_depth: int = 2,
        max_pages: int = 10,
        batch_size: int = 32
    ):
        self.db_type = db_type
        self.db_params = db_params or {'dbname': 'web_embeddings.db'}
        self.llm_provider = llm_provider
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.batch_size = batch_size
        self.visited_urls = set()
        self.pages_processed = 0
        
        # Initialize LLM client
        if llm_provider == 'anthropic':
            self.llm_client = Anthropic()
        else:
            self.llm_client = OpenAI()
            
        # Initialize database connection
        self._init_db()
        
    def _init_db(self):
        """Initialize database connection and create necessary tables"""
        if self.db_type == 'sqlite':
            self.conn = sqlite3.connect(self.db_params['dbname'])
            self.cur = self.conn.cursor()
        else:
            self.conn = psycopg2.connect(**self.db_params)
            self.cur = self.conn.cursor()
            
        # Create tables if they don't exist
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS web_content (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                summary TEXT,
                ngram TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        
    def _clean_text(self, text: str) -> str:
        """Clean HTML text by removing unwanted elements and normalizing whitespace"""
        # Remove script and style elements
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def _summarize_text(self, text: str) -> str:
        """Summarize text using LLM"""
        if self.llm_provider == 'anthropic':
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240320",
                max_tokens=1000,
                system="Summarize the following text concisely while preserving key information:",
                messages=[{"role": "user", "content": text}]
            )
            return response.content[0].text
        else:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Summarize the following text concisely while preserving key information:"},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
            
    def _get_page_content(self, url: str) -> Optional[Dict]:
        """Fetch and process a single web page"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else url
            
            # Get main content (prioritize article, main, or content areas)
            content = None
            for selector in ['article', 'main', '#content', '.content']:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text()
                    break
                    
            if not content:
                content = soup.get_text()
                
            # Clean text
            content = self._clean_text(content)
            
            # Summarize content
            summary = self._summarize_text(content)
            
            # Generate ngrams from summary
            ngrams = get_ngrams(summary, n=16)  # Using medium-sized ngrams
            
            # Compute embeddings for ngrams
            embeddings = compute_embeddings(
                ngrams,
                embedding_type='transformer',  # Use TinyBERT for efficiency
                llm_provider=self.llm_provider
            )
            
            return {
                'url': url,
                'title': title,
                'summary': summary,
                'ngrams': ngrams,
                'embeddings': embeddings
            }
            
        except Exception as e:
            logging.error(f"Error processing {url}: {str(e)}")
            return None
            
    def _get_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract and normalize links from a page"""
        links = []
        base_url = urlparse(url)
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            
            # Only include links from the same domain
            if urlparse(absolute_url).netloc == base_url.netloc:
                links.append(absolute_url)
                
        return links
        
    def _process_batch(self, batch: List[Dict]):
        """Process and store a batch of web content"""
        for item in batch:
            try:
                for ngram, embedding in zip(item['ngrams'], item['embeddings']):
                    if self.db_type == 'sqlite':
                        self.cur.execute("""
                            INSERT OR IGNORE INTO web_content 
                            (url, title, summary, ngram, embedding)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            item['url'],
                            item['title'],
                            item['summary'],
                            ngram,
                            embedding.tobytes()
                        ))
                    else:
                        self.cur.execute("""
                            INSERT INTO web_content 
                            (url, title, summary, ngram, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (url) DO NOTHING
                        """, (
                            item['url'],
                            item['title'],
                            item['summary'],
                            ngram,
                            embedding.tobytes()
                        ))
                self.conn.commit()
            except Exception as e:
                logging.error(f"Error storing content for {item['url']}: {str(e)}")
                self.conn.rollback()
                
    def index_website(self, url: str, depth: int = 0):
        """Recursively index a website up to max_depth"""
        if depth > self.max_depth or self.pages_processed >= self.max_pages:
            return
            
        if url in self.visited_urls:
            return
            
        self.visited_urls.add(url)
        logging.info(f"Processing {url} (depth {depth})")
        
        try:
            # Get and process page content
            content = self._get_page_content(url)
            if not content:
                return
                
            # Process in batches
            self._process_batch([content])
            self.pages_processed += 1
            
            # Get links and process recursively
            if depth < self.max_depth:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                links = self._get_links(url, soup)
                
                for link in links:
                    if self.pages_processed >= self.max_pages:
                        break
                    self.index_website(link, depth + 1)
                    time.sleep(1)  # Be nice to servers
                    
        except Exception as e:
            logging.error(f"Error indexing {url}: {str(e)}")
            
    def index_websites(self, urls: List[str]):
        """Index multiple websites"""
        for url in urls:
            self.index_website(url)
            
    def close(self):
        """Close database connection"""
        self.cur.close()
        self.conn.close()
        
def main():
    parser = argparse.ArgumentParser(description='Web content indexer for AcceleRAG')
    parser.add_argument('--web_src', required=True,
                      help='Path to file containing list of websites to index (one per line)')
    parser.add_argument('--ngram_size', type=int, default=16,
                      help='Size of ngrams to generate (default: 16)')
    parser.add_argument('--max_depth', type=int, default=2,
                      help='Maximum depth for recursive website crawling (default: 2)')
    parser.add_argument('--max_pages', type=int, default=10,
                      help='Maximum number of pages to process per website (default: 10)')
    parser.add_argument('--api_key_path', required=True,
                      help='Path to file containing API key for LLM provider')
    parser.add_argument('--llm_provider', choices=['anthropic', 'openai'], default='anthropic',
                      help='LLM provider to use (default: anthropic)')
    parser.add_argument('--db_type', choices=['sqlite', 'postgres'], default='sqlite',
                      help='Database type to use (default: sqlite)')
    parser.add_argument('--dbname', default='web_embeddings.db',
                      help='Database name (default: web_embeddings.db)')
    parser.add_argument('--host', default='localhost',
                      help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432,
                      help='PostgreSQL port (default: 5432)')
    parser.add_argument('--user', default='postgres',
                      help='PostgreSQL user (default: postgres)')
    parser.add_argument('--password', default='postgres',
                      help='PostgreSQL password (default: postgres)')
    
    args = parser.parse_args()
    
    # Load API key from file
    try:
        with open(args.api_key_path, 'r') as f:
            api_key = f.read().strip()
            if args.llm_provider == 'anthropic':
                os.environ['ANTHROPIC_API_KEY'] = api_key
            else:
                os.environ['OPENAI_API_KEY'] = api_key
    except Exception as e:
        logging.error(f"Error loading API key: {str(e)}")
        return
        
    # Load websites from file
    try:
        with open(args.web_src, 'r') as f:
            websites = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error loading websites from {args.web_src}: {str(e)}")
        return
        
    # Set up database parameters
    if args.db_type == 'sqlite':
        db_params = {'dbname': args.dbname}
    else:
        db_params = {
            'dbname': args.dbname,
            'user': args.user,
            'password': args.password,
            'host': args.host,
            'port': args.port
        }
        
    # Initialize and run indexer
    indexer = WebContentIndexer(
        db_type=args.db_type,
        db_params=db_params,
        llm_provider=args.llm_provider,
        max_depth=args.max_depth,
        max_pages=args.max_pages
    )
    
    try:
        logging.info(f"Starting to index {len(websites)} websites")
        indexer.index_websites(websites)
        logging.info("Indexing completed successfully")
    except Exception as e:
        logging.error(f"Error during indexing: {str(e)}")
    finally:
        indexer.close()
        
if __name__ == "__main__":
    main() 
