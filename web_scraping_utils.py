import arxiv
import os
import logging
from abc import ABC, abstractmethod

class WebScraper(ABC):
    """Base class for web scrapers"""
    
    @abstractmethod
    def fetch_documents(self, query, max_results=100):
        """Fetch documents from source"""
        pass
    
    @abstractmethod
    def save_document(self, doc, base_path):
        """Save document to filesystem"""
        pass

class ArxivScraper(WebScraper):
    def __init__(self):
        self.client = arxiv.Client()
        
    def fetch_documents(self, query, max_results=100):
        """Fetch papers from ArXiv matching query"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for paper in self.client.results(search):
            results.append({
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [author.name for author in paper.authors],
                'categories': paper.categories,
                'id': paper.entry_id.split('/')[-1],
                'pdf_url': paper.pdf_url
            })
        return results
    
    def save_document(self, doc, base_path):
        """Save ArXiv paper metadata and abstract"""
        for category in doc['categories']:
            category_path = os.path.join(base_path, category)
            os.makedirs(category_path, exist_ok=True)
            
            doc_path = os.path.join(category_path, f"{doc['id']}.txt")
            with open(doc_path, 'w') as f:
                f.write(f"Title: {doc['title']}\n\n")
                f.write(f"Authors: {', '.join(doc['authors'])}\n\n")
                f.write(f"Abstract:\n{doc['abstract']}\n")

def create_document_hierarchy(scraper, queries, output_dir, max_results_per_query=100):
    """Create document hierarchy from web scraping results"""
    logging.info(f"Creating document hierarchy in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for query in queries:
        logging.info(f"Fetching documents for query: {query}")
        try:
            docs = scraper.fetch_documents(query, max_results_per_query)
            for doc in docs:
                try:
                    scraper.save_document(doc, output_dir)
                except Exception as e:
                    logging.error(f"Error saving document {doc.get('id', 'unknown')}: {e}")
        except Exception as e:
            logging.error(f"Error fetching documents for query '{query}': {e}")
            
    logging.info("Document hierarchy creation complete")
