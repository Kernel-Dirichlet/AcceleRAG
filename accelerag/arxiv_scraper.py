import argparse
import logging
import requests
import os
from web_scraping_utils import ArxivScraper, create_document_hierarchy
from preprocessors import extract_text_from_pdf

def main():
    # Set up logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('arxiv_scraping.log'),
            logging.StreamHandler()
        ]
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scrape ArXiv papers and create document hierarchy')
    parser.add_argument('--max_documents', default=100,
                      help='Maximum number of documents to download (default: 1000)')
    parser.add_argument('--output_dir', default='arxiv_papers',
                      help='Output directory for downloaded papers (default: arxiv_papers)')
    parser.add_argument('--save_mode', choices=['binary', 'text'], default='binary',
                      help='Save format for papers - binary (PDF) or extracted text (default: binary)')
    parser.add_argument('--topic',
                      help='Topic to search for (e.g., "machine learning", "computer vision")')
    
    args = parser.parse_args()

    # Define default search queries for different categories
    default_queries = [
        # Computer Science - AI/ML
        'cat:cs.AI OR cat:cs.LG',  # Artificial Intelligence and Machine Learning
        'cat:cs.CL',  # Computation and Language (NLP)
        'cat:cs.CV',  # Computer Vision
        'cat:cs.RO',  # Robotics
        
        # Computer Science - Systems
        'cat:cs.DC OR cat:cs.DS',  # Distributed Computing and Data Structures
        'cat:cs.DB',  # Databases
        'cat:cs.SE',  # Software Engineering
        
        # Computer Science - Theory
        'cat:cs.CC OR cat:cs.LO',  # Computational Complexity and Logic
        'cat:cs.CR',  # Cryptography and Security
    ]

    # Use topic-specific query if provided, otherwise use default queries
    if args.topic:
        queries = [f'all:{args.topic}']
        max_per_query = args.max_documents
    else:
        queries = default_queries
        max_per_query = args.max_documents // len(queries)
    
    try:
        # Initialize scraper
        scraper = ArxivScraper()
        logging.info("Successfully connected to ArXiv API")
        
        total_downloaded = 0
        
        # Create document hierarchy
        for query in queries:
            logging.info(f"Fetching papers for query: {query}")
            docs = scraper.fetch_documents(query, max_per_query)
            
            for doc in docs:
                try:
                    categories = doc['categories']
                    title = doc['title']
                    doc_id = doc['id']
                    pdf_url = doc['pdf_url']
                    
                    logging.info(f"Saving paper: {title} (ID: {doc_id})")
                    logging.info(f"Categories: {', '.join(categories)}")
                    
                    # Save metadata and abstract
                    scraper.save_document(doc, args.output_dir)
                    
                    # Download and save PDF
                    for category in categories:
                        if args.save_mode == 'binary':
                            pdf_path = f"{args.output_dir}/{category}/{doc_id}.pdf"
                            response = requests.get(pdf_url)
                            if response.status_code == 200:
                                with open(pdf_path, 'wb') as f:
                                    f.write(response.content)
                                logging.info(f"Downloaded PDF to {pdf_path}")
                            else:
                                logging.error(f"Failed to download PDF from {pdf_url}")
                        else:  # text mode
                            # First save PDF temporarily
                            temp_pdf = f"{args.output_dir}/temp_{doc_id}.pdf"
                            response = requests.get(pdf_url)
                            if response.status_code == 200:
                                with open(temp_pdf, 'wb') as f:
                                    f.write(response.content)
                                    
                                # Extract text and save
                                text = extract_text_from_pdf(temp_pdf)
                                if text:
                                    txt_path = f"{args.output_dir}/{category}/{doc_id}.txt"
                                    with open(txt_path, 'w', encoding='utf-8') as f:
                                        f.write(text)
                                    logging.info(f"Extracted and saved text to {txt_path}")
                                else:
                                    logging.error(f"Failed to extract text from {temp_pdf}")
                                    
                                # Remove temporary PDF
                                os.remove(temp_pdf)
                            else:
                                logging.error(f"Failed to download PDF from {pdf_url}")
                    
                    total_downloaded += 1
                    logging.info(f"Total papers downloaded: {total_downloaded}")
                    
                except Exception as e:
                    logging.error(f"Error saving document {doc.get('id', 'unknown')}: {e}")
                    
        logging.info(f"Successfully downloaded {total_downloaded} papers to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during scraping: {e}")

if __name__ == "__main__":
    main()
