import os
import logging
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return None

def preprocess_arxiv_papers(input_dir, output_dir):
    """Process downloaded ArXiv papers and extract text content"""
    processed_docs = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        category_output = os.path.join(output_dir, category)
        os.makedirs(category_output, exist_ok=True)
        
        processed_docs[category] = []
        
        for filename in os.listdir(category_path):
            if not filename.endswith('.pdf'):
                continue
                
            pdf_path = os.path.join(category_path, filename)
            doc_id = filename[:-4]
            
            text = extract_text_from_pdf(pdf_path)
            if text is None:
                continue
                
            txt_path = os.path.join(category_output, f"{doc_id}.txt")
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                processed_docs[category].append(doc_id)
                logging.info(f"Processed {pdf_path} -> {txt_path}")
            except Exception as e:
                logging.error(f"Failed to save text for {doc_id}: {e}")
                
    return processed_docs

def clean_text(text):
    """Clean extracted text by removing PDF artifacts"""
    lines = text.split('\n')
    lines = [line for line in lines if not line.strip().isdigit()]
    lines = [line for line in lines if len(line.strip()) > 0]
    text = ' '.join(lines)
    text = ' '.join(text.split())
    return text
