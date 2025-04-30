import argparse
import json
import logging
import os
from indexing_utils import process_corpus, MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(description='Index documents for AcceleRAG')
    parser.add_argument('--dir', required=True,
                      help='Directory containing documents to index')
    parser.add_argument('--ngram_size', type=int, default=16,
                      help='Size of n-grams')
    parser.add_argument('--dbname', 
                      help='SQLite database name (default: {dir_name}_embeddings.db.sqlite)')
    parser.add_argument('--model', default='huawei-noah/TinyBERT_General_4L_312D',
                      help='Model to use for embeddings (default: TinyBERT)')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Batch size for processing')
    parser.add_argument('--device', default='cpu',
                      help='Device to run model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Generate default database name if not provided
    if not args.dbname:
        dir_name = os.path.basename(os.path.normpath(args.dir))
        args.dbname = f"{dir_name}_embeddings.db.sqlite"
    
    # Load tag hierarchy
    try:
        with open('tag_hierarchy.json', 'r') as f:
            tag_hierarchy = json.load(f)
    except FileNotFoundError:
        logging.error("tag_hierarchy.json not found")
        return
    except json.JSONDecodeError:
        logging.error("Invalid JSON in tag_hierarchy.json")
        return
        
    # Process corpus with SQLite parameters
    process_corpus(
        corpus_dir=args.dir,
        tag_hierarchy=tag_hierarchy,
        ngram_size=args.ngram_size,
        batch_size=args.batch_size,
        db_params={'dbname': args.dbname},
        model_name=args.model,
        device=args.device
    )
    
if __name__ == "__main__":
    main()

