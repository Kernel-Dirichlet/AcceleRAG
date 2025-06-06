import os
import sys
import json
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cotarag.cota_engine import CoTAEngine
from cotarag.cota_engine.thought_actions import ThoughtAction, LLMThoughtAction
from cotarag.accelerag.query_engines.query_engines import AnthropicEngine
from cotarag.accelerag.managers.rag_managers import RAGManager

class ToolformerSOTASummary(LLMThoughtAction):
    def __init__(self, query_engine=None):
        super().__init__(api_key=os.environ.get("CLAUDE_API_KEY"))
        self.query_engine = query_engine

    def thought(self, context):
        return (
            "Summarize the current state of the art regarding Toolformer and related models. "
            "Extract technical insights from the following context:\n\n" + context
        )

    def action(self, thought_output):
        return self.query_engine.generate_response(thought_output)


class ToolformerIdeaProposal(LLMThoughtAction):
    def __init__(self, query_engine=None):
        super().__init__(api_key=os.environ.get("CLAUDE_API_KEY"))
        self.query_engine = query_engine

    def thought(self, sota_summary):
        return (
            "Given the above summary, propose a novel research idea that improves on Toolformer. "
            "This idea should address one or more current limitations such as static tool selection or lack of feedback loops.\n\n"
            f"{sota_summary}"
        )

    def action(self, thought_output):
        return self.query_engine.generate_response(thought_output)


class StubFunctionWriter(LLMThoughtAction):
    def __init__(self, query_engine=None):
        super().__init__(api_key=os.environ.get("CLAUDE_API_KEY"))
        self.query_engine = query_engine

    def thought(self, research_idea):
        return (
            "Write Python stub functions to represent the components of the following research idea. "
            "Define function headers and docstrings only — no implementations.\n\n"
            f"{research_idea}"
        )

    def action(self, thought_output):
        filename = "proposed_toolformer_extension.py"
        with open(filename, "w") as f:
            f.write(thought_output)
        return f"Stub functions written to {filename}:\n\n{thought_output}"

def parse_args():
    parser = argparse.ArgumentParser(description='Research Assistant for Database Design Trends Analysis')
    parser.add_argument('--indexer', 
                       choices=['text', 'centroid'],
                       default='text',
                       help='Type of indexer to use (default: text)')
    parser.add_argument('--retriever',
                       choices=['text', 'centroid'],
                       default='text',
                       help='Type of retriever to use (default: text)')
    parser.add_argument('--top_k',
                       type=int,
                       default=5,
                       help='Number of chunks to retrieve (default: 5)')
    parser.add_argument('--ngram_size',
                       type=int,
                       default=32,
                       help='Size of ngrams for indexing (default: 32)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n=== Research Assistant: Database Design Trends Analysis ===")
    print("Initializing components...")
    
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    # Initialize RAG manager
    rag_manager = RAGManager(
        api_key=api_key,
        dir_to_idx='../cotarag/accelerag/arxiv_mini',
        grounding='hard',
        enable_cache=True,
        use_cache=True,
        cache_thresh=0.9,
        logging_enabled=True,
        force_reindex=True,
        query_engine=AnthropicEngine(api_key=api_key),
        hard_grounding_prompt='../cotarag/accelerag/prompts/hard_grounding_prompt.txt',
        soft_grounding_prompt='../cotarag/accelerag/prompts/soft_grounding_prompt.txt',
        template_path='../cotarag/accelerag/web_rag_template.txt'
    )
    
    # Set database path - ensure it's in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'arxiv_mini_embeddings.db.sqlite')
    print(f"\nUsing database path: {db_path}")
    
    # Set paths on both indexer and retriever
    rag_manager.indexer.db_path = db_path
    rag_manager.retriever.db_path = db_path
    
    # Index documents
    print("\nIndexing documents...")
    rag_manager.index(
        db_params={'dbname': db_path},
        ngram_size=args.ngram_size
    )
    
    # Query and retrieve
    print("\nQuerying documents...")
    query = "summarize the emerging trends in database design"
    
    # Debug: Print retriever state before retrieval
    print("\nRetriever state:")
    print(f"DB path: {rag_manager.retriever.db_path}")
    print(f"Dir to idx: {rag_manager.retriever.dir_to_idx}")
    
    # Get chunks using the manager's retrieve method
    chunks = rag_manager.retrieve(query, top_k=args.top_k)
    print(f"\nRetrieved {len(chunks)} chunks:")
    
    if not chunks:
        print("\nERROR: No chunks retrieved!")
        print("Checking database contents...")
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            print(f"Tables in database: {tables}")
            if tables:
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    count = cur.fetchone()[0]
                    print(f"Rows in {table[0]}: {count}")
            conn.close()
        except Exception as e:
            print(f"Error checking database: {e}")
        return
    
    for i, (chunk, similarity) in enumerate(chunks, 1):
        print(f"\nChunk {i} (Similarity: {similarity:.4f}):")
        print(f"{chunk[:200]}...")
    
    # Generate response
    response = rag_manager.generate_response(
        query=query,
        top_k=args.top_k,
        show_similarity=True
    )
    
    print("\nGenerated Response:")
    print("-" * 50)
    print(response)

    # Initialize CoTA Engine with Toolformer pipeline
    print("\nInitializing CoTA Engine for Toolformer analysis...")
    query_engine = AnthropicEngine(api_key=api_key)
    
    # Create thought actions
    sota_summary = ToolformerSOTASummary(query_engine=query_engine)
    idea_proposal = ToolformerIdeaProposal(query_engine=query_engine)
    stub_writer = StubFunctionWriter(query_engine=query_engine)
    
    # Initialize CoTA Engine with the pipeline
    engine = CoTAEngine([
        sota_summary,
        idea_proposal,
        stub_writer
    ])
    
    # Run the pipeline on the RAG response
    print("\nRunning Toolformer analysis pipeline...")
    results = engine.run(response)
    
    print("\nPipeline Results:")
    print("-" * 50)
    print(results)

if __name__ == '__main__':
    main() 
