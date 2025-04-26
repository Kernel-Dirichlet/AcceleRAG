import os
import json
import logging
import sqlite3
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
import anthropic
from openai import OpenAI
from indexing_utils import process_corpus, compute_embeddings
from retrieval_utils import fetch_top_k
from web_search_utils import score_response
from caching_utils import init_cache, cache_response, get_cached_response, verify_cache_schema, extract_score_from_response

class Scorer:
    """Base class for response scoring."""
    def __init__(self, llm_provider: str = 'anthropic'):
        self.llm_provider = llm_provider
        
    def score(self, response: str, query: str) -> float:
        """Score a response for a given query.
        
        Args:
            response: Generated response
            query: Original query
            
        Returns:
            Quality score between 0 and 100
        """
        return score_response(response, query, self.llm_provider)

class Indexer:
    """Base class for document indexing."""
    def __init__(
        self,
        embedding_type: str = 'transformer',
        ngram_size: int = 16,
        llm_provider: str = 'anthropic',
        custom_index_fn: Optional[Callable] = None,
        custom_index_args: Optional[Dict] = None
    ):
        self.embedding_type = embedding_type
        self.ngram_size = ngram_size
        self.llm_provider = llm_provider
        self.custom_index_fn = custom_index_fn
        self.custom_index_args = custom_index_args or {}
        
    def index(
        self,
        corpus_dir: str,
        tag_hierarchy: Dict,
        db_params: Dict,
        batch_size: int = 100
    ) -> None:
        """Index documents in the specified directory.
        
        Args:
            corpus_dir: Directory containing documents
            tag_hierarchy: Tag hierarchy for organization
            db_params: Database parameters
            batch_size: Size of batches for processing
        """
        if self.custom_index_fn:
            # Call custom indexing function with provided arguments
            self.custom_index_fn(
                corpus_dir=corpus_dir,
                tag_hierarchy=tag_hierarchy,
                db_params=db_params,
                **self.custom_index_args
            )
        else:
            # Use default process_corpus
            process_corpus(
                corpus_dir=corpus_dir,
                tag_hierarchy=tag_hierarchy,
                ngram_size=self.ngram_size,
                batch_size=batch_size,
                db_params=db_params,
                embedding_type=self.embedding_type,
                llm_provider=self.llm_provider
            )

class Retriever:
    """Base class for document retrieval."""
    def __init__(
        self,
        db_params: Dict,
        custom_retrieve_fn: Optional[Callable] = None,
        custom_retrieve_args: Optional[Dict] = None
    ):
        self.db_params = db_params
        self.custom_retrieve_fn = custom_retrieve_fn
        self.custom_retrieve_args = custom_retrieve_args or {}
        
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks from the database.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        if self.custom_retrieve_fn:
            # Call custom retrieval function with provided arguments
            return self.custom_retrieve_fn(
                query=query,
                db_params=self.db_params,
                k=top_k,
                **self.custom_retrieve_args
            )
        else:
            # Use default fetch_top_k
            with open('tag_hierarchy.json', 'r') as f:
                tag_hierarchy = json.load(f)
                
            return fetch_top_k(
                query,
                self.db_params,
                tag_hierarchy=tag_hierarchy,
                k=top_k
            )

class RAGManager:
    def __init__(
        self,
        embedding_type: str = 'transformer',
        ngram_size: int = 16,
        api_key: str = None,
        llm_provider: str = 'anthropic',
        rag_mode: str = 'local',
        grounding: str = 'soft',
        quality_thresh: float = 80.0,
        dir_to_idx: str = None,
        scorer: Optional[Scorer] = None,
        indexer: Optional[Indexer] = None,
        retriever: Optional[Retriever] = None,
        cache_db: Optional[str] = None,
        enable_cache: bool = True,
        use_cache: bool = False,
        cache_thresh: float = 0.9,
        force_reindex: bool = False
    ):
        """Initialize RAG manager with configuration parameters.
        
        Args:
            embedding_type: Type of embeddings to use ('transformer' or 'llm')
            ngram_size: Size of n-grams for text chunking
            api_key: Path to API key file
            llm_provider: LLM provider to use ('anthropic' or 'openai')
            rag_mode: RAG mode ('local' or 'agentic')
            grounding: Grounding mode ('soft' or 'hard')
            quality_thresh: Quality score threshold for caching
            dir_to_idx: Path to directory to index
            scorer: Custom scorer instance
            indexer: Custom indexer instance
            retriever: Custom retriever instance
            cache_db: Optional path to external cache database
            enable_cache: Whether to enable caching of responses
            use_cache: Whether to use cached responses when available
            cache_thresh: Similarity threshold for cache hits
            force_reindex: Whether to force reindexing even if documents are already indexed
        """
        self.embedding_type = embedding_type
        self.ngram_size = ngram_size
        self.llm_provider = llm_provider
        self.rag_mode = rag_mode
        self.grounding = grounding
        self.quality_thresh = quality_thresh
        self.dir_to_idx = dir_to_idx
        self.cache_db = cache_db
        self.enable_cache = enable_cache
        self.use_cache = use_cache
        self.cache_thresh = cache_thresh
        self.force_reindex = force_reindex
        
        # Initialize components
        self.scorer = scorer or Scorer(llm_provider)
        self.indexer = indexer or Indexer(embedding_type, ngram_size, llm_provider)
        
        # Load API key
        if api_key:
            try:
                with open(api_key, 'r') as f:
                    self.api_key = f.read().strip()
                if llm_provider == 'anthropic':
                    os.environ['ANTHROPIC_API_KEY'] = self.api_key
                else:
                    os.environ['OPENAI_API_KEY'] = self.api_key
            except Exception as e:
                raise ValueError(f"Error loading API key: {e}")
        
        # Initialize LLM client
        if llm_provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        # Set up logging
        logging.basicConfig(
            filename='rag_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Initialize database path
        if dir_to_idx:
            dir_name = os.path.basename(os.path.normpath(dir_to_idx))
            self.db_path = f"{dir_name}_embeddings.db.sqlite"
        else:
            self.db_path = "embeddings.db.sqlite"
            
        self.db_params = {'dbname': self.db_path}
        self.retriever = retriever or Retriever(self.db_params)
        
        # Initialize cache if enabled
        if enable_cache or use_cache:
            self._init_cache()
            
    def _init_cache(self) -> None:
        """Initialize the cache table in the appropriate database."""
        try:
            if self.cache_db:
                # Verify external cache database schema
                if not verify_cache_schema(self.cache_db):
                    raise ValueError(f"External cache database {self.cache_db} has incorrect schema")
                logging.info(f"Using external cache database: {self.cache_db}")
            else:
                # Initialize cache in embeddings database
                init_cache(self.db_path)
                logging.info("Cache table initialized in embeddings database")
        except Exception as e:
            logging.error(f"Error initializing cache: {e}")
            raise
            
    def _is_indexed(self) -> bool:
        """Check if documents are already indexed in the database.
        
        Returns:
            bool: True if documents are indexed, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Check if database has tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            
            if not tables:
                return False
                
            # Check if we have at least one document table
            # Document tables are named after their tags in the hierarchy
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name NOT IN ('sqlite_sequence', 'response_cache')
            """)
            doc_tables = cur.fetchall()
            
            return len(doc_tables) > 0
            
        except sqlite3.Error as e:
            logging.error(f"Error checking index status: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
            
    def index(self, batch_size: int = 100) -> None:
        """Index documents in the specified directory.
        
        Args:
            batch_size: Size of batches for processing
        """
        if not self.dir_to_idx:
            raise ValueError("No directory specified for indexing")
            
        try:
            # Check if documents are already indexed
            if not self.force_reindex and self._is_indexed():
                logging.info("Documents are already indexed. Use force_reindex=True to reindex.")
                return
                
            # Load tag hierarchy
            with open('tag_hierarchy.json', 'r') as f:
                tag_hierarchy = json.load(f)
                
            # Process corpus using indexer
            self.indexer.index(
                corpus_dir=self.dir_to_idx,
                tag_hierarchy=tag_hierarchy,
                db_params=self.db_params,
                batch_size=batch_size
            )
            
            logging.info("Documents indexed successfully")
            
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            raise
            
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks from the database.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        try:
            return self.retriever.retrieve(query, top_k)
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            raise
            
    def generate_response(
        self,
        query: str,
        use_cache: Optional[bool] = None,
        cache_thresh: Optional[float] = None,
        grounding: Optional[str] = None
    ) -> str:
        """Generate response for a query using RAG.
        
        Args:
            query: Query string
            use_cache: Whether to use cached responses (overrides instance setting)
            cache_thresh: Similarity threshold for cache hits (overrides instance setting)
            grounding: Grounding mode ('soft' or 'hard') to use for this response (overrides instance setting)
            
        Returns:
            Generated response
        """
        try:
            # Use provided parameters or instance settings
            use_cache = use_cache if use_cache is not None else self.use_cache
            cache_thresh = cache_thresh if cache_thresh is not None else self.cache_thresh
            grounding = grounding if grounding is not None else self.grounding
            
            # Validate grounding mode
            if grounding not in ['soft', 'hard']:
                raise ValueError("grounding must be either 'soft' or 'hard'")
            
            # Check cache if enabled
            if use_cache:
                cached_result = get_cached_response(
                    self.db_path, 
                    query, 
                    cache_thresh,
                    cache_db=self.cache_db
                )
                if cached_result:
                    answer, similarity = cached_result
                    logging.info(f"Cache hit with similarity {similarity:.2f}")
                    return answer
                    
            # Retrieve relevant chunks
            chunks = self.retrieve(query)
            if not chunks:
                return "No relevant information found in the database."
                
            context = "\n\n".join(chunks)
            
            # Load appropriate grounding prompt based on grounding mode
            # 'soft' uses soft_grounding_prompt.txt which allows general knowledge
            # 'hard' uses hard_grounding_prompt.txt which strictly uses context
            prompt_path = os.path.join('prompts', f'{grounding}_grounding_prompt.txt')
            with open(prompt_path, 'r') as f:
                prompt_template = f.read().strip()
                
            # Format prompt with context and query
            prompt = prompt_template.format(context=context, query=query)
            
            # Generate response
            if self.llm_provider == 'anthropic':
                response = self.client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            # Score response
            quality_score_text = self.scorer.score(answer, query)
            quality_score = extract_score_from_response(quality_score_text)
            
            # Cache response if enabled and quality score meets threshold
            if self.enable_cache and quality_score >= self.quality_thresh:
                cache_response(
                    self.db_path,
                    query,
                    answer,
                    quality_score_text,  # Pass the full score text for caching
                    self.quality_thresh,
                    cache_db=self.cache_db
                )
                
            return answer
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            raise 
