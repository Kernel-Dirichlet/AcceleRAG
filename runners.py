import os
import json
import logging
import sqlite3
import numpy as np
import anthropic
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from indexing_utils import (
    process_corpus, 
    compute_embeddings,
    get_model_provider,
    MODEL_CONFIGS,
    EMBEDDING_MODELS
)
from retrieval_utils import fetch_top_k
from web_search_utils import score_response
from caching_utils import init_cache, cache_response, get_cached_response, verify_cache_schema, extract_score_from_response
from agentic_utils import AgenticIndexer
from base_classes import Scorer, Indexer, Retriever, Embedder, TransformerEmbedder, ClaudeEmbedder

class RAGManager:
    def __init__(
        self,
        model_name='huawei-noah/TinyBERT_General_4L_312D',
        ngram_size=16,
        api_key=None,
        rag_mode='local',
        grounding='soft',
        quality_thresh=80.0,
        dir_to_idx=None,
        scorer=None,
        indexer=None,
        retriever=None,
        cache_db=None,
        enable_cache=False,  # Controls writing to cache
        use_cache=False,     # Controls reading from cache
        cache_thresh=0.9,
        force_reindex=False,
        agentic_indexer=None,
        logging_enabled=True
    ):
        """Initialize RAG manager with configuration parameters.
        
        Args:
            model_name: Name of the model to use
            ngram_size: Size of n-grams for text chunking
            api_key: Path to API key file
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
            agentic_indexer: Custom agentic indexer instance (required for agentic mode)
            logging_enabled: Whether to enable detailed logging (default: True)
        """
        # Initialize basic attributes
        self.model_name = model_name
        self.ngram_size = ngram_size
        self.rag_mode = rag_mode
        self.grounding = grounding
        self.quality_thresh = quality_thresh
        self.dir_to_idx = dir_to_idx
        self.cache_db = cache_db
        self.enable_cache = enable_cache
        self.use_cache = use_cache
        self.cache_thresh = cache_thresh
        self.force_reindex = force_reindex
        self.logging_enabled = logging_enabled
        
        # Get model configuration
        model_config = get_model_provider(model_name)
        self.provider = model_config['provider']
        
        # Load API key first
        if api_key:
            try:
                with open(api_key, 'r') as f:
                    self.api_key = f.read().strip()
                # Determine provider based on API key format
                if self.api_key.startswith('sk-ant-'):
                    self.provider = 'anthropic'
                    os.environ['ANTHROPIC_API_KEY'] = self.api_key
                else:
                    self.provider = 'openai'
                    os.environ['OPENAI_API_KEY'] = self.api_key
            except Exception as e:
                raise ValueError(f"Error loading API key: {e}")
        
        # Initialize embedder based on provider
        if self.provider == 'transformer':
            self.embedder = TransformerEmbedder(model_name)
        elif self.provider == 'anthropic':
            self.embedder = ClaudeEmbedder(model_name, self.api_key)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")
        
        # Initialize components with embedder
        self.scorer = scorer or Scorer(self.provider, self.api_key)
        self.indexer = indexer or Indexer(
            model_name=model_name,
            ngram_size=ngram_size,
            embedder=self.embedder
        )
        
        # Set up logging if enabled
        if logging_enabled:
            logging.basicConfig(
                filename='rag_manager.log',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - [%(mode)s] - %(message)s'
            )
            self.logger = logging.getLogger('RAGManager')
            self.logger = logging.LoggerAdapter(self.logger, {'mode': rag_mode})
        else:
            # Create a null logger
            self.logger = logging.getLogger('RAGManager')
            self.logger.addHandler(logging.NullHandler())
            self.logger = logging.LoggerAdapter(self.logger, {'mode': rag_mode})
        
        # Initialize database path
        if dir_to_idx:
            dir_name = os.path.basename(os.path.normpath(dir_to_idx))
            self.db_path = f"{dir_name}_embeddings.db.sqlite"
        else:
            self.db_path = "embeddings.db.sqlite"
            
        self.db_params = {'dbname': self.db_path}
        self.retriever = retriever or Retriever(
            self.db_params,
            embedder=self.embedder
        )
        
        # Initialize cache if writing is enabled
        if self.enable_cache:
            self._init_cache()
            
        # Initialize agent if in agentic mode
        if rag_mode == 'agentic':
            if logging_enabled:
                self.logger.info("Initializing agentic mode")
            if agentic_indexer is None:
                if logging_enabled:
                    self.logger.info("No custom agent provided, using default ArxivAgent")
                from agentic_utils import ArxivAgent
                self.agentic_indexer = ArxivAgent(embedder=self.embedder)
            else:
                if not isinstance(agentic_indexer, AgenticIndexer):
                    if logging_enabled:
                        self.logger.error("Provided agentic_indexer is not a subclass of AgenticIndexer")
                    raise ValueError("agentic_indexer must be a subclass of AgenticIndexer")
                self.agentic_indexer = agentic_indexer
                if logging_enabled:
                    self.logger.info(f"Using custom agentic indexer: {agentic_indexer.__class__.__name__}")
        
        # Initialize LLM client based on provider
        if self.provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)
            
    def _init_cache(self):
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
            
    def _is_indexed(self):
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
                
            if self.rag_mode == 'agentic':
                # For agentic mode, check if we have at least one document table
                # Document tables are named after their tags in the hierarchy
                cur.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' 
                    AND name NOT IN ('sqlite_sequence', 'response_cache')
                """)
                doc_tables = cur.fetchall()
                return len(doc_tables) > 0
            else:
                # For local mode, check if we have the 'documents' table
                cur.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' 
                    AND name = 'documents'
                """)
                return cur.fetchone() is not None
            
        except sqlite3.Error as e:
            if self.logging_enabled:
                self.logger.error(f"Error checking index status: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
            
    def index(
        self,
        batch_size=100,
        force_reindex=None,
        ngram_size=None,
        indexer=None,
        **kwargs
    ):
        """Index documents using the specified indexer.
        
        Args:
            batch_size: Size of batches for processing
            force_reindex: Whether to force reindexing (overrides instance setting)
            ngram_size: Size of n-grams to use (overrides instance setting)
            indexer: Custom indexer to use (overrides instance setting)
            **kwargs: Additional arguments passed to the indexer
        """
        try:
            # Use provided parameters or instance settings
            force_reindex = force_reindex if force_reindex is not None else self.force_reindex
            ngram_size = ngram_size if ngram_size is not None else self.ngram_size
            indexer = indexer if indexer is not None else self.indexer
            
            # Check if documents are already indexed
            if not force_reindex and self._is_indexed():
                if self.logging_enabled:
                    self.logger.info("Documents are already indexed. Use force_reindex=True to reindex.")
                return
            
            if self.rag_mode == 'agentic':
                if self.logging_enabled:
                    self.logger.info("Starting agentic indexing process")
                    self.logger.info(f"Using agent: {self.agentic_indexer.__class__.__name__}")
                    self.logger.info(f"Batch size: {batch_size}")
                    
                # Add required parameters to kwargs
                kwargs.update({
                    'db_params': self.db_params,
                    'batch_size': batch_size
                })
                
                # Process corpus using agentic indexer
                self.agentic_indexer.index(**kwargs)
                
                if self.logging_enabled:
                    self.logger.info("Agentic indexing completed successfully")
            else:
                if self.logging_enabled:
                    self.logger.info("Starting local indexing process")
                    self.logger.info(f"Batch size: {batch_size}")
                
                # Process corpus using indexer without tag hierarchy
                process_corpus(
                    corpus_dir=self.dir_to_idx,
                    db_params=self.db_params,
                    batch_size=batch_size,
                    ngram_size=ngram_size,
                    embedder=self.embedder
                )
                
                if self.logging_enabled:
                    self.logger.info("Local indexing completed successfully")
            
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error during indexing: {str(e)}")
            raise
            
    def retrieve(self, query, top_k=5):
        """Retrieve relevant chunks from the database.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        try:
            if self.rag_mode == 'agentic':
                if self.logging_enabled:
                    self.logger.info(f"Starting agentic retrieval for query: {query}")
                    self.logger.info(f"Using agent: {self.agentic_indexer.__class__.__name__}")
                    self.logger.info(f"Top k: {top_k}")
                
                results = self.agentic_indexer.retrieve(query, top_k)
                if self.logging_enabled:
                    self.logger.info(f"Retrieved {len(results)} chunks")
                return results
            else:
                if self.logging_enabled:
                    self.logger.info("Starting local retrieval")
                return self.retriever.retrieve(query, top_k)
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error during retrieval: {str(e)}")
            raise
            
    def generate_response(
        self,
        query,
        use_cache=None,
        enable_cache=None,
        cache_thresh=None,
        grounding=None
    ):
        """Generate response for a query using RAG.
        
        Args:
            query: Query string
            use_cache: Whether to use cached responses (overrides instance setting)
            enable_cache: Whether to enable caching of responses (overrides instance setting)
            cache_thresh: Similarity threshold for cache hits (overrides instance setting)
            grounding: Grounding mode ('soft' or 'hard', overrides instance setting)
            
        Returns:
            Generated response
        """
        try:
            # Permanently update class attributes if new values provided
            if grounding is not None:
                self.grounding = grounding
            if use_cache is not None:
                self.use_cache = use_cache
            if enable_cache is not None:
                self.enable_cache = enable_cache
            if cache_thresh is not None:
                self.cache_thresh = cache_thresh
            
            if self.logging_enabled:
                self.logger.info(f"Generating response with grounding={self.grounding}, use_cache={self.use_cache}, enable_cache={self.enable_cache}")
            
            # Initialize cache if needed
            if self.enable_cache:
                init_cache(self.db_path)
            
            # Check cache if reading is enabled
            if self.use_cache:
                cached_result = get_cached_response(
                    self.db_path, 
                    query, 
                    self.cache_thresh,
                    cache_db=self.cache_db,
                    model_name=self.model_name
                )
                if cached_result:
                    answer, similarity = cached_result
                    if self.logging_enabled:
                        self.logger.info(f"Cache hit with similarity {similarity:.2f}")
                    return answer
                    
            # Retrieve relevant chunks
            chunks = self.retrieve(query)
            context = "\n\n".join(chunks) if chunks else ""
            
            # Load appropriate grounding prompt based on mode
            prompt_file = 'prompts/hard_grounding_prompt.txt' if self.grounding == 'hard' else 'prompts/soft_grounding_prompt.txt'
            try:
                with open(prompt_file, 'r') as f:
                    prompt_template = f.read().strip()
            except FileNotFoundError:
                if self.logging_enabled:
                    self.logger.error(f"Prompt file {prompt_file} not found")
                raise ValueError(f"Prompt file {prompt_file} not found")
                
            # Format prompt with context and query
            prompt = prompt_template.format(context=context, query=query)
            
            if self.logging_enabled:
                self.logger.info(f"Using {self.grounding} grounding with prompt from {prompt_file}")
            
            # Generate response using the correct LLM provider
            if self.provider == 'anthropic':
                response = anthropic.Anthropic(api_key=self.api_key).messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                
            # Score response
            quality_score_text = self.scorer.score(answer, query)
            quality_score = extract_score_from_response(quality_score_text)
            
            if self.logging_enabled:
                self.logger.info(f"Response quality score: {quality_score}")
            
            # Cache response if writing is enabled and quality score meets threshold
            if self.enable_cache and quality_score >= self.quality_thresh:
                if self.logging_enabled:
                    self.logger.info("Caching response")
                cache_response(
                    self.db_path,
                    query,
                    answer,
                    quality_score_text,
                    self.quality_thresh,
                    cache_db=self.cache_db,
                    model_name=self.model_name
                )
                
            return answer
            
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error generating response: {str(e)}")
            raise

    def create_hierarchy(self, query, goals):
        """Create document hierarchy using the agent.
        
        Args:
            query: Query string
            goals: User goals for organization
            
        Returns:
            Created hierarchy structure
        """
        if self.rag_mode != 'agentic':
            if self.logging_enabled:
                self.logger.error("create_hierarchy called in non-agentic mode")
            raise ValueError("create_hierarchy is only available in agentic mode")
            
        if self.logging_enabled:
            self.logger.info(f"Creating hierarchy for query: {query}")
            self.logger.info(f"User goals: {goals}")
        
        try:
            hierarchy = self.agentic_indexer.create_hierarchy(query, goals)
            if self.logging_enabled:
                self.logger.info("Hierarchy created successfully")
                self.logger.debug(f"Hierarchy structure: {json.dumps(hierarchy, indent=2)}")
            return hierarchy
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error creating hierarchy: {str(e)}")
            raise 
