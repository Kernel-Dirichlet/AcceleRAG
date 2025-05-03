import os
import json
import logging
import sqlite3
import numpy as np
import anthropic
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from retrieval_utils import fetch_top_k
from web_search_utils import score_response
from caching_utils import init_cache, cache_response, get_cached_response, verify_cache_schema, extract_score_from_response
from base_classes import Scorer, Indexer, Retriever, Embedder
from embedders import TransformerEmbedder, ClaudeEmbedder
from scorers import DefaultScorer
from retrievers import DefaultRetriever
from indexers import DefaultIndexer
from query_engines import AnthropicEngine
from caches import DefaultCache


class RAGManager:
    """Main RAG manager class for document indexing, retrieval, and response generation.
    
    Usage:
        # Basic initialization
        rag = RAGManager(
            api_key='path/to/api_key.txt',  # Required for LLM operations
            dir_to_idx='path/to/documents',  # Directory to index
            grounding='soft',  # or 'hard' for strict context adherence
            quality_thresh=80.0,  # Minimum quality score for caching
            enable_cache=True,  # Enable response caching
            use_cache=True,  # Use cached responses
            cache_thresh=0.9,  # Similarity threshold for cache hits
            logging_enabled=True,  # Enable detailed logging
            query_engine=None  # Add query engine parameter
        )
        
        # Index documents
        rag.index()  # Will prompt for reindexing if already indexed
        
        # Generate responses
        response = rag.generate_response(
            query="What is the capital of France?",
            use_cache=True,  # Override instance setting
            cache_thresh=0.9,  # Override instance setting
            grounding='hard'  # Override instance setting
        )
        
        # Retrieve relevant chunks
        chunks = rag.retrieve(
            query="What is the capital of France?",
            top_k=5  # Number of chunks to retrieve
        )
    """
    def __init__(
        self,
        api_key,
        grounding='soft',
        quality_thresh=80.0,
        device = 'cpu',
        dir_to_idx=None,
        embedder = None,
        scorer=None,
        indexer=None,
        retriever=None,
        cache_db=None,
        enable_cache=True,  # Enable caching by default
        use_cache=True,     # Use cache by default
        cache_thresh=0.9,
        force_reindex=False,
        logging_enabled=True,
        query_engine=None,
        show_similarity=False  # Option to show embedding similarity
    ):
        """Initialize RAG manager with configuration parameters.
        
        Args:
            api_key: Path to API key file
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
            logging_enabled: Whether to enable detailed logging (default: True)
            query_engine: Custom query engine instance
            show_similarity: Option to show embedding similarity
        """
        # Initialize basic attributes
        self.grounding = grounding
        self.quality_thresh = quality_thresh
        self.dir_to_idx = dir_to_idx
        self.cache_db = cache_db
        self.enable_cache = enable_cache
        self.use_cache = use_cache
        self.cache_thresh = cache_thresh
        self.force_reindex = force_reindex
        self.logging_enabled = logging_enabled
        self.show_similarity = show_similarity
        
        # Load API key first
        if api_key:
            try:
                with open(api_key, 'r') as f:
                    self.api_key = f.read().strip()
                # Determine provider based on API key format
                if self.api_key.startswith('sk-ant-'):
                    self.provider = 'anthropic'
                else:
                    self.provider = 'openai'
            except Exception as e:
                raise ValueError(f"Error loading API key: {e}")
        
        
        # Initialize components
        self.scorer = scorer or DefaultScorer(self.provider,
                                              self.api_key)
        self.embedder = embedder or TransformerEmbedder(device = device)
        self.indexer = indexer or DefaultIndexer(embedder = self.embedder)
        self.retriever = retriever or DefaultRetriever(dir_to_idx = dir_to_idx,
                                                       embedder = self.embedder)
       
        self.query_engine = query_engine or AnthropicEngine(api_key = self.api_key)
        
        # Initialize cache
        self.cache = DefaultCache()
        if self.enable_cache:
            self._init_cache()
            
        # Set up logging if enabled
        if logging_enabled:
            logging.basicConfig(
                filename='rag_manager.log',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('RAGManager')
        else:
            # Create a null logger
            self.logger = logging.getLogger('RAGManager')
            self.logger.addHandler(logging.NullHandler())
        
    def _init_cache(self):
        """Initialize the cache table in the appropriate database."""
        try:
            if self.cache_db:
                # Verify external cache database schema
                if not self.cache.verify_cache_schema(self.cache_db):
                    raise ValueError(f"External cache database {self.cache_db} has incorrect schema")
                logging.info(f"Using external cache database: {self.cache_db}")
            else:
                # Initialize cache in prompt_cache.db
                self.cache.init_cache(None)  # None will use prompt_cache.db by default
                logging.info("Cache table initialized in prompt_cache.db")
        except Exception as e:
            logging.error(f"Error initializing cache: {e}")
            raise
            
    def _is_indexed(self):
        """Check if documents are already indexed in the database."""
        try:
            # Get all files in the directory to check their structure
            all_files = self.indexer._get_all_files(self.dir_to_idx)
            if not all_files:
                return False
                
            # Get all tables in the database
            conn = sqlite3.connect(self.retriever.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            tables = {row[0] for row in cur.fetchall()}
            
            # If no tables exist, not indexed
            if not tables:
                return False
                
            # Check if any of the tables have content
            has_content = False
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                if count > 0:
                    has_content = True
                    break
                    
            conn.close()
            
            # If no tables with content exist, not indexed
            if not has_content:
                return False
                
            # Check if we have tables for each directory in the structure
            for full_path, rel_path in all_files:
                tag = self.indexer._get_tag_from_path(rel_path, True)  # Always use tag hierarchy
                if tag:
                    table_name = self.indexer._sanitize_table_name(tag)
                    if table_name not in tables:
                        return False
            
            return True
            
        except sqlite3.Error as e:
            if self.logging_enabled:
                self.logger.error(f"Error checking index status: {e}")
            return False

    def _drop_tables(self):
        """Drop all tables in the database except response_cache and SQLite system tables."""
        try:
            conn = sqlite3.connect(self.retriever.db_path)
            cur = conn.cursor()
            
            # Get all tables except SQLite system tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            tables = cur.fetchall()
            
            # Drop each table except response_cache
            for table in tables:
                table_name = table[0]
                if table_name != 'response_cache':
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                    if self.logging_enabled:
                        self.logger.info(f"Dropped table: {table_name}")
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error dropping tables: {e}")
            raise
            
    def index(self, **kwargs):
        """Index documents in the specified directory."""
        try:
            if self.logging_enabled:
                self.logger.info("Starting indexing process")
            
            # Check if directory exists and has files
            if os.path.exists(self.dir_to_idx) and os.listdir(self.dir_to_idx):
                response = input("Documents exist in the directory. Do you want to reindex? (y/n): ")
                if response.lower() != 'y':
                    if self.logging_enabled:
                        self.logger.info("User chose not to reindex")
                    return
            
            # Call indexer's index method
            self.indexer.index(
                corpus_dir=self.dir_to_idx,
                db_params={'dbname': self.retriever.db_path},
                tag_hierarchy=None,
                **kwargs
            )
            
            if self.logging_enabled:
                self.logger.info("Indexing completed successfully")
                
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error during indexing: {e}")
            raise
            
    def retrieve(self, query, top_k=5):
        """Retrieve relevant chunks from the database."""
        try:
            if self.logging_enabled:
                self.logger.info(f"Starting local retrieval with top_k={top_k}")
            
            chunks = self.retriever.retrieve(query, top_k)
            
            if self.logging_enabled:
                if chunks:
                    self.logger.info(f"Retrieved {len(chunks)}/{top_k} chunks:")
                    for i, (chunk, score) in enumerate(chunks):
                        self.logger.info(f"Chunk {i+1} (similarity: {score:.4f}):\n{chunk}\n")
                else:
                    self.logger.warning("No chunks retrieved")
            
            return chunks
            
        except Exception as e:
            if self.logging_enabled:
                self.logger.error(f"Error during retrieval: {str(e)}")
            raise
            
    def _get_cached_response(self, query, threshold, metric='cosine', **kwargs):
        """Retrieve a cached response if a similar query exists.
        
        Args:
            query: Query string
            threshold: Similarity threshold for cache hits
            metric: Distance metric to use ('cosine', 'euclidean', or custom function)
            **kwargs: Additional arguments for custom distance function
            
        Returns:
            Tuple of (response, similarity) if found, None otherwise
        """
        try:
            # Use specified cache db or default to prompt_cache.db
            target_db = self.cache_db if self.cache_db else 'prompt_cache.db'
            
            # Verify schema if using external cache
            if self.cache_db and not verify_cache_schema(self.cache_db):
                raise ValueError("External cache database has incorrect schema")
                
            # Compute query embedding using the embedder
            query_embedding = self.embedder.embed(query)
            
            # Connect to cache database
            conn = sqlite3.connect(target_db)
            cur = conn.cursor()
            
            # Get all cached responses
            cur.execute("""
                SELECT query, response, embedding, score_text
                FROM response_cache
            """)
            results = cur.fetchall()
            
            if not results:
                return None
                
            # Find best matching cached response
            best_similarity = 0.0
            best_response = None
            best_score_text = None
            
            for cached_query, response, cached_embedding, score_text in results:
                try:
                    # Convert string embedding to numpy array
                    cached_embedding = np.array(eval(cached_embedding))
                    
                    # Compute similarity based on metric
                    if metric == 'cosine':
                        similarity = np.dot(query_embedding, cached_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                        )
                    elif metric == 'euclidean':
                        distance = np.linalg.norm(query_embedding - cached_embedding)
                        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    else:
                        # Assume metric is a custom function
                        similarity = metric(query_embedding, cached_embedding, **kwargs)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_response = response
                        best_score_text = score_text
                        
                except Exception as e:
                    if self.logging_enabled:
                        self.logger.error(f"Error computing similarity: {e}")
                    continue
                    
            cur.close()
            conn.close()
            
            if best_response:
                if self.logging_enabled:
                    self.logger.info(f"Cache hit with similarity {best_similarity:.2f}")
                return best_response, best_similarity
                
            return None
            
        except sqlite3.Error as e:
            if self.logging_enabled:
                self.logger.error(f"Error retrieving cached response: {e}")
            return None

    def generate_response(
        self,
        query,
        use_cache=None,
        enable_cache=None,
        cache_thresh=None,
        grounding=None,
        show_similarity=None,
        **kwargs
    ):
        """Generate response for a query using RAG.
        
        Args:
            query: Query string
            use_cache: Whether to use cached responses (overrides instance setting)
            enable_cache: Whether to enable caching of responses (overrides instance setting)
            cache_thresh: Similarity threshold for cache hits (overrides instance setting)
            grounding: Grounding mode ('soft' or 'hard', overrides instance setting)
            show_similarity: Option to show embedding similarity
            **kwargs: Additional arguments to pass to the retriever
            
        Returns:
            Generated response
            
        Usage:
            # Basic response generation
            response = rag.generate_response("What is the capital of France?")
            
            # Force hard grounding for this response
            response = rag.generate_response(
                "What is the capital of France?",
                grounding='hard'
            )
            
            # Override cache settings for this response
            response = rag.generate_response(
                "What is the capital of France?",
                use_cache=True,
                cache_thresh=0.95
            )
        """
        try:
            # Update settings if provided
            if use_cache is not None:
                self.use_cache = use_cache
            if enable_cache is not None:
                self.enable_cache = enable_cache
            if cache_thresh is not None:
                self.cache_thresh = cache_thresh
            if grounding is not None:
                self.grounding = grounding
            if show_similarity is not None:
                self.show_similarity = show_similarity

            if self.logging_enabled:
                self.logger.info(f"Generating response with grounding={self.grounding}, use_cache={self.use_cache}, enable_cache={self.enable_cache}")
            
            # Determine cache database path
            cache_db = self.cache_db if self.cache_db else 'prompt_cache.db'
            
            # Check cache if enabled
            if self.use_cache:
                cached_result = self.cache.get_cached_response(
                    cache_db,
                    query,
                    self.cache_thresh
                )
                if cached_result:
                    answer, similarity = cached_result
                    if self.logging_enabled:
                        self.logger.info(f"Cache hit with similarity {similarity:.4f}")
                    if self.show_similarity:
                        print(f"Cache hit! Similarity score: {similarity:.4f}")
                    return answer

            # Generate new response
            chunks = self.retrieve(query, **kwargs)
            context = "\n\n".join(chunks) if chunks else ""
            
            # Load appropriate grounding prompt
            prompt_file = 'prompts/hard_grounding_prompt.txt' if self.grounding == 'hard' else 'prompts/soft_grounding_prompt.txt'
            try:
                with open(prompt_file, 'r') as f:
                    prompt_template = f.read().strip()
            except FileNotFoundError:
                if self.logging_enabled:
                    self.logger.error(f"Prompt file {prompt_file} not found")
                raise ValueError(f"Prompt file {prompt_file} not found")

            # Format and generate response
            prompt = prompt_template.format(context=context, query=query)
            answer = self.query_engine.generate_response(prompt, grounding=self.grounding)

            # Score response
            score_text, quality_score = self.scorer.score(answer, query)
            
            if self.logging_enabled:
                self.logger.info(f"Response quality score: {quality_score:.4f}")

            # Cache response if enabled and quality threshold met
            if self.enable_cache and quality_score >= self.quality_thresh:
                if self.logging_enabled:
                    self.logger.info(f"Caching response with quality score {quality_score:.4f}")
                
                # Cache the response
                self.cache.cache_response(
                    cache_db,
                    query,
                    answer,
                    score_text,
                    self.quality_thresh,
                    cache_db=cache_db
                )
                
                if self.logging_enabled:
                    self.logger.info(f"Response cached in {cache_db}")

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
