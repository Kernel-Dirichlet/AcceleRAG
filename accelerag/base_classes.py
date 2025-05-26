import os
import json
import logging
from anthropic import Anthropic
from openai import OpenAI
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Optional, Tuple, List, Any, Dict
from .arxiv_category_mapper import get_category_description, get_related_categories, suggest_tables_for_query

class Embedder(ABC):
    """Abstract base class for embedding models."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for given text."""
        pass
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
        
    def __str__(self) -> str:
        return self.__repr__()



class ClaudeEmbedder(Embedder):
    """Embedder using Claude for embeddings."""
    def __init__(self, model_name='claude-3-sonnet-20240229', api_key=None):
        super().__init__(model_name)
        self.client = Anthropic(api_key=api_key)
        
    def embed(self, text):
        """Compute embedding for a single text."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system="Return an embedding vector for the provided text.",
            messages=[{"role": "user", "content": text}]
        )
        return np.array(response.content[0].text)
        
    def embed_batch(self, texts):
        """Compute embeddings for a batch of texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return np.stack(embeddings)

class Scorer(ABC):
    """Abstract base class for response scoring."""
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        
    @abstractmethod
    def score(self, response: str, query: str) -> Tuple[str, float]:
        """Score a response for a given query."""
        pass
    
    @abstractmethod
    def score_json(self,response,query,context_chunks):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider}')"
        
    def __str__(self) -> str:
        return self.__repr__()

class Indexer(ABC):
    """Abstract base class for document indexing."""
    def __init__(
        self,
        model_name: str,
        ngram_size: int = 16,
        embedder: Optional[Embedder] = None
    ):
        self.model_name = model_name
        self.ngram_size = ngram_size
        self.embedder = embedder
        
    @abstractmethod
    def index(
        self,
        corpus_dir: str,
        tag_hierarchy: Dict[str, Any],
        db_params: Dict[str, Any],
        batch_size: int = 100
    ) -> None:
        """Index documents in the specified directory."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', ngram_size={self.ngram_size})"
        
    def __str__(self) -> str:
        return self.__repr__()

class Retriever(ABC):
    """Abstract base class for document retrieval."""
    def __init__(
        self,
        db_params: Dict[str, Any],
        embedder: Optional[Embedder] = None
    ):
        self.db_params = db_params
        self.embedder = embedder
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Retrieve relevant chunks from the database."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_params={self.db_params})"
        
    def __str__(self) -> str:
        return self.__repr__()

class QueryEngine(ABC):
    """Abstract base class for query engines that handle LLM interactions."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            self._set_api_key()
            
    @abstractmethod
    def _set_api_key(self) -> None:
        """Set API key in environment variables."""
        pass
        
    @abstractmethod
    def generate_response(self, prompt: str, grounding: str = 'soft') -> str:
        """Generate response using the LLM."""
        pass
    
    def create_tag_hierarchy(self, directory_path, output_file="tag_hierarchy.json"):
        """Create tag hierarchy from directory structure and save to JSON"""
        logging.info(f"Creating tag hierarchy from directory: {directory_path}")
        tag_hierarchy = {}
        
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories and files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]
            
            # Get relative path from the root directory
            rel_path = os.path.relpath(root, directory_path)
            if rel_path == '.':
                continue
                
            # Split path into parts
            path_parts = rel_path.split(os.sep)
            
            # Build the hierarchy
            current = tag_hierarchy
            for part in path_parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add files to the current level
            if files:
                current['_files'] = [f.split('.')[0] for f in files]
            
        # Save the hierarchy
        with open(output_file, 'w') as f:
            json.dump(tag_hierarchy, indent=2, fp=f)
            
        logging.info(f"Tag hierarchy saved to {output_file}")
        return tag_hierarchy

    def is_arxiv_query(self, query, tag_hierarchy):
        """Determine if a query is likely about ArXiv papers"""
        # Check if the query contains ArXiv-specific keywords
        arxiv_keywords = ['arxiv', 'paper', 'research', 'publication', 'conference', 'journal']
        query_lower = query.lower()
        
        # Check for ArXiv category patterns in the tag hierarchy
        has_arxiv_categories = any('.' in tag for tag in tag_hierarchy.keys())
        
        return (any(keyword in query_lower for keyword in arxiv_keywords) or has_arxiv_categories)

    def route_query(self, user_query, tag_hierarchy):
        """Route query to relevant tags using LLM and ArXiv category mapper"""
        logging.info(f"Routing query: {user_query}")
        
        # Determine if this is an ArXiv query
        is_arxiv = self.is_arxiv_query(user_query, tag_hierarchy)
        
        if is_arxiv:
            logging.info("Query identified as ArXiv-related")
            # First, get suggested tables from arxiv_category_mapper
            suggested_tables = suggest_tables_for_query(user_query)
            if suggested_tables:
                logging.info(f"ArXiv category mapper suggested tables: {suggested_tables}")
                return suggested_tables
                
            # If no suggestions from mapper, use LLM-based routing for ArXiv
            tag_hierarchy_str = json.dumps(tag_hierarchy, separators=(',', ':'))
            
            prompt_template = """Given the following ArXiv category hierarchy and user query, identify the most relevant categories to search in.
The hierarchy contains ArXiv categories like 'cs.AI', 'math.LO', etc.
For database-related queries, consider categories like 'cs.DB' (Databases), 'cs.DS' (Data Structures), and 'cs.DC' (Distributed Computing).
Return only the most specific relevant categories as a JSON array of strings.

Tag Hierarchy:
{tag_hierarchy}

User Query: {user_query}

Return only a JSON array of ArXiv category strings, nothing else."""

        else:
            logging.info("Query identified as non-ArXiv")
            # Use regular LLM-based routing for non-ArXiv queries
            tag_hierarchy_str = json.dumps(tag_hierarchy, separators=(',', ':'))
            
            prompt_template = """Given the following tag hierarchy and user query, identify the most relevant tags to search in.
The tag hierarchy is a JSON object where keys are tag names and values are nested tag hierarchies.
For database-related queries, consider tags related to databases, data structures, and distributed systems.
Return only the most specific relevant tags as a JSON array of strings, including the full path for nested tags.
For example, if the tag hierarchy has "Computer Science/Databases/NoSQL", return the full path "Computer Science/Databases/NoSQL".

Tag Hierarchy:
{tag_hierarchy}

User Query: {user_query}

Return only a JSON array of tag strings with full paths, nothing else."""

        formatted_prompt = prompt_template.format(
            tag_hierarchy=tag_hierarchy_str,
            user_query=user_query
        )

        try:
            # Use the abstract generate_response method instead of direct API call
            response_text = self.generate_response(formatted_prompt, grounding='hard')
            tags = json.loads(response_text)
            
            if not isinstance(tags, list):
                logging.error(f"Invalid tag format returned: {tags}")
                return []
                
            # Convert tags to table names based on query type
            table_names = []
            for tag in tags:
                if is_arxiv:
                    # For ArXiv queries, use the category mapper
                    if '.' in tag:  # If it's an ArXiv category
                        table_names.append(suggest_tables_for_query(user_query, tag)[0])
                    else:
                        logging.warning(f"Non-ArXiv category found in ArXiv query: {tag}")
                else:
                    # For non-ArXiv queries, use regular path conversion
                    table_names.append(tag.replace('/', '_'))
                    
            logging.info(f"Found relevant tables: {table_names}")
            return table_names
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response: {e}")
            return []
        except Exception as e:
            logging.error(f"Error routing query: {e}")
            return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(api_key={'*' * 8 if self.api_key else None})"
        
    def __str__(self) -> str:
        return self.__repr__()

class PromptCache(ABC):
    """Abstract base class for prompt caching systems."""
    @abstractmethod
    def init_cache(self, db_path: str) -> None:
        """Initialize the cache table in the SQLite database."""
        pass
        
    @abstractmethod
    def verify_cache_schema(self, db_path: str) -> bool:
        """Verify that the database has the correct response_cache table schema."""
        pass
        
    @abstractmethod
    def cache_response(
        self,
        db_path: str,
        query: str,
        response: str,
        quality_score: Optional[str] = None,
        quality_thresh: float = 80.0,
        cache_db: Optional[str] = None
    ) -> None:
        """Cache a query and its response if quality score meets threshold."""
        pass
        
    @abstractmethod
    def get_cached_response(
        self,
        db_path: str,
        query: str,
        threshold: float,
        cache_db: Optional[str] = None
    ) -> Optional[Tuple[str, float]]:
        """Retrieve a cached response if a similar query exists."""
        pass
        
    @abstractmethod
    def clear_cache(self, db_path: str) -> None:
        """Clear all entries from the cache table."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
        
    def __str__(self) -> str:
        return self.__repr__()




