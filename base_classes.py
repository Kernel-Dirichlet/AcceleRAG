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



