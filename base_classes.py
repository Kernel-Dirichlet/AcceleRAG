import os
import json
import logging
from indexing_utils import process_corpus, get_model_provider, compute_embeddings
from retrieval_utils import fetch_top_k, compute_query_embedding
from web_search_utils import score_response
from anthropic import Anthropic
from openai import OpenAI
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class Embedder(ABC):
    """Abstract base class for embedding models."""
    def __init__(self, model_name):
        self.model_name = model_name
        
    @abstractmethod
    def embed(self, text):
        """Compute embedding for given text."""
        pass
        
    @abstractmethod
    def embed_batch(self, texts):
        """Compute embeddings for a batch of texts."""
        pass

class TransformerEmbedder(Embedder):
    """Embedder using transformer models like TinyBERT."""
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D', device='cpu'):
        super().__init__(model_name)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
    def embed(self, text):
        """Compute embedding for a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[:,0,:].cpu().numpy()[0]
        
    def embed_batch(self, texts):
        """Compute embeddings for a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[:,0,:].cpu().numpy()

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

class Scorer:
    """Base class for response scoring."""
    def __init__(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key
        if provider == 'anthropic':
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)
        
    def score(self, response, query):
        """Score a response for a given query."""
        return score_response(response, query, self.provider, self.api_key)

class Indexer:
    """Base class for document indexing."""
    def __init__(
        self,
        model_name='huawei-noah/TinyBERT_General_4L_312D',
        ngram_size=16,
        custom_index_fn=None,
        custom_index_args=None,
        embedder=None
    ):
        self.model_name = model_name
        self.ngram_size = ngram_size
        self.custom_index_fn = custom_index_fn
        self.custom_index_args = custom_index_args or {}
        
        # Initialize embedder
        if embedder is None:
            model_config = get_model_provider(model_name)
            if model_config['provider'] == 'transformer':
                self.embedder = TransformerEmbedder(model_name)
            elif model_config['provider'] == 'anthropic':
                self.embedder = ClaudeEmbedder(model_name)
            else:
                raise ValueError(f"Unsupported model provider: {model_config['provider']}")
        else:
            self.embedder = embedder
        
    def index(
        self,
        corpus_dir,
        tag_hierarchy,
        db_params,
        batch_size=100
    ):
        """Index documents in the specified directory."""
        if self.custom_index_fn:
            # Call custom indexing function with provided arguments
            self.custom_index_fn(
                corpus_dir=corpus_dir,
                tag_hierarchy=tag_hierarchy,
                db_params=db_params,
                **self.custom_index_args
            )
        else:
            # Use default process_corpus with embedder
            process_corpus(
                corpus_dir=corpus_dir,
                tag_hierarchy=tag_hierarchy,
                ngram_size=self.ngram_size,
                batch_size=batch_size,
                db_params=db_params,
                embedder=self.embedder
            )

class Retriever:
    """Base class for document retrieval."""
    def __init__(
        self,
        db_params,
        custom_retrieve_fn=None,
        custom_retrieve_args=None,
        embedder=None
    ):
        self.db_params = db_params
        self.custom_retrieve_fn = custom_retrieve_fn
        self.custom_retrieve_args = custom_retrieve_args or {}
        self.embedder = embedder
        
    def retrieve(self, query, top_k=5):
        """Retrieve relevant chunks from the database."""
        if self.custom_retrieve_fn:
            # Call custom retrieval function with provided arguments
            return self.custom_retrieve_fn(
                query=query,
                db_params=self.db_params,
                k=top_k,
                **self.custom_retrieve_args
            )
        else:
            # Use default fetch_top_k with embedder
            with open('tag_hierarchy.json', 'r') as f:
                tag_hierarchy = json.load(f)
                
            return fetch_top_k(
                query,
                self.db_params,
                tag_hierarchy=tag_hierarchy,
                k=top_k,
                embedder=self.embedder
            ) 
