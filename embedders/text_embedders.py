import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from anthropic import Anthropic
from base_classes import Embedder

class TextEmbedder(Embedder):
    """Text Embedder using transformer models like TinyBERT."""
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



