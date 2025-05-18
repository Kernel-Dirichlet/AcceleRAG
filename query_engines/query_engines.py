from base_classes import QueryEngine
from anthropic import Anthropic
import os
import dotenv 
from openai import OpenAI 

class AnthropicEngine(QueryEngine):
    """Default query engine using Anthropic's Claude models."""
    def __init__(self, api_key=None, model_name="claude-3-7-sonnet-20250219"):
        super().__init__(api_key)
        self.model_name = model_name
        self.client = Anthropic()
        
    def _set_api_key(self):
        """Set Anthropic API key in environment."""
        os.environ['ANTHROPIC_API_KEY'] = self.api_key
        
    def generate_response(self, prompt, grounding='soft'):
        """Generate response using Claude."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class OpenAIEngine(QueryEngine):
    """Query engine using OpenAI's models."""
    def __init__(self, api_key=None, model_name="gpt-4"):
        super().__init__(api_key)
        self.model_name = model_name
        self.client = OpenAI()
        
    def _set_api_key(self):
        """Set OpenAI API key in environment."""
        os.environ['OPENAI_API_KEY'] = self.api_key
        
    def generate_response(self, prompt, grounding='soft'):
        """Generate response using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content 
