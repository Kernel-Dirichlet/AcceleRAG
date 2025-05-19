from base_classes import QueryEngine
from anthropic import Anthropic
import os
from openai import OpenAI 

class AnthropicEngine(QueryEngine):
    """Default query engine using Anthropic's Claude models."""
    def __init__(self, api_key=None, model_name="claude-3-7-sonnet-20250219"):
        super().__init__(api_key)
        self.model_name = model_name
        self.client = Anthropic(api_key=self.api_key)
        
    def _set_api_key(self):
        """Set Anthropic API key in environment."""
        os.environ['ANTHROPIC_API_KEY'] = self.api_key
        
    def generate_response(self, prompt, grounding='soft'):
        """Generate response using Claude."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

class OpenAIEngine(QueryEngine):
    """Query engine using OpenAI's models."""
    def __init__(self, api_key=None, model_name="gpt-4o"):
        super().__init__(api_key)
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        
    def _set_api_key(self):
        """Set OpenAI API key in environment."""
        os.environ['OPENAI_API_KEY'] = self.api_key
        
    def generate_response(self, prompt, grounding='soft'):
        """Generate response using OpenAI.
        
        Args:
            prompt: The input prompt
            grounding: Grounding mode ('soft' or 'hard')
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.7 if grounding == 'soft' else 0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") 

