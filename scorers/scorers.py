from anthropic import Anthropic
from openai import OpenAI
import re
import logging
from base_classes import Scorer

class DefaultScorer(Scorer):
    """Default scorer using LLM-based response quality assessment."""
    def __init__(self, provider, api_key):
        super().__init__(provider, api_key)
        if provider == 'anthropic':
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)
            
        # Load the scoring template
        with open('web_rag_template.txt', 'r') as f:
            self.template = f.read().strip()
            
    def score(self, response, query):
        """Score a response using the specified provider.
        
        Args:
            response: The response to score
            query: The original query
            
        Returns:
            tuple: (score_text, score_value) where score_text is the full response
                  and score_value is the parsed numerical score
        """
        # Format the prompt using the template
        prompt = self.template.format(
            context=response,
            query=f"Score this response on a scale of 0-100 based on relevance and accuracy to the query: {query}"
        )
        
        if self.provider == 'anthropic':
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.choices[0].message.content
            
        # Extract numerical score from the response
        try:
            # Look for various score patterns
            patterns = [
                r'#\s*Evaluation Score:\s*(\d+)\s*/\s*100',  # "# Evaluation Score: XX/100"
                r'Evaluation Score:\s*(\d+)\s*/\s*100',  # "Evaluation Score: XX/100"
                r'Score:\s*(\d+)\s*/\s*100',  # "Score: XX/100"
                r'#\s*Score:\s*(\d+)\s*/\s*100',  # "# Score: XX/100"
                r'(\d+)\s*/\s*100',  # "XX/100"
                r'(\d+)\s*out of\s*100',  # "XX out of 100"
                r'#\s*Evaluation Score:\s*(\d+)',  # "# Evaluation Score: XX"
                r'Evaluation Score:\s*(\d+)',  # "Evaluation Score: XX"
                r'Score:\s*(\d+)',  # "Score: XX"
                r'#\s*Score:\s*(\d+)',  # "# Score: XX"
            ]
            
            # First try to find a fraction pattern
            for pattern in patterns:
                match = re.search(pattern, score_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    score = float(match.group(1))
                    # Ensure score is between 0 and 100
                    score_value = max(0.0, min(100.0, score))
                    return score_text, score_value
                    
            logging.warning(f"No score found in response: {score_text[:100]}...")
            return score_text, 0.0
        except Exception as e:
            logging.error(f"Error extracting score: {e}")
            return score_text, 0.0 
