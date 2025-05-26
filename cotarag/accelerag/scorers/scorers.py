import os
from ..base_classes import Scorer
import anthropic
from openai import OpenAI
import re
import logging

class DefaultScorer(Scorer):
    """Default scorer - LLM evaluation."""
    
    def __init__(self, provider, api_key):
        super().__init__(provider, api_key)
        # Load evaluation prompt from file
        try:
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'scorer_prompt.txt')
            with open(prompt_path, 'r') as f:
                self.evaluation_prompt = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"Scorer prompt file not found at {prompt_path}")
        except Exception as e:
            raise ValueError(f"Error loading scorer prompt: {e}")
        
    def score(self, response, query):
        """Score a response using LLM evaluation."""
        # Format evaluation prompt
        prompt = self.evaluation_prompt.format(
            query=query,
            response=response,
            context="No context provided"
        )
        
        # Get evaluation from LLM
        if self.provider == 'anthropic':
            client = anthropic.Anthropic(api_key=self.api_key)
            evaluation = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text
        else:
            client = OpenAI(api_key=self.api_key)
            evaluation = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            
        # Extract overall score (average of criteria)
        try:
            scores = [float(s) for s in evaluation.split() if s.replace('.','').isdigit()]
            overall_score = sum(scores) / len(scores)
        except:
            overall_score = 0.0
            
        return evaluation, overall_score
        
    def score_json(self, response, query, context_chunks):
        """
        Score a response and return detailed JSON output.
        
        Args:
            response: The response to score
            query: The original query
            context_chunks: List of tuples containing (chunk_text, similarity_score)
            
        Returns:
            Dictionary containing detailed scoring information
        """
        # Format context chunks for evaluation
        context_str = "\n\n".join([
            f"Chunk {i+1} (similarity: {score:.4f}):\n{chunk}"
            for i, (chunk, score) in enumerate(context_chunks)
        ])
        
        # Format evaluation prompt
        prompt = self.evaluation_prompt.format(
            query=query,
            response=response,
            context=context_str
        )
        
        # Get evaluation from LLM
        if self.provider == 'anthropic':
            client = anthropic.Anthropic(api_key=self.api_key)
            evaluation = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text
        else:
            client = OpenAI(api_key=self.api_key)
            evaluation = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            
        # Parse evaluation to extract scores
        try:
            # Extract scores from evaluation
            hallucination_score = 0.0
            quality_score = 0.0
            
            for line in evaluation.split('\n'):
                if 'HALLUCINATION SCORE:' in line:
                    try:
                        hallucination_score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'QUALITY SCORE:' in line:
                    try:
                        quality_score = float(line.split(':')[1].strip())
                    except:
                        pass
                        
            # Format context chunks
            context_list = [
                {
                    "text": chunk,
                    "similarity_score": score
                }
                for chunk, score in context_chunks
            ]
            overall_score = (0.3 * quality_score) + (0.7 * (10 - hallucination_score)) 
            return {
                "score": overall_score,  # Overall score is just the quality score
                "response": response,
                "context": context_list,
                "hallucination_risk": 10 - hallucination_score,  # Convert to risk score (out of 10)
                "quality_score": quality_score,  # Quality score (out of 10)
                "evaluation": evaluation
            }
            
        except Exception as e:
            # Return basic structure if parsing fails
            return {
                "score": 0.0,
                "response": response,
                "context": [
                    {
                        "text": chunk,
                        "similarity_score": score
                    }
                    for chunk, score in context_chunks
                ],
                "hallucination_risk": 10.0,  # Maximum risk on error (out of 10)
                "quality_score": 0.0,
                "error": str(e)
            } 

