import argparse
import logging
from datetime import datetime
import json
import anthropic
import os
from openai import OpenAI

# Set up logging
logging.basicConfig(
    filename='responses.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def search_web(query, search_provider='anthropic'):
    """Search web using specified provider and return text chunks"""
    
    if search_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-sonnet-20240320",
            max_tokens=1000,
            system="Search the web for relevant information about the query. Return the search results as a list of text chunks.",
            messages=[{"role": "user", "content": query}]
        )
        chunks = response.content[0].text.split('\n\n')
        
    elif search_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set") 
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Search the web for relevant information about the query. Return the search results as a list of text chunks."},
                {"role": "user", "content": query}
            ],
            max_tokens=1000
        )
        chunks = response.choices[0].message.content.split('\n\n')
        
    else:
        raise ValueError("search_provider must be 'anthropic' or 'openai'")
        
    return chunks

def score_chunks(chunks, llm_provider='anthropic'):
    """Score relevance of context chunks and estimate hallucination risk"""
    
    prompt = f"""Please score each context chunk below on a scale of 1-10 based on relevance to the query.
    Then estimate the likelihood of hallucination (0-100%) given these context pieces.
    
    {chunks}
    
    Respond in JSON format:
    {{
        "scores": [scores for each chunk],
        "hallucination_risk": percentage
    }}
    """
    
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-sonnet-20240320",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text)
        
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        
    else:
        raise ValueError("llm_provider must be 'anthropic' or 'openai'")
        
    return result["scores"], result["hallucination_risk"]

def interactive_web_rag(llm_provider='anthropic', score=False):
    """Interactive RAG agent with web search retrieval"""
    
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError("llm_provider must be 'anthropic' or 'openai'")
        
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        chunks = search_web(query, llm_provider)
        context = "\n\n".join(chunks)
        
        if score:
            chunk_scores, hall_risk = score_chunks(context, llm_provider)
            score_msg = f"\nChunk Scores: {chunk_scores}\nHallucination Risk: {hall_risk}%"
            print(score_msg)
            logging.info(f"Query: {query}{score_msg}")
            
        with open('web_rag_template.txt', 'r') as f:
            template = f.read()
            prompt = template.format(
                context=context,
                query=query
            )

def main():
    parser = argparse.ArgumentParser(description='Run web search RAG agent')
    parser.add_argument('--mode', choices=['interactive'], default='interactive',
                      help='Mode to run the agent in')
    parser.add_argument('--score', action='store_true',
                      help='Enable scoring of retrieved chunks')
    parser.add_argument('--llm_provider', choices=['anthropic', 'openai'], default='anthropic',
                      help='LLM provider to use (default: anthropic)')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_web_rag(args.llm_provider, args.score)

if __name__ == "__main__":
    main()