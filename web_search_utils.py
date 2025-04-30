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

def score_response(response, query, provider, api_key):
    """Score a response using the specified provider."""
    if provider == 'anthropic':
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"Score this response to the query '{query}':\n\n{response}\n\nScore from 0-100 based on relevance and accuracy."
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        client = OpenAI(api_key=api_key)
        prompt = f"Score this response to the query '{query}':\n\n{response}\n\nScore from 0-100 based on relevance and accuracy."
        response = client.chat.completions.create(
            model="gpt-4",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

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
            score_msg = score_response(context, query, llm_provider, api_key)
            print(score_msg)
            logging.info(f"Query: {query}{score_msg}")
            
        with open('web_rag_template.txt', 'r') as f:
            template = f.read()
            prompt = template.format(
                context=context,
                query=query
            )
