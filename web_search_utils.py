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

def score_response(response, query, llm_provider='anthropic'):
    """Score the final LLM response for accuracy and relevance"""
    if llm_provider == 'anthropic':
        client = anthropic.Anthropic()
        prompt = f"""Evaluate the following response to the query. Consider:
1. Accuracy of information
2. Relevance to the query
3. Completeness of the answer
4. Logical coherence

Query: {query}
Response: {response}

Provide a score from 0-100 and a brief explanation of the score."""

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:  # openai
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""Evaluate the following response to the query. Consider:
1. Accuracy of information
2. Relevance to the query
3. Completeness of the answer
4. Logical coherence

Query: {query}
Response: {response}

Provide a score from 0-100 and a brief explanation of the score."""
            }],
            max_tokens=1000
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
            score_msg = score_response(context, query, llm_provider)
            print(score_msg)
            logging.info(f"Query: {query}{score_msg}")
            
        with open('web_rag_template.txt', 'r') as f:
            template = f.read()
            prompt = template.format(
                context=context,
                query=query
            )
