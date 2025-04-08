import anthropic
import json
import os
from openai import OpenAI

def create_tag_hierarchy(directory_path, output_file="tag_hierarchy.json"):
    """Create tag hierarchy from directory structure and save to JSON"""
    tag_hierarchy = {}
    
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        rel_path = os.path.relpath(root, directory_path)
        if rel_path == '.':
            continue
            
        path_parts = rel_path.split(os.sep)
        
        current = tag_hierarchy
        for i, part in enumerate(path_parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
            
        if path_parts[-1] not in current:
            current[path_parts[-1]] = [f.split('.')[0] for f in files if not f.startswith('.')]

    with open(output_file, 'w') as f:
        json.dump(tag_hierarchy, indent=2, fp=f)
        
    return tag_hierarchy

def route_query(user_query, tag_hierarchy, template_path="query_router_prompt.txt", llm_provider='anthropic'):
    """Route user query to hierarchical tags using LLM"""
    try:
        with open(template_path, 'r') as f:
            prompt_template = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt template file '{template_path}' not found")
        return []

    formatted_prompt = prompt_template.format(
        tag_hierarchy=json.dumps(tag_hierarchy, indent=2),
        user_query=user_query
    )

    response_text = ""
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240320",
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user", 
                "content": [{"type": "text", "text": formatted_prompt}]
            }]
        )
        response_text = response.content[0].text.strip()
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=1000,
            temperature=0
        )
        response_text = response.choices[0].message.content.strip()
    else:
        raise ValueError("llm_provider must be 'anthropic' or 'openai'")

    try:
        tags = json.loads(response_text)
        return tags
    except json.JSONDecodeError:
        print("Error: Failed to parse LLM response as JSON")
        return []
