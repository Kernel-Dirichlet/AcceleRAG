import anthropic
import json
import os
import logging
from arxiv_category_mapper import get_category_description, get_related_categories, suggest_tables_for_query

def create_tag_hierarchy(directory_path, output_file="tag_hierarchy.json"):
    """Create tag hierarchy from directory structure and save to JSON"""
    logging.info(f"Creating tag hierarchy from directory: {directory_path}")
    tag_hierarchy = {}
    
    for root, dirs, files in os.walk(directory_path):
        # Skip hidden directories and files
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files = [f for f in files if not f.startswith('.')]
        
        # Get relative path from the root directory
        rel_path = os.path.relpath(root, directory_path)
        if rel_path == '.':
            continue
            
        # Split path into parts
        path_parts = rel_path.split(os.sep)
        
        # Build the hierarchy
        current = tag_hierarchy
        for part in path_parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add files to the current level
        if files:
            current['_files'] = [f.split('.')[0] for f in files]
        
    # Save the hierarchy
    with open(output_file, 'w') as f:
        json.dump(tag_hierarchy, indent=2, fp=f)
        
    logging.info(f"Tag hierarchy saved to {output_file}")
    return tag_hierarchy

def is_arxiv_query(query, tag_hierarchy):
    """Determine if a query is likely about ArXiv papers"""
    # Check if the query contains ArXiv-specific keywords
    arxiv_keywords = ['arxiv', 'paper', 'research', 'publication', 'conference', 'journal']
    query_lower = query.lower()
    
    # Check for ArXiv category patterns in the tag hierarchy
    has_arxiv_categories = any('.' in tag for tag in tag_hierarchy.keys())
    
    return (any(keyword in query_lower for keyword in arxiv_keywords) or has_arxiv_categories)

def route_query(user_query, tag_hierarchy):
    """Route query to relevant tags using LLM and ArXiv category mapper"""
    logging.info(f"Routing query: {user_query}")
    
    # Determine if this is an ArXiv query
    is_arxiv = is_arxiv_query(user_query, tag_hierarchy)
    
    if is_arxiv:
        logging.info("Query identified as ArXiv-related")
        # First, get suggested tables from arxiv_category_mapper
        suggested_tables = suggest_tables_for_query(user_query)
        if suggested_tables:
            logging.info(f"ArXiv category mapper suggested tables: {suggested_tables}")
            return suggested_tables
            
        # If no suggestions from mapper, use LLM-based routing for ArXiv
        tag_hierarchy_str = json.dumps(tag_hierarchy, separators=(',', ':'))
        
        prompt_template = """Given the following ArXiv category hierarchy and user query, identify the most relevant categories to search in.
The hierarchy contains ArXiv categories like 'cs.AI', 'math.LO', etc.
For database-related queries, consider categories like 'cs.DB' (Databases), 'cs.DS' (Data Structures), and 'cs.DC' (Distributed Computing).
Return only the most specific relevant categories as a JSON array of strings.

Tag Hierarchy:
{tag_hierarchy}

User Query: {user_query}

Return only a JSON array of ArXiv category strings, nothing else."""

    else:
        logging.info("Query identified as non-ArXiv")
        # Use regular LLM-based routing for non-ArXiv queries
        tag_hierarchy_str = json.dumps(tag_hierarchy, separators=(',', ':'))
        
        prompt_template = """Given the following tag hierarchy and user query, identify the most relevant tags to search in.
The tag hierarchy is a JSON object where keys are tag names and values are nested tag hierarchies.
For database-related queries, consider tags related to databases, data structures, and distributed systems.
Return only the most specific relevant tags as a JSON array of strings, including the full path for nested tags.
For example, if the tag hierarchy has "Computer Science/Databases/NoSQL", return the full path "Computer Science/Databases/NoSQL".

Tag Hierarchy:
{tag_hierarchy}

User Query: {user_query}

Return only a JSON array of tag strings with full paths, nothing else."""

    formatted_prompt = prompt_template.format(
        tag_hierarchy=tag_hierarchy_str,
        user_query=user_query
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        tags = json.loads(response.content[0].text)
        if not isinstance(tags, list):
            logging.error(f"Invalid tag format returned: {tags}")
            return []
            
        # Convert tags to table names based on query type
        table_names = []
        for tag in tags:
            if is_arxiv:
                # For ArXiv queries, use the category mapper
                if '.' in tag:  # If it's an ArXiv category
                    table_names.append(suggest_tables_for_query(user_query, tag)[0])
                else:
                    logging.warning(f"Non-ArXiv category found in ArXiv query: {tag}")
            else:
                # For non-ArXiv queries, use regular path conversion
                table_names.append(tag.replace('/', '_'))
                
        logging.info(f"Found relevant tables: {table_names}")
        return table_names
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {e}")
        return []
    except anthropic._exceptions.OverloadedError:
        logging.error("Anthropic API is overloaded")
        return []
    except Exception as e:
        logging.error(f"Error routing query: {e}")
        return []

