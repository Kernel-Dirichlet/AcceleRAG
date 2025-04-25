"""
ArXiv category mapping and routing utilities.
Provides mapping between ArXiv category codes and their full descriptions,
and helper functions for query routing.
"""

# Primary ArXiv categories and their descriptions
PRIMARY_CATEGORIES = {
    'cs': 'Computer Science',
    'math': 'Mathematics',
    'physics': 'Physics',
    'q-bio': 'Quantitative Biology',
    'q-fin': 'Quantitative Finance',
    'stat': 'Statistics',
    'eess': 'Electrical Engineering and Systems Science',
    'econ': 'Economics',
    'astro-ph': 'Astrophysics',
    'cond-mat': 'Condensed Matter',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math-ph': 'Mathematical Physics',
    'nlin': 'Nonlinear Sciences',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'quant-ph': 'Quantum Physics'
}

# Computer Science subcategories
CS_SUBCATEGORIES = {
    'AI': 'Artificial Intelligence',
    'AR': 'Architecture',
    'CC': 'Computational Complexity',
    'CE': 'Computational Engineering, Finance, and Science',
    'CG': 'Computational Geometry',
    'CL': 'Computation and Language',
    'CR': 'Cryptography and Security',
    'CV': 'Computer Vision and Pattern Recognition',
    'CY': 'Computers and Society',
    'DB': 'Databases',
    'DC': 'Distributed, Parallel, and Cluster Computing',
    'DL': 'Digital Libraries',
    'DM': 'Discrete Mathematics',
    'DS': 'Data Structures and Algorithms',
    'ET': 'Emerging Technologies',
    'FL': 'Formal Languages and Automata Theory',
    'GL': 'General Literature',
    'GR': 'Graphics',
    'GT': 'Game Theory',
    'HC': 'Human-Computer Interaction',
    'IR': 'Information Retrieval',
    'IT': 'Information Theory',
    'LG': 'Machine Learning',
    'LO': 'Logic in Computer Science',
    'MA': 'Multiagent Systems',
    'MM': 'Multimedia',
    'MS': 'Mathematical Software',
    'NA': 'Numerical Analysis',
    'NE': 'Neural and Evolutionary Computing',
    'NI': 'Networking and Internet Architecture',
    'OS': 'Operating Systems',
    'PF': 'Performance',
    'PL': 'Programming Languages',
    'RO': 'Robotics',
    'SC': 'Symbolic Computation',
    'SD': 'Sound',
    'SE': 'Software Engineering',
    'SI': 'Social and Information Networks',
    'SY': 'Systems and Control'
}

# Mathematics subcategories
MATH_SUBCATEGORIES = {
    'AG': 'Algebraic Geometry',
    'AT': 'Algebraic Topology',
    'AP': 'Analysis of PDEs',
    'CT': 'Category Theory',
    'CA': 'Classical Analysis and ODEs',
    'CO': 'Combinatorics',
    'AC': 'Commutative Algebra',
    'CV': 'Complex Variables',
    'DG': 'Differential Geometry',
    'DS': 'Dynamical Systems',
    'FA': 'Functional Analysis',
    'GM': 'General Mathematics',
    'GN': 'General Topology',
    'GT': 'Geometric Topology',
    'GR': 'Group Theory',
    'HO': 'History and Overview',
    'IT': 'Information Theory',
    'KT': 'K-Theory and Homology',
    'LO': 'Logic',
    'MP': 'Mathematical Physics',
    'MG': 'Metric Geometry',
    'NT': 'Number Theory',
    'NA': 'Numerical Analysis',
    'OA': 'Operator Algebras',
    'OC': 'Optimization and Control',
    'PR': 'Probability',
    'QA': 'Quantum Algebra',
    'RT': 'Representation Theory',
    'RA': 'Rings and Algebras',
    'SP': 'Spectral Theory',
    'ST': 'Statistics Theory',
    'SG': 'Symplectic Geometry'
}

def get_category_description(category_code):
    """
    Get the full description of an ArXiv category code.
    
    Args:
        category_code (str): ArXiv category code (e.g., 'cs.AI', 'math.LO')
        
    Returns:
        tuple: (primary_category, subcategory, full_description)
    """
    if '.' not in category_code:
        return None, None, PRIMARY_CATEGORIES.get(category_code, 'Unknown')
        
    primary, sub = category_code.split('.')
    primary_desc = PRIMARY_CATEGORIES.get(primary, 'Unknown')
    
    if primary == 'cs':
        sub_desc = CS_SUBCATEGORIES.get(sub, 'Unknown')
    elif primary == 'math':
        sub_desc = MATH_SUBCATEGORIES.get(sub, 'Unknown')
    else:
        sub_desc = 'Unknown'
        
    full_desc = f"{primary_desc}: {sub_desc}"
    return primary_desc, sub_desc, full_desc

def get_related_categories(category_code):
    """
    Get related categories that might be relevant for query routing.
    
    Args:
        category_code (str): ArXiv category code
        
    Returns:
        list: List of related category codes
    """
    if '.' not in category_code:
        return []
        
    primary, sub = category_code.split('.')
    related = []
    
    # Add cross-disciplinary relationships
    if primary == 'cs':
        if sub == 'AI':
            related.extend(['cs.LG', 'cs.CL', 'cs.CV', 'stat.ML'])
        elif sub == 'LG':
            related.extend(['cs.AI', 'stat.ML', 'math.ST'])
        elif sub == 'CL':
            related.extend(['cs.AI', 'cs.LG', 'cs.IR'])
            
    elif primary == 'math':
        if sub == 'LO':
            related.extend(['cs.LO', 'math.AC', 'math.CT'])
            
    return related

def suggest_tables_for_query(query, category_code=None):
    """
    Suggest relevant tables to query based on the search query and optional category.
    
    Args:
        query (str): The search query
        category_code (str, optional): Specific ArXiv category code
        
    Returns:
        list: List of suggested table names to query
    """
    # Basic keyword matching for category suggestion
    query_lower = query.lower()
    suggested_tables = []
    
    # If category is specified, use it as primary
    if category_code:
        primary, sub, _ = get_category_description(category_code)
        suggested_tables.append(sanitize_table_name(category_code))
        suggested_tables.extend([sanitize_table_name(cat) for cat in get_related_categories(category_code)])
    
    # Add tables based on query keywords
    if any(word in query_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
        suggested_tables.extend(['cs_AI', 'cs_LG', 'stat_ML'])
    if any(word in query_lower for word in ['computer vision', 'image', 'visual']):
        suggested_tables.extend(['cs_CV', 'cs_AI'])
    if any(word in query_lower for word in ['natural language', 'nlp', 'language']):
        suggested_tables.extend(['cs_CL', 'cs_AI'])
    if any(word in query_lower for word in ['logic', 'formal', 'proof']):
        suggested_tables.extend(['math_LO', 'cs_LO'])
        
    return list(set(suggested_tables))  # Remove duplicates

def sanitize_table_name(name):
    """Sanitize table name for SQLite compatibility"""
    # Replace dots, spaces, and other special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure the name starts with a letter or underscore
    if not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = '_' + sanitized
    return sanitized 
