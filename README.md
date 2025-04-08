# AcceleRAG

A high-performance RAG (Retrieval-Augmented Generation) system focused on speed and accuracy. Version 0.1.0

## Overview

AcceleRAG is designed to provide fast and accurate document retrieval and generation capabilities. It features a modular document chunking system with n-gram based indexing as the default strategy, flexible database support, and advanced embedding models.

## Features

### Current Features (v0.1.0)
- Modular document chunking with n-gram based indexing as default
- Support for SQLite and PostgreSQL databases
- BERT and LLM-based embeddings
- Tag hierarchy for document organization
- Prompt template management
- Hallucination evaluation system

### Upcoming Features
- Support for unstructured data
- Additional LLM providers
- Enhanced metadata support
- Improved performance optimizations

## Installation

Currently, AcceleRAG is not available on PyPI. To use it, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/accelerag.git
cd accelerag
pip install -r requirements.txt
```

## Core Components

### Document Chunking

AcceleRAG provides a modular document chunking system, with n-gram based chunking as the default strategy. The system is designed to be extensible, allowing you to implement custom chunking strategies while maintaining the core functionality.

#### Default N-gram Chunking

The default n-gram based chunking creates overlapping chunks of text, allowing for more precise retrieval. The n-gram size can be configured based on your needs:

| N-gram Size | Use Case | Pros | Cons |
|-------------|----------|------|------|
| Small (2-4) | Quick retrieval | Fast processing | Less context |
| Medium (8-16) | Balanced | Good context | Moderate speed |
| Large (32+) | Deep context | Rich context | Slower processing |

Example configuration:
```python
from accelerag import process_corpus

process_corpus(
    corpus_dir="path/to/documents",
    ngram_size=16,  # Medium size for balanced performance
    db_type="sqlite",
    db_params={"dbname": "embeddings.db"}
)
```

#### Custom Chunking Strategies

You can implement custom chunking strategies by extending the base chunker class:

```python
from accelerag import BaseChunker

class CustomChunker(BaseChunker):
    def chunk_document(self, text):
        # Implement your custom chunking logic
        return chunks
```

### Database Flexibility

AcceleRAG supports multiple database backends:

```python
# SQLite configuration
db_params = {
    "dbname": "embeddings.db"
}

# PostgreSQL configuration
db_params = {
    "host": "localhost",
    "port": 5432,
    "dbname": "embeddings",
    "user": "postgres",
    "password": "your_password"
}
```

### Tag Hierarchy

Documents must be organized in a specific directory structure that matches the tag hierarchy:

```
documents/
├── cats/
│   ├── behavior.txt
│   ├── care.txt
│   └── facts.txt
├── dogs/
│   ├── behavior.txt
│   ├── care.txt
│   └── facts.txt
└── parrots/
    ├── behavior.txt
    ├── care.txt
    └── facts.txt
```

### Embedding Models

AcceleRAG supports multiple embedding models:

```python
# BERT embeddings
embeddings = compute_embeddings(
    ngrams=["your text here"],
    embedding_type='transformer',
    model_name='prajjwal1/bert-tiny'
)

# LLM-based embeddings
embeddings = compute_embeddings(
    ngrams=["your text here"],
    embedding_type='llm',
    llm_provider='anthropic'
)
```

### Prompt Templates

Manage RAG prompts with customizable templates:

```python
from accelerag import PromptTemplate

template = PromptTemplate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Answer the following question: {question}",
    context_prompt="Use the following context: {context}"
)
```

## Limitations of RAG

While RAG systems are powerful, they have inherent limitations:

1. **Context Window**: Limited by the model's context window size
2. **Retrieval Quality**: Dependent on the quality of the indexed documents
3. **Latency**: Additional processing time for retrieval and context integration
4. **Storage Requirements**: Need to store both documents and embeddings

## Comparison with LangChain

AcceleRAG differs from LangChain in several key ways:

1. **Performance Focus**: Optimized for speed and efficiency
2. **Simplified API**: More straightforward interface
3. **Flexible Backend**: Support for multiple database systems
4. **Modular Chunking**: Customizable document processing
5. **Tag-based Organization**: Hierarchical document structure

## Hallucination Evaluation

AcceleRAG includes a system to evaluate potential hallucinations in generated responses:

```python
from accelerag import evaluate_hallucination

score = evaluate_hallucination(
    generated_text="Your generated text",
    source_chunks=["Relevant source 1", "Relevant source 2"]
)
```

## Usage Examples

### Indexing Documents

```python
from accelerag import process_corpus

process_corpus(
    corpus_dir="path/to/documents",
    ngram_size=16,
    db_type="sqlite",
    db_params={"dbname": "embeddings.db"}
)
```

### Querying Documents

```python
from accelerag import fetch_top_k

results = fetch_top_k(
    query="Your question here",
    db_params={"dbname": "embeddings.db"},
    tag_hierarchy=tag_hierarchy,
    k=5
)
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

### Free Use
- Free for research and non-commercial purposes
- Modifications must be shared under the same license
- Source code must be provided with any distributed software

### Commercial Use
- Requires a separate commercial license
- Contact the maintainers for licensing options
- Network service requirements apply

### Network Service Requirements
If you run AcceleRAG as a network service, you must provide the source code to users.

## Contact

For commercial licensing inquiries or other questions, please contact:
- Email: ezw193@gmail.com

## Note on ArXiv Scraper

The arxiv_scraper component is functional but has not been fully tested in production environments. Use with caution and report any issues you encounter.

## Disclaimer

This is version 0.1.0 of AcceleRAG. While it is functional, it is still under active development. Some features may change in future releases. 
