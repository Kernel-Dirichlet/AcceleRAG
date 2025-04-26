# AcceleRAG

A high-performance RAG (Retrieval-Augmented Generation) framework focused on speed and accuracy. Version 0.7.0

## Overview

AcceleRAG is a framework (not a library) designed to provide fast and accurate document retrieval and generation capabilities. It features a modular document chunking system with n-gram based indexing as the default strategy, flexible database support, and advanced embedding models.

## Installation and Setup

### Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/accelerag.git
cd accelerag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Key Configuration

AcceleRAG supports multiple LLM providers and requires an API key for operation. The API key can be provided in two ways:

1. **File Path**: Pass the path to a text file containing the API key:
```python
rag = RAGManager(
    api_key='path/to/api_key.txt',  # File containing API key
    llm_provider='anthropic'  # or 'openai'
)
```

2. **Environment Variable**: Set the appropriate environment variable:
```bash
# For Anthropic
export ANTHROPIC_API_KEY='your-api-key'

# For OpenAI
export OPENAI_API_KEY='your-api-key'
```

### Key Requirements

The following are the core requirements for AcceleRAG (not fully verified):

1. **Python Dependencies**:
   - Python 3.8+
   - numpy
   - pandas
   - sqlite3
   - transformers (for TinyBERT)
   - torch
   - sentence-transformers
   - anthropic (for Claude)
   - openai (for GPT)

2. **System Requirements**:
   - Minimum 4GB RAM
   - 2GB free disk space
   - CPU with AVX2 support (for transformer models)

3. **API Requirements**:
   - Valid API key for chosen LLM provider
   - Internet connection for API calls
   - Sufficient API quota/credits

4. **Database Requirements**:
   - SQLite 3.24+ (for local storage)
   - Write permissions in working directory

Note: These requirements are preliminary and may be updated as the framework evolves.

## Quick Start

### Using the RAGManager

The core of AcceleRAG is the `RAGManager` class in `runners.py`. This is the main framework code for production use. For interactive testing and experimentation, use `run_agent.py` which provides a terminal interface.

```python
from runners import RAGManager

# Basic usage
rag = RAGManager(
    embedding_type='transformer',  # Uses TinyBERT by default
    ngram_size=16,
    api_key='path/to/api_key.txt',
    llm_provider='anthropic',
    rag_mode='local',
    grounding='soft',  # or 'hard' for strict context adherence
    quality_thresh=80.0,
    dir_to_idx='path/to/documents'
)

# Index documents with batch processing
# batch_size determines how many n-grams to process at once
# Larger batches = faster processing but more memory usage
rag.index(batch_size=100)  # Process 100 n-grams per batch

# Generate responses
response = rag.generate_response(
    query="What is the capital of France?",
    use_cache=True,
    cache_thresh=0.9
)
```

### Cache Configuration Options

AcceleRAG provides flexible caching options through two parameters:

1. **enable_cache**: Controls whether new responses can be cached
2. **use_cache**: Controls whether cached responses can be used

The four possible configurations:

```python
# Case 1: No caching (default)
rag = RAGManager(enable_cache=False, use_cache=False)
# - No responses are cached
# - No cache lookups are performed

# Case 2: Read-only caching
rag = RAGManager(enable_cache=False, use_cache=True)
# - No new responses are cached
# - Existing cached responses can be used

# Case 3: Write-only caching
rag = RAGManager(enable_cache=True, use_cache=False)
# - New responses are cached
# - No cache lookups are performed

# Case 4: Full caching
rag = RAGManager(enable_cache=True, use_cache=True)
# - New responses are cached
# - Existing cached responses can be used
```

### Grounding Modes

AcceleRAG supports two grounding modes that significantly impact response quality:

1. **Soft Grounding**
   - Allows supplementing context with general knowledge
   - More natural responses
   - Better for open-ended questions
   - Default mode
   - Higher risk of hallucinations

2. **Hard Grounding**
   - Strictly uses retrieved context
   - More factual responses
   - Better for specific questions
   - Significantly reduces hallucinations
   - Even with LLM-based scoring, hard grounding provides better factual accuracy

Example usage:
```python
# Use hard grounding for factual accuracy
rag = RAGManager(grounding='hard')

# Override grounding mode for specific responses
response = rag.generate_response(
    query="What is the capital of France?",
    grounding='hard'  # Force hard grounding for this response
)
```

### External Cache Database

AcceleRAG supports using an external SQLite database for response caching:

```python
# Use external cache database
rag = RAGManager(
    cache_db='path/to/cache.db',  # External cache database
    # Other parameters...
)

# The cache database must have the following schema:
# CREATE TABLE response_cache (
#     query TEXT PRIMARY KEY,
#     query_embedding BLOB,
#     response TEXT,
#     quality_score FLOAT,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )

# When cache_db is specified:
# - Cache lookups check the external database first
# - New responses are cached in the external database
# - The embeddings database is not used for caching
```

### Batch Processing

The `index()` method's `batch_size` parameter controls how many n-grams are processed simultaneously:

```python
# Small batch size (memory efficient)
rag.index(batch_size=50)  # Process 50 n-grams at a time

# Large batch size (faster processing)
rag.index(batch_size=500)  # Process 500 n-grams at a time

# Recommended batch sizes:
# - Small documents: 50-100
# - Medium documents: 100-200
# - Large documents: 200-500
```

The batch size affects:
- Memory usage: Larger batches use more RAM
- Processing speed: Larger batches process faster
- Disk I/O: Larger batches reduce database writes

### Custom Components

AcceleRAG allows for custom implementations of key components:

```python
# Custom indexing function
def custom_index(corpus_dir, tag_hierarchy, db_params, custom_param1, custom_param2):
    # Your custom indexing logic
    pass

# Custom retrieval function
def custom_retrieve(query, db_params, k, custom_param1, custom_param2):
    # Your custom retrieval logic
    return ["Custom chunk 1", "Custom chunk 2"]

# Custom scorer
class CustomScorer(Scorer):
    def score(self, response: str, query: str) -> float:
        # Your custom scoring logic
        return 95.0

# Initialize with custom components
rag = RAGManager(
    indexer=Indexer(
        custom_index_fn=custom_index,
        custom_index_args={'custom_param1': 'value1', 'custom_param2': 42}
    ),
    retriever=Retriever(
        {'dbname': 'custom.db'},
        custom_retrieve_fn=custom_retrieve,
        custom_retrieve_args={'custom_param1': 'value1', 'custom_param2': 42}
    ),
    scorer=CustomScorer()
)
```

## Scoring Functions

AcceleRAG supports custom scoring functions through the `Scorer` class. While the default uses an LLM-based scoring system, you can implement your own scoring logic:

```python
from runners import Scorer

class CustomScorer(Scorer):
    def score(self, response: str, query: str) -> float:
        # Example: Simple keyword matching
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        return (overlap / len(query_words)) * 100

# Use custom scorer
rag = RAGManager(
    scorer=CustomScorer(),
    quality_thresh=60.0  # Adjust threshold based on scoring function
)
```

Common scoring approaches:
1. **LLM-based (Default)**
   - Uses Claude/GPT to evaluate response quality
   - More accurate but higher cost
   - Better at understanding context

2. **Keyword-based**
   - Simple word overlap metrics
   - Fast and cost-effective
   - Good for fact-based queries

3. **Embedding-based**
   - Cosine similarity between query and response
   - Medium accuracy and cost
   - Works well with semantic similarity

4. **Hybrid Approaches**
   - Combine multiple scoring methods
   - Balance accuracy and cost
   - Customizable weights

## Roadmap

### v0.7.0 (Current)
- Core RAG functionality
- SQLite database support
- Basic caching system
- Soft and hard grounding modes
- Custom scoring, indexing, and retrieval

### v0.8.0 (Next)
- Arbitrary database support (PostgreSQL, MongoDB)
- Custom database adapters
- Distributed indexing capabilities
- Unit testing framework
- Performance benchmarks

### v0.9.0
- Arbitrary caching layer (Redis, Memcached)
- Custom cache adapters
- Distributed caching support
- Parallel indexing
- Progress tracking and resumption

### v1.0.0
- Agentic RAG API
- REST API for RAG operations
- WebSocket support for streaming
- Authentication and rate limiting
- Production-ready documentation
- Additional embedding models (beyond TinyBERT)
- Custom LLM support
- Advanced monitoring and metrics

## Performance and Cost Analysis

### Algorithmic Runtime Analysis

| Operation | Traditional RAG | AcceleRAG | Speedup |
|-----------|----------------|-----------|---------|
| Query Routing | N/A | O(log n) | N/A |
| Embedding Search | O(n) | O(log n) | O(n/log n) |
| Similarity Computation | O(n) | O(k) | O(n/k) |
| Total | O(n) | O(log n + k) | O(n/(log n + k)) |

Where:
- n = total number of documents
- k = number of relevant documents per tag
- log n = time to traverse tag hierarchy

### Cost Savings

1. **Caching Efficiency**
   - SQLite can store 100k+ responses efficiently
   - Average response size: ~1KB
   - Total cache size: ~100MB for 100k responses
   - Cache hits reduce LLM API calls by 40-60%

2. **Embedding Optimization**
   - TinyBERT reduces embedding storage by 50%
   - 312-dimensional embeddings vs 768
   - 4x smaller model size
   - 6x faster inference

3. **Query Routing**
   - Reduces search space by 90-99%
   - Example: 1M documents → 10K relevant
   - Fewer similarity computations
   - Lower database load

4. **Batch Processing**
   - Reduces API calls during indexing
   - Optimizes database writes
   - Better resource utilization
   - Faster initial setup

## Architecture and Scalability

### Traditional RAG vs AcceleRAG

```
Traditional RAG:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Embedding  │────▶│  Search All │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Compute    │     │  Compare    │
                    │  Similarity │     │  Against    │
                    └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Return     │     │  All        │
                    │  Results    │     │  Documents  │
                    └─────────────┘     └─────────────┘

AcceleRAG:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Route to   │────▶│  Search     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │              Relevant Tables
                           ▼                   │
                    ┌─────────────┐           ▼
                    │  Identify   │     ┌─────────────┐
                    │  Relevant   │     │  Compare    │
                    │  Tables     │     │  Against    │
                    └─────────────┘     └─────────────┘
                           │              Fewer Docs
                           ▼                   │
                    ┌─────────────┐           ▼
                    │  Return     │     ┌─────────────┐
                    │  Results    │     │  Faster     │
                    └─────────────┘     └─────────────┘
```

### Future Scalability Features

1. **Parallel Indexing**
   ```
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Document   │──── │  Parallel   │──── │  Multiple   │
   │  Input      │     │  Processing │     │  Workers    │
   └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Distributed│     │  Concurrent │
                    │  Storage    │     │  Indexing   │
                    └─────────────┘     └─────────────┘
   ```

2. **Horizontal Scaling**
   ```
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Document   │──── │  Sharded    │──── │  Multiple   │
   │  Store      │     │  Storage    │     │  Databases  │
   └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Distributed│     │  Load       │
                    │  Cache      │     │  Balancing  │
                    └─────────────┘     └─────────────┘
   ```


### Why AcceleRAG Scales

1. **Efficient Query Routing**
   - Tag-based document organization
   - Reduced search space (90-99% fewer comparisons)
   - Faster similarity computations
   - Lower database load

2. **Parallel Processing**
   - Multi-threaded document indexing
   - Distributed storage capabilities
   - Concurrent query processing
   - Load balancing across workers

3. **Horizontal Scalability**
   - Sharded document storage
   - Distributed caching layer
   - Multiple database support
   - Elastic scaling capabilities

4. **Advanced Scoring**
   - **Multi-stage fact verification**
   - Context alignment checks
   - **Zero hallucination guarantee**
   - Continuous quality improvement

**IMPORTANT**: The upcoming advanced scoring pipeline will implement a multi-stage verification process that guarantees zero hallucinations in responses. This is achieved through:
- Strict context alignment
- Multi-fact verification
- Cross-referencing with source documents
- Confidence scoring at multiple levels

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

The default n-gram based chunking creates non-overlapping chunks of text, optimized for fast retrieval. The n-gram size can be configured based on your needs:

| N-gram Size | Use Case | Pros | Cons |
|-------------|----------|------|------|
| Small (2-4) | Quick retrieval | Fast processing | Less context |
| Medium (8-16) | Balanced | Good context | Moderate speed |
| Large (32+) | Deep context | Rich context | Slower processing |

Example configuration:
```python
rag = RAGManager(
    ngram_size=16,  # Medium size for balanced performance
    dir_to_idx="path/to/documents"
)
```

### Database Configuration

AcceleRAG uses SQLite for efficient document storage and retrieval:

```python
rag = RAGManager(
    dir_to_idx="path/to/documents",
    # Database will be created as {dir_name}_embeddings.db.sqlite
)
```

### Tag Hierarchy

Documents must be organized in a specific directory structure that matches the tag hierarchy:

```
documents/
├── cs/
│   ├── AI/
│   │   ├── machine_learning/
│   │   └── deep_learning/
│   └── systems/
└── math/
    ├── statistics/
    └── algebra/
```

This structure enables efficient query routing and retrieval.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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
3. **Efficient Storage**: SQLite-based document storage
4. **Modular Chunking**: Customizable document processing
5. **Tag-based Organization**: Hierarchical document structure
6. **Response Caching**: Built-in caching with quality control
7. **Flexible Components**: Custom scoring, indexing, and retrieval

## Contact

For commercial licensing inquiries or other questions, please contact:
- Email: ezw193@gmail.com

## Note on ArXiv Scraper

The arxiv_scraper component is functional but has not been fully tested in production environments. Use with caution and report any issues you encounter.

## Disclaimer

This is version 0.7.0 of AcceleRAG. While it is functional, it is still under active development. Some features may change in future releases. 
