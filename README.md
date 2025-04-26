# AcceleRAG

A high-performance RAG (Retrieval-Augmented Generation) system focused on speed and accuracy. Version 0.8.0

## Overview

AcceleRAG is designed to provide fast and accurate document retrieval and generation capabilities. It features a modular document chunking system with n-gram based indexing as the default strategy, flexible database support, and advanced embedding models.

## Quick Start

### Using the RAGManager

The most flexible way to use AcceleRAG is through the `RAGManager` class:

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

# Index documents
rag.index(batch_size=100)

# Generate responses
response = rag.generate_response(
    query="What is the capital of France?",
    use_cache=True,
    cache_thresh=0.9
)
```

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

## Performance Optimization: Query Routing

AcceleRAG achieves dramatic speed improvements through intelligent query routing. Here's how it works:

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

### Speed Benefits

1. **Reduced Search Space**
   - Traditional RAG: Searches entire document collection
   - AcceleRAG: Searches only relevant tables
   - Example: 1M documents → 10K relevant documents

2. **Faster Similarity Computation**
   - Traditional RAG: Computes similarity against all embeddings
   - AcceleRAG: Computes similarity against subset of embeddings
   - Example: 1M comparisons → 10K comparisons

3. **Optimized Database Queries**
   - Traditional RAG: Single large query
   - AcceleRAG: Multiple targeted queries
   - Example: 1 query on 1M rows → 3 queries on 10K rows each

### Performance Metrics

#### Algorithmic Runtime Analysis

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

#### Space-Time Analysis

| Component | Traditional RAG | AcceleRAG | Improvement |
|-----------|----------------|-----------|-------------|
| Embedding Storage | O(n) | O(k) per table | O(n/k) |
| Cache Storage | N/A | O(m) | N/A |
| Query Time | O(n) | O(log n + k) | O(n/(log n + k)) |
| Cache Lookup | N/A | O(1) | N/A |

Where:
- m = number of cached responses
- k = average documents per tag table

### Why TinyBERT is Effective

TinyBERT is particularly effective for AcceleRAG because:

1. **Speed vs Accuracy Tradeoff**
   - 4x smaller than BERT-base
   - 6x faster inference
   - Only ~5-10% lower accuracy for similarity tasks

2. **Memory Efficiency**
   - 4-layer architecture vs 12-layer BERT
   - 312-dimensional embeddings vs 768
   - 50% smaller embedding storage

3. **Optimized for Similarity**
   - Fine-tuned on semantic similarity tasks
   - Effective for short text comparisons
   - Robust to domain shifts

## Features

### Current Features (v0.8.0)
- Modular document chunking with n-gram based indexing as default
- SQLite database support
- BERT and LLM-based embeddings
- Tag hierarchy for document organization
- Prompt template management
- Hallucination evaluation system
- Flexible component architecture
- Response caching with quality thresholds
- Soft and hard grounding modes
- Custom scoring, indexing, and retrieval

### Grounding Modes

AcceleRAG supports two grounding modes:

1. **Soft Grounding**
   - Allows some general knowledge
   - More natural responses
   - Better for open-ended questions
   - Default mode

2. **Hard Grounding**
   - Strictly uses retrieved context
   - More factual responses
   - Better for specific questions
   - Reduces hallucinations

Example usage:
```python
rag = RAGManager(
    grounding='hard',  # or 'soft'
    quality_thresh=80.0
)
```

### Response Caching

AcceleRAG includes an intelligent caching system:

```python
# Enable caching
rag = RAGManager(
    enable_cache=True,
    quality_thresh=80.0  # Only cache high-quality responses
)

# Generate response with caching
response = rag.generate_response(
    query="What is the capital of France?",
    use_cache=True,
    cache_thresh=0.9  # Similarity threshold for cache hits
)
```

The cache:
- Stores query embeddings and responses
- Uses cosine similarity for matching
- Respects quality thresholds
- Automatically manages storage

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

## Hallucination Evaluation

AcceleRAG includes a system to evaluate potential hallucinations in generated responses:

```python
from accelerag import evaluate_hallucination

score = evaluate_hallucination(
    generated_text="Your generated text",
    source_chunks=["Relevant source 1", "Relevant source 2"]
)
```

## Contact

For commercial licensing inquiries or other questions, please contact:
- Email: ezw193@gmail.com

## Note on ArXiv Scraper

The arxiv_scraper component is functional but has not been fully tested in production environments. Use with caution and report any issues you encounter.

## Disclaimer

This is version 0.8.0 of AcceleRAG. While it is functional, it is still under active development. Some features may change in future releases. 
