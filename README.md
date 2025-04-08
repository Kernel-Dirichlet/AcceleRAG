# AcceleRAG

A lightweight, flexible, and scalable RAG (Retrieval-Augmented Generation) framework that prioritizes transparency and control over abstraction.

## Overview

AcceleRAG is designed to provide a straightforward yet powerful approach to RAG, avoiding the pitfalls of over-abstracted frameworks. It focuses on:

- Direct control over the indexing and retrieval pipeline
- Flexible database integration
- Transparent embedding computation
- Hierarchical document organization
- Hallucination risk assessment

## Version 0.1.0

Current features:
- Document indexing with n-gram based chunking
- SQLite and PostgreSQL support
- BERT and LLM-based embeddings
- Tag hierarchy for document organization
- Hallucination risk evaluation
- Prompt template management

Coming soon:
- Support for unstructured data (PDFs, images, etc.)
- Additional LLM providers
- Enhanced metadata support
- Vector similarity search optimizations

## Installation

```bash
pip install accelerag
```

## Core Components

### 1. Indexing Stage

The indexing stage is the foundation of AcceleRAG's retrieval system. It uses n-grams as the default chunking strategy.

#### N-gram Configuration

```python
# Default n-gram size is 3
process_corpus(
    corpus_dir="documents",
    ngram_size=3,  # Adjustable based on needs
    batch_size=32
)
```

#### N-gram Size Tradeoffs

| N-gram Size | Pros | Cons |
|-------------|------|------|
| Small (1-3) | - Better for short, precise queries<br>- Lower computational overhead | - May miss context<br>- More chunks to process |
| Medium (4-6) | - Good balance of context and precision<br>- Works well for most use cases | - Moderate computational cost<br>- May split some concepts |
| Large (7+) | - Captures complete concepts<br>- Better for long-form queries | - Higher computational cost<br>- May include irrelevant context |

### 2. Database Flexibility

AcceleRAG supports both SQLite and PostgreSQL, allowing you to choose based on your scalability needs.

```python
# SQLite configuration
db_params = {
    'dbname': 'embeddings.db'
}

# PostgreSQL configuration
db_params = {
    'dbname': 'embeddings',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}
```

#### Metadata Support (Coming Soon)

Future versions will support rich metadata association with chunks:
- Document source
- Creation date
- Author information
- Custom tags
- Access permissions

### 3. Tag Hierarchy

Documents must be organized in a hierarchical structure where files are placed at leaf nodes.

```
documents/
├── Computer_Science/
│   ├── AI/
│   │   ├── Machine_Learning/
│   │   │   ├── paper1.txt
│   │   │   └── paper2.txt
│   │   └── NLP/
│   │       ├── paper3.txt
│   │       └── paper4.txt
│   └── Systems/
│       ├── Database/
│       │   └── paper5.txt
│       └── Distributed/
│           └── paper6.txt
└── Mathematics/
    ├── Algebra/
    │   └── paper7.txt
    └── Calculus/
        └── paper8.txt
```

### 4. Embedding Models

AcceleRAG supports both traditional models like BERT and modern LLMs for embeddings.

#### BERT-based Embeddings

```python
compute_embeddings(
    ngrams=chunks,
    embedding_type='transformer',
    model_name='prajjwal1/bert-tiny'
)
```

#### LLM-based Embeddings

```python
compute_embeddings(
    ngrams=chunks,
    embedding_type='llm',
    llm_provider='anthropic'  # or 'openai'
)
```

### 5. Prompt Templates

Manage your RAG prompts through template files:

```txt
Context: {context}

Question: {query}

Please provide a detailed answer based on the context above.
```

### 6. Limitations and Design Philosophy

#### RAG Limitations

1. **Context Window Constraints**
   - Limited by model's context window
   - Tradeoff between context length and computational cost

2. **Retrieval Quality**
   - Dependent on chunking strategy
   - Affected by embedding quality
   - Limited by database search capabilities

#### Why Not LangChain?

AcceleRAG was created to address several issues with frameworks like LangChain:

1. **Inversion of Control Risks**
   - Hidden complexity in abstraction layers
   - Difficult to debug and optimize
   - Security vulnerabilities in black-box components

2. **Scalability Issues**
   - Overhead from unnecessary abstractions
   - Limited control over resource usage
   - Difficult to customize for specific needs

3. **Transparency**
   - Direct access to all components
   - Clear data flow
   - Explicit error handling

### 7. Hallucination Evaluation

AcceleRAG includes a hallucination risk assessment system:

```python
chunk_scores, hall_risk = score_chunks(
    context=retrieved_chunks,
    llm_provider='anthropic'
)
```

The system evaluates:
- Relevance of retrieved chunks
- Consistency between chunks
- Confidence in the answer
- Potential for hallucination

## Usage Example

```python
from accelerag import process_corpus, fetch_top_k

# Index documents
process_corpus(
    corpus_dir="documents",
    ngram_size=3,
    db_type="sqlite",
    db_params={"dbname": "embeddings.db"}
)

# Query documents
results = fetch_top_k(
    query="What is machine learning?",
    db_params={"dbname": "embeddings.db"},
    tag_hierarchy=tag_hierarchy,
    k=5
)
```

## License

AcceleRAG is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This license ensures:

1. **Free Use for Research and Non-Commercial Purposes**
   - Free to use, modify, and distribute
   - Source code must remain open
   - Modifications must be shared under the same license

2. **Commercial Use Requirements**
   - Commercial use requires a separate commercial license
   - Contact the maintainers for commercial licensing options
   - Commercial licenses include additional support and features

3. **Network Service Requirements**
   - If you run AcceleRAG as a network service, you must provide source code
   - Users must be able to access the source code of the running service
   - Modifications to the service must be shared with users

For more details, see the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. By contributing, you agree to license your contributions under the same AGPL-3.0 license.

## Contact

For commercial licensing inquiries, please contact the maintainers at [contact email]. 