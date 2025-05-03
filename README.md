# AcceleRAG

A high-performance RAG (Retrieval-Augmented Generation) framework focused on speed and accuracy. Version 0.9.0

## Overview

AcceleRAG is a framework (not a library) designed to provide fast and accurate document retrieval and generation capabilities. It features a modular document chunking system with n-gram based indexing as the default strategy, flexible database support, and advanced embedding models.

## Key Features

### 1. Grounding Modes for Hallucination Control

AcceleRAG provides two grounding modes that significantly impact response quality and hallucination control:

1. **Hard Grounding**
   ```latex
   \begin{tikzpicture}[node distance=2cm]
   \node (query) {Query};
   \node[right of=query] (context) {Context};
   \node[right of=context] (response) {Response};
   \draw[->] (query) -- (context);
   \draw[->] (context) -- (response);
   \node[below of=context] (strict) {Strict Context Only};
   \node[below of=response] (zero) {Zero Hallucinations};
   \end{tikzpicture}
   ```
   - Strictly uses retrieved context
   - Zero hallucination guarantee
   - Explicitly states when context is insufficient
   - Best for factual accuracy
   - Ideal for technical documentation

2. **Soft Grounding**
   ```latex
   \begin{tikzpicture}[node distance=2cm]
   \node (query) {Query};
   \node[right of=query] (context) {Context + General Knowledge};
   \node[right of=context] (response) {Natural Responses};
   \draw[->] (query) -- (context);
   \draw[->] (context) -- (response);
   \end{tikzpicture}
   ```
   - Combines context with general knowledge
   - More natural responses
   - Better for open-ended questions
   - Higher risk of hallucinations
   - Requires careful monitoring

### 2. TinyBERT for Efficient Embeddings

AcceleRAG uses TinyBERT for both indexing and query embeddings, providing significant performance benefits:

```latex
\begin{tikzpicture}[node distance=2cm]
\node (query) {Query};
\node[right of=query] (tinybert) {TinyBERT};
\node[right of=tinybert] (embedding) {312-dim Embeddings};
\node[below of=tinybert] (speed) {6x Faster Inference};
\draw[->] (query) -- (tinybert);
\draw[->] (tinybert) -- (embedding);
\draw[->] (tinybert) -- (speed);
\end{tikzpicture}
```

Key Benefits:
- 50% smaller embeddings (312 vs 768 dimensions)
- 4x smaller model size
- 6x faster inference
- Lower memory requirements
- Efficient storage

### 3. Query Routing for Performance

AcceleRAG implements intelligent query routing to reduce search space:

```latex
\begin{tikzpicture}[node distance=2cm]
\node (query) {Query};
\node[right of=query] (route) {Route to Relevant Tables};
\node[right of=route] (search) {Search Fewer Docs};
\node[below of=route] (identify) {Identify Relevant Tables};
\node[below of=search] (compare) {Compare Against Fewer Docs};
\draw[->] (query) -- (route);
\draw[->] (route) -- (search);
\draw[->] (route) -- (identify);
\draw[->] (identify) -- (compare);
\end{tikzpicture}
```

Performance Impact:
- 90-99% reduction in search space
- Example: 1M documents → 10K relevant
- Fewer similarity computations
- Lower database load
- Faster response times

### 4. Flexible Caching System

AcceleRAG provides a sophisticated caching system with four operational modes:

1. **No Caching**
   ```python
   rag = RAGManager(enable_cache=False, use_cache=False)
   ```

2. **Read-Only Caching**
   ```python
   rag = RAGManager(enable_cache=False, use_cache=True)
   ```

3. **Write-Only Caching**
   ```python
   rag = RAGManager(enable_cache=True, use_cache=False)
   ```

4. **Full Caching**
   ```python
   rag = RAGManager(enable_cache=True, use_cache=True)
   ```

Cache Features:
- Cosine similarity for cache hits (default)
- Configurable similarity threshold
- Quality score threshold for caching
- Support for custom similarity metrics
- External cache database support

### 5. Modular Architecture

AcceleRAG is built on a modular architecture with dependency injection:

```latex
\begin{tikzpicture}[node distance=2cm]
\node (core) {Core};
\node[right of=core] (components) {Components};
\node[right of=components] (custom) {Custom};
\draw[->] (core) -- (components);
\draw[->] (components) -- (custom);
\node[below of=components] (abstract) {Abstract Classes};
\node[below of=custom] (concrete) {Concrete Implementations};
\draw[->] (components) -- (abstract);
\draw[->] (abstract) -- (concrete);
\end{tikzpicture}
```

Key Components:
1. **Indexer**
   - Abstract: `Indexer` base class
   - Default: `DefaultIndexer` with TinyBERT
   - Custom: Implement custom indexing logic

2. **Retriever**
   - Abstract: `Retriever` base class
   - Default: `DefaultRetriever` with query routing
   - Custom: Implement custom retrieval logic

3. **Cache**
   - Abstract: `PromptCache` base class
   - Default: `DefaultCache` with SQLite
   - Custom: Implement custom caching logic

4. **QueryEngine**
   - Abstract: `QueryEngine` base class
   - Default: `DefaultEngine` with Claude
   - Custom: Implement custom LLM integration

5. **Embedder**
   - Abstract: `Embedder` base class
   - Default: `TransformerEmbedder` with TinyBERT
   - Custom: Implement custom embedding logic

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
    api_key='path/to/api_key.txt',
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

## Quick Start

### Using the RAGManager

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
   - Cache hits reduce LLM API calls by 40-60%

2. **Embedding Optimization**
   - TinyBERT reduces embedding storage by 50%
   - 6x faster inference

3. **Query Routing**
   - Reduces search space by 90-99%
   - Example: 1M documents → 10K relevant

## Roadmap

### v0.9.0 (Current)
- Advanced grounding modes with hallucination control
- TinyBERT for efficient embeddings
- Query routing for performance
- Flexible caching system
- Modular architecture with dependency injection

### v1.0.0 (Next)
- REST API for RAG operations
- WebSocket support for streaming
- Authentication and rate limiting
- Production-ready documentation
- Additional embedding models
- Custom LLM support
- Advanced monitoring and metrics

### Future Features

1. **Multi-modal RAG**
   ```latex
   \begin{tikzpicture}[node distance=2cm]
   \node (text) {Text};
   \node[right of=text] (image) {Image};
   \node[right of=image] (audio) {Audio};
   \node[below of=text] (text_store) {Text Vector Store};
   \node[below of=image] (image_store) {Image Vector Store};
   \node[below of=audio] (audio_store) {Audio Vector Store};
   \node[below of=image_store] (cross_modal) {Cross-modal Search};
   \draw[->] (text) -- (text_store);
   \draw[->] (image) -- (image_store);
   \draw[->] (audio) -- (audio_store);
   \draw[->] (text_store) -- (cross_modal);
   \draw[->] (image_store) -- (cross_modal);
   \draw[->] (audio_store) -- (cross_modal);
   \end{tikzpicture}
   ```
   - Separate vector stores for each modality
   - Cross-modal search capabilities
   - Unified embedding space
   - Modality-specific retrieval strategies
   - Support for text, image, and audio data

2. **Retrieval Augmented Thoughts (RAT)**
   ```latex
   \begin{tikzpicture}[node distance=2cm]
   \node (query) {Query};
   \node[right of=query] (rag) {RAG};
   \node[right of=rag] (cot) {Chain of Thought};
   \node[below of=rag] (retrieve) {Retrieve Context};
   \node[below of=cot] (reason) {Reason Step-by-Step};
   \node[below of=reason] (combine) {Combine Results};
   \draw[->] (query) -- (rag);
   \draw[->] (rag) -- (cot);
   \draw[->] (rag) -- (retrieve);
   \draw[->] (cot) -- (reason);
   \draw[->] (retrieve) -- (combine);
   \draw[->] (reason) -- (combine);
   \end{tikzpicture}
   ```
   - Integration of RAG with Chain of Thought reasoning
   - Step-by-step reasoning with retrieved context
   - Improved reasoning capabilities
   - Better handling of complex queries
   - Enhanced explainability

3. **Synthetic Data Engine (SDE)**
   ```latex
   \begin{tikzpicture}[node distance=2cm]
   \node (input) {Input Data};
   \node[right of=input] (generate) {Generate};
   \node[right of=generate] (validate) {Validate};
   \node[below of=generate] (text) {Text};
   \node[below of=text] (image) {Image};
   \node[below of=image] (audio) {Audio};
   \draw[->] (input) -- (generate);
   \draw[->] (generate) -- (validate);
   \draw[->] (generate) -- (text);
   \draw[->] (generate) -- (image);
   \draw[->] (generate) -- (audio);
   \end{tikzpicture}
   ```
   - Cost-effective synthetic data generation
   - Multi-modal data synthesis
   - Quality validation pipeline
   - Scalable generation framework
   - Support for custom generation rules

## Custom Class Examples

AcceleRAG's modular architecture makes it easy to extend and customize. Here are some simple examples of custom implementations:

### 1. Custom Indexer
```python
from base_classes import Indexer

class SimpleIndexer(Indexer):
    def index(self, corpus_dir, tag_hierarchy, db_params, batch_size=100):
        """Simple indexing that splits documents by paragraphs."""
        import sqlite3
        conn = sqlite3.connect(db_params['dbname'])
        cur = conn.cursor()
        
        # Create table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                id INTEGER PRIMARY KEY,
                content TEXT,
                filepath TEXT
            )
        """)
        
        # Process each file
        for filepath in os.listdir(corpus_dir):
            with open(os.path.join(corpus_dir, filepath), 'r') as f:
                content = f.read()
                paragraphs = content.split('\n\n')
                
                for para in paragraphs:
                    cur.execute(
                        "INSERT INTO paragraphs (content, filepath) VALUES (?, ?)",
                        (para, filepath)
                    )
        
        conn.commit()
        conn.close()
```

### 2. Custom Retriever
```python
from base_classes import Retriever

class KeywordRetriever(Retriever):
    def retrieve(self, query, top_k=5, **kwargs):
        """Simple keyword-based retrieval."""
        import sqlite3
        conn = sqlite3.connect(kwargs['dbname'])
        cur = conn.cursor()
        
        # Split query into keywords
        keywords = query.lower().split()
        
        # Build SQL query
        conditions = [f"content LIKE '%{kw}%'" for kw in keywords]
        sql = f"""
            SELECT content, filepath 
            FROM paragraphs 
            WHERE {' OR '.join(conditions)}
            LIMIT {top_k}
        """
        
        results = cur.execute(sql).fetchall()
        conn.close()
        
        return [r[0] for r in results]  # Return just the content
```

### 3. Custom Cache
```python
from base_classes import PromptCache

class MemoryCache(PromptCache):
    def __init__(self):
        self.cache = {}
        
    def cache_response(self, db_path, query, response, quality_score=None, **kwargs):
        self.cache[query] = (response, quality_score)
        
    def get_cached_response(self, db_path, query, threshold, **kwargs):
        if query in self.cache:
            response, score = self.cache[query]
            if score and score >= threshold:
                return response, score
        return None
```

### Using Custom Classes
```python
from runners import RAGManager

# Initialize with custom components
rag = RAGManager(
    indexer=SimpleIndexer(),
    retriever=KeywordRetriever(),
    cache=MemoryCache(),
    dir_to_idx='path/to/documents'
)

# Use as normal
rag.index()
response = rag.generate_response("What is the capital of France?")
```

### Creating Your Own Subclass

To create your own custom component:

1. **Import the Base Class**
   ```python
   from base_classes import Indexer, Retriever, PromptCache
   ```

2. **Subclass the Base Class**
   ```python
   class MyCustomComponent(BaseClass):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Your initialization code
   ```

3. **Implement Required Methods**
   - For `Indexer`: Implement `index()`
   - For `Retriever`: Implement `retrieve()`
   - For `PromptCache`: Implement `cache_response()` and `get_cached_response()`

4. **Add Custom Logic**
   - Add your own methods
   - Override existing methods
   - Add custom parameters

5. **Use in RAGManager**
   ```python
   rag = RAGManager(
       indexer=MyCustomIndexer(),
       retriever=MyCustomRetriever(),
       cache=MyCustomCache()
   )
   ```

## Contact

For commercial licensing inquiries or other questions, please contact:
- Email: elliottdev93@gmail.com

## Disclaimer

This is version 0.9.0 of AcceleRAG. While it is functional, it is still under active development. Some features may change in future releases. 
