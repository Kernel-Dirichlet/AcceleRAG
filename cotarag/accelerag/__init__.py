"""
AcceleRAG __init__
"""

# Base classes - users should subclass these when using the AcceleRAG engine
from .base_classes import (
        Embedder,
        Scorer,
        Indexer,
        Retriever,
        QueryEngine,
        PromptCache
        )

# Embedders - these will be responsible for embedding docs and user queries to be used by the Indexers
from .embedders import (
        TextEmbedder,
        ImageEmbedder
        )


# Indexers - these are responsible for the indexing step in RAG pipelines
from .indexers import (
        TextIndexer,
        ImageIndexer
        )

# Retrievers - these are responsible for the retrieval step in RAG pipelines
from .retrievers import (
        TextRetriever,
        ImageRetriever,
        is_arxiv_query,
        route_query
        )

# QueryEngines - these are LLM interfaces that take the user query + context from the retriever and passes to an LLM
from .query_engines import (
        AnthropicEngine,
        OpenAIEngine
        )

# Scorers - these are responsible for scoring/evaluation of LLM outputs for hallucination and quality 
from .scorers import (
        DefaultScorer
        )
# Cacher - these are responsble for custom prompt caching when not using the RAGManager default cache
from .cachers import (
        DefaultCache
        )

# Managers + utilities - RAGManager primary class and additional utilities 
from .managers import RAGManager
from .arxiv_scraping_utils import (
        WebScraper,
        ArxivScraper,
        create_document_hierarchy,
        )

from .arxiv_category_mapper import (
        get_category_description,
        get_related_categories,
        suggest_tables_for_query,
        sanitize_table_name,
        PRIMARY_CATEGORIES,
        CS_SUBCATEGORIES,
        MATH_SUBCATEGORIES
        )

from .preprocessors import *

__all__ = [
        "Embedder",
        "Indexer",
        "Retriever",
        "QueryEngine",
        "Scorer",
        "PromptCache",
        "TextEmbedder",
        "ImageEmbedder",
        "TextIndexer",
        "ImageIndexer",
        "TextRetriever",
        "ImageRetriever",
        "is_arxiv_query",
        "route_query",
        "AnthropicEngine",
        "OpenAIEngine",
        "DefaultScorer",
        "DefaultCache",
        "RAGManager",
        "WebScraper",
        "ArxivScraper",
        "get_category_description",
        "get_related_categories",
        "suggest_tables_for_query",
        "sanitize_table_name",
        "create_document_hierarchy",
        "PRIMARY_CATEGORIES",
        "CS_SUBCATEGORIES",
        "MATH_SUBCATEGORIES",
    ]
        

