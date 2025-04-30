import os
import json
import logging
from base_classes import Indexer
from web_scraping_utils import ArxivScraper, create_document_hierarchy
from indexing_utils import process_corpus
from abc import ABC, abstractmethod
import anthropic
from openai import OpenAI

class AgenticIndexer(ABC):
    """Base class for agentic indexing implementations."""
    def __init__(self, output_dir='agentic_papers', max_docs=100):
        self.output_dir = output_dir
        self.max_docs = max_docs
        os.makedirs(output_dir, exist_ok=True)
        
    @abstractmethod
    def create_hierarchy(self, query, goals):
        """Create a document hierarchy based on query and goals."""
        pass
        
    @abstractmethod
    def index(self, **kwargs):
        """Index documents using implementation-specific parameters.
        
        Args:
            **kwargs: Implementation-specific parameters for indexing
        """
        pass
        
    @abstractmethod
    def retrieve(self, query, top_k=5):
        """Retrieve relevant chunks from the database."""
        pass

class ArxivAgent(AgenticIndexer):
    """Agent for handling ArXiv documents."""
    def __init__(self, output_dir='agentic_papers', max_docs=100, llm_provider='anthropic'):
        super().__init__(output_dir, max_docs)
        self.scraper = ArxivScraper()
        self.llm_provider = llm_provider
        self.tag_hierarchy = None
        self.model_name = 'huawei-noah/TinyBERT_General_4L_312D'  # Default to TinyBERT
        
    def create_hierarchy(self, query, goals):
        """Create document hierarchy using LLM.
        
        Args:
            query: Query string used to fetch papers
            goals: User goals for organizing papers
            
        Returns:
            dict: Created hierarchy structure
        """
        try:
            # Fetch documents
            documents = self.scraper.fetch_documents(query, self.max_docs)
            
            # Load hierarchy creation prompt
            with open('./prompts/create_hierarchy_prompt.txt', 'r') as f:
                prompt_template = f.read()
                
            # Format prompt with documents and goals
            prompt = prompt_template.format(
                goals=goals,
                documents="\n".join([f"Document {i}:\n{doc}" for i, doc in enumerate(documents)])
            )
            
            # Get hierarchy from LLM
            if self.llm_provider == 'anthropic':
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                hierarchy_json = response.content[0].text
            else:
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                hierarchy_json = response.choices[0].message.content
                
            # Parse hierarchy JSON and store as class attribute
            self.tag_hierarchy = json.loads(hierarchy_json)
            return self.tag_hierarchy
            
        except Exception as e:
            logging.error(f"Error creating hierarchy: {str(e)}")
            raise
            
    def index(self, **kwargs):
        """Index documents using the agent.
        
        Args:
            query: Query string to search for papers
            db_params: Database parameters
            batch_size: Batch size for processing
            max_docs: Maximum number of documents to index (optional)
        """
        try:
            query = kwargs.get('query')
            db_params = kwargs.get('db_params')
            batch_size = kwargs.get('batch_size', 100)
            max_docs = kwargs.get('max_docs', self.max_docs)
            
            if not query or not db_params:
                raise ValueError("query and db_params are required for indexing")
                
            # Fetch documents
            documents = self.scraper.fetch_documents(query, max_docs)
            
            # Create temporary directory for documents
            temp_dir = os.path.join(self.output_dir, 'temp_docs')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save documents
            for i, doc in enumerate(documents):
                # Extract relevant fields from document dictionary
                doc_content = f"Title: {doc.get('title', '')}\n"
                doc_content += f"Authors: {', '.join(doc.get('authors', []))}\n"
                doc_content += f"Abstract: {doc.get('abstract', '')}\n"
                doc_content += f"Published: {doc.get('published', '')}\n"
                doc_content += f"URL: {doc.get('url', '')}\n"
                
                with open(os.path.join(temp_dir, f'doc_{i}.txt'), 'w') as f:
                    f.write(doc_content)
            
            # Create hierarchy if not already created
            if self.tag_hierarchy is None:
                self.create_hierarchy(query, "Organize papers by mathematical topics and subtopics")
                
            # Process corpus using TinyBERT for embeddings
            process_corpus(
                corpus_dir=temp_dir,
                tag_hierarchy=self.tag_hierarchy,
                db_params=db_params,
                batch_size=batch_size,
                model_name=self.model_name  # Use TinyBERT for embeddings
            )
            
            # Clean up temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            
        except Exception as e:
            logging.error(f"Error indexing documents: {str(e)}")
            raise
            
    def retrieve(self, query, top_k=5):
        """Retrieve relevant chunks using the agent."""
        try:
            if self.tag_hierarchy is None:
                raise ValueError("Tag hierarchy not initialized. Call create_hierarchy first.")
                
            # Use the scraper's retrieval functionality with TinyBERT embeddings
            return self.scraper.retrieve(query, self.tag_hierarchy, top_k, model_name=self.model_name)
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            raise 
