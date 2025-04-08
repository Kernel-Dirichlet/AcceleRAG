"""
Query Assistance Pipeline

This module outlines how the system processes complex research-oriented queries by:
1. Routing the query to relevant topic tags using query_utils.route_query()
2. Retrieving relevant document chunks using retrieval_utils.fetch_top_k()
3. Ensuring answers are grounded in the retrieved evidence

Example query: "Summarize the most promising research directions in CS"

Pipeline steps:
1. Query Routing:
   - Maps query to hierarchical CS tags like ["Computer Science", "AI/ML"], ["Computer Science", "Systems"] etc.
   - Ensures we search relevant document collections

2. Evidence Retrieval:
   - Fetches semantically similar chunks from tagged document collections
   - Uses BERT embeddings to find top-k relevant passages
   - Retrieves from ArXiv papers organized by CS subcategories

3. Answer Generation:
   - Augments user query with retrieved evidence chunks
   - Requires answers to cite specific papers/chunks as sources
   - Maintains traceability between claims and source documents

4. Quality Control:
   - Answers must be supported by retrieved chunks
   - Claims require explicit citations to ArXiv papers
   - Responses indicate confidence based on evidence quality

The system prioritizes:
- Factual accuracy through evidence grounding
- Coverage across CS subfields via tag hierarchy
- Transparency in connecting claims to sources
- Up-to-date insights from recent ArXiv papers
"""
