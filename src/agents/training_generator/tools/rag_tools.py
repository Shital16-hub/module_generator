"""
RAG Tools for Training Generator Agent

Using LangChain with proper Qdrant filter format.
"""

from typing import List, Dict, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Handle imports for both direct execution and module import
try:
    from ..config import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from agents.training_generator.config import config


# ============================================================================
# INITIALIZATION
# ============================================================================

class RAGTools:
    """
    RAG tools using LangChain abstractions.
    
    Easy to switch vector stores - just change the client initialization.
    """
    
    def __init__(self):
        """Initialize RAG tools with LangChain vector store"""
        
        # Initialize embeddings (HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            check_compatibility=False
        )
        
        # Create LangChain vector store wrapper
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=self.embeddings
        )
        
    # ========================================================================
    # SEMANTIC SEARCH TOOLS
    # ========================================================================
    
    def search_stories(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for JIRA user stories using semantic search.
        
        Args:
            query: Search query
            module: Filter by module (e.g., "Payment", "Authentication")
            top_k: Number of results to return
            
        Returns:
            List of story documents with metadata
        """
        top_k = top_k or config.SEARCH_TOP_K
        
        # Build Qdrant filter (not dict!)
        filter_conditions = [
            FieldCondition(key="source", match=MatchValue(value="JIRA")),
            FieldCondition(key="type", match=MatchValue(value="user_story"))
        ]
        
        # Add module filter if specified
        if module:
            filter_conditions.append(
                FieldCondition(key="module", match=MatchValue(value=module))
            )
        
        search_filter = Filter(must=filter_conditions)
        
        # Search using LangChain
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=search_filter
        )
        
        # Format results
        stories = []
        for doc, score in docs_with_scores:
            # Only include if score meets threshold
            if score < config.MIN_RELEVANCE_SCORE:
                continue
                
            stories.append({
                'id': doc.metadata['id'],
                'content': doc.page_content,
                'metadata': {
                    'source': doc.metadata['source'],
                    'type': doc.metadata['type'],
                    'module': doc.metadata['module'],
                    'title': doc.metadata['title'],
                    'description': doc.metadata.get('description', ''),
                    'acceptance_criteria': doc.metadata.get('acceptance_criteria', []),
                    'status': doc.metadata.get('status', ''),
                    'priority': doc.metadata.get('priority', ''),
                    'epic': doc.metadata.get('epic', ''),
                    'story_points': doc.metadata.get('story_points', 0),
                    'labels': doc.metadata.get('labels', [])
                },
                'score': score,
                # Store relationships for later use
                '_relationships': {
                    'tested_by': doc.metadata.get('tested_by', []),
                    'relates_to': doc.metadata.get('relates_to', []),
                    'blocks': doc.metadata.get('blocks', []),
                    'depends_on': doc.metadata.get('depends_on', [])
                }
            })
        
        return stories
    
    def search_documentation(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for Confluence documentation using semantic search.
        
        Args:
            query: Search query
            module: Filter by module
            top_k: Number of results to return
            
        Returns:
            List of documentation with metadata
        """
        top_k = top_k or config.SEARCH_TOP_K
        
        # Build Qdrant filter
        filter_conditions = [
            FieldCondition(key="source", match=MatchValue(value="Confluence")),
            FieldCondition(key="type", match=MatchValue(value="documentation"))
        ]
        
        # Add module filter if specified
        if module:
            filter_conditions.append(
                FieldCondition(key="module", match=MatchValue(value=module))
            )
        
        search_filter = Filter(must=filter_conditions)
        
        # Search using LangChain
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=search_filter
        )
        
        # Format results
        docs = []
        for doc, score in docs_with_scores:
            # Only include if score meets threshold
            if score < config.MIN_RELEVANCE_SCORE:
                continue
                
            docs.append({
                'id': doc.metadata['id'],
                'content': doc.page_content,
                'metadata': {
                    'source': doc.metadata['source'],
                    'type': doc.metadata['type'],
                    'module': doc.metadata['module'],
                    'title': doc.metadata['title'],
                    'doc_type': doc.metadata.get('doc_type', ''),
                    'labels': doc.metadata.get('labels', []),
                    'page_url': doc.metadata.get('page_url', '')
                },
                'score': score,
                # Store relationships
                '_relationships': {
                    'linked_jira_issues': doc.metadata.get('linked_jira_issues', []),
                    'linked_test_cases': doc.metadata.get('linked_test_cases', [])
                }
            })
        
        return docs
    
    def search_test_cases(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for Zephyr test cases using semantic search.
        
        Args:
            query: Search query
            module: Filter by module
            top_k: Number of results to return
            
        Returns:
            List of test cases with metadata
        """
        top_k = top_k or config.SEARCH_TOP_K
        
        # Build Qdrant filter
        filter_conditions = [
            FieldCondition(key="source", match=MatchValue(value="Zephyr")),
            FieldCondition(key="type", match=MatchValue(value="test_case"))
        ]
        
        # Add module filter if specified
        if module:
            filter_conditions.append(
                FieldCondition(key="module", match=MatchValue(value=module))
            )
        
        search_filter = Filter(must=filter_conditions)
        
        # Search using LangChain
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=search_filter
        )
        
        # Format results
        tests = []
        for doc, score in docs_with_scores:
            # Only include if score meets threshold
            if score < config.MIN_RELEVANCE_SCORE:
                continue
                
            tests.append({
                'id': doc.metadata['id'],
                'content': doc.page_content,
                'metadata': {
                    'source': doc.metadata['source'],
                    'type': doc.metadata['type'],
                    'module': doc.metadata['module'],
                    'title': doc.metadata['title'],
                    'objective': doc.metadata.get('objective', ''),
                    'priority': doc.metadata.get('priority', ''),
                    'test_type': doc.metadata.get('test_type', ''),
                    'automation_status': doc.metadata.get('automation_status', ''),
                    'test_steps': doc.metadata.get('test_steps', []),
                    'preconditions': doc.metadata.get('preconditions', [])
                },
                'score': score,
                # Store relationships
                '_relationships': {
                    'linked_stories': doc.metadata.get('linked_stories', []),
                    'linked_requirements': doc.metadata.get('linked_requirements', [])
                }
            })
        
        return tests
    
    # ========================================================================
    # RELATIONSHIP-BASED RETRIEVAL
    # ========================================================================
    
    def find_test_cases_by_stories(self, story_ids: List[str]) -> Dict[str, List[str]]:
        """
        Find test cases linked to given story IDs.
        
        Args:
            story_ids: List of JIRA story IDs
            
        Returns:
            Dictionary mapping story_id -> list of test_case_ids
        """
        # Get the stories to extract relationships
        stories = self.batch_retrieve_by_ids(story_ids, source="JIRA")
        
        # Build relationship map
        story_test_map = {}
        for story in stories:
            story_id = story['id']
            tested_by = story.get('_relationships', {}).get('tested_by', [])
            story_test_map[story_id] = tested_by
        
        return story_test_map
    
    # ========================================================================
    # BATCH RETRIEVAL
    # ========================================================================
    
    def batch_retrieve_by_ids(
        self,
        ids: List[str],
        source: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve multiple documents by their IDs.
        
        Args:
            ids: List of document IDs
            source: Filter by source (JIRA, Confluence, Zephyr)
            
        Returns:
            List of documents
        """
        if not ids:
            return []
        
        results = []
        
        for doc_id in ids:
            # Build Qdrant filter for exact ID match
            filter_conditions = [
                FieldCondition(key="id", match=MatchValue(value=doc_id))
            ]
            
            # Add source filter if specified
            if source:
                filter_conditions.append(
                    FieldCondition(key="source", match=MatchValue(value=source))
                )
            
            search_filter = Filter(must=filter_conditions)
            
            # Use direct Qdrant scroll for exact ID retrieval (more efficient)
            hits = self.client.scroll(
                collection_name=config.QDRANT_COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=1,
                with_payload=True
            )[0]
            
            if hits:
                hit = hits[0]
                results.append({
                    'id': hit.payload['id'],
                    'content': hit.payload.get('content', ''),
                    'metadata': {k: v for k, v in hit.payload.items() 
                                if k not in ['content', 'id', 'full_content']},
                    '_relationships': {
                        'tested_by': hit.payload.get('tested_by', []),
                        'relates_to': hit.payload.get('relates_to', []),
                        'blocks': hit.payload.get('blocks', []),
                        'depends_on': hit.payload.get('depends_on', []),
                        'linked_jira_issues': hit.payload.get('linked_jira_issues', []),
                        'linked_test_cases': hit.payload.get('linked_test_cases', []),
                        'linked_stories': hit.payload.get('linked_stories', []),
                        'linked_requirements': hit.payload.get('linked_requirements', [])
                    }
                })
        
        return results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection"""
        info = self.client.get_collection(config.QDRANT_COLLECTION_NAME)
        return {
            'total_documents': info.points_count,
            'collection_name': config.QDRANT_COLLECTION_NAME,
            'vector_size': info.config.params.vectors.size
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

rag_tools = RAGTools()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def search_stories(query: str, module: Optional[str] = None, top_k: int = None) -> List[Dict]:
    """Search for JIRA stories"""
    return rag_tools.search_stories(query, module, top_k)


def search_documentation(query: str, module: Optional[str] = None, top_k: int = None) -> List[Dict]:
    """Search for Confluence documentation"""
    return rag_tools.search_documentation(query, module, top_k)


def search_test_cases(query: str, module: Optional[str] = None, top_k: int = None) -> List[Dict]:
    """Search for Zephyr test cases"""
    return rag_tools.search_test_cases(query, module, top_k)


def find_test_cases_by_stories(story_ids: List[str]) -> Dict[str, List[str]]:
    """Find test cases linked to stories"""
    return rag_tools.find_test_cases_by_stories(story_ids)


def batch_retrieve_by_ids(ids: List[str], source: Optional[str] = None) -> List[Dict]:
    """Batch retrieve documents by IDs"""
    return rag_tools.batch_retrieve_by_ids(ids, source)