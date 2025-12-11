"""
RAG Tools for Training Generator Agent - Fixed Filter Format
"""

from typing import List, Dict, Optional
import json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from ..config import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from agents.training_generator.config import config


class RAGTools:
    """RAG tools using LangChain with Qdrant vector store"""
    
    def __init__(self):
        """Initialize RAG tools with embeddings and vector store"""
        
        # Initialize embeddings - MUST match indexing!
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{config.EMBEDDING_MODEL_NAME}",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

        # Initialize LangChain vector store wrapper
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=config.QDRANT_COLLECTION_NAME,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

    def _format_result(self, doc, score: float) -> Dict:
        """Format a search result"""
        try:
            content = json.loads(doc.metadata.get('content', '{}'))
        except:
            content = {}
        
        # Extract ID from metadata
        doc_id = doc.metadata.get('document_id', 'unknown')
        
        return {
            "id": doc_id,
            "score": float(score),
            "document_type": doc.metadata.get('document_type', 'unknown'),
            "module": doc.metadata.get('module', ''),
            "metadata": content
        }

    def search_stories(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Search for JIRA user stories"""
        top_k = top_k or config.SEARCH_TOP_K
        
        # NO FILTER - LangChain will search all, we filter in Python
        # This is the most compatible approach
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k * 3  # Get more results to filter
        )
        
        # Filter in Python
        results = []
        for doc, score in docs_with_scores:
            # Check document type
            if doc.metadata.get('document_type') != 'jira_story':
                continue
            
            # Check module if specified
            if module and doc.metadata.get('module') != module:
                continue
            
            result = self._format_result(doc, score)
            result["metadata"]["source"] = "JIRA"
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results

    def search_documentation(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Search for Confluence documentation"""
        top_k = top_k or config.SEARCH_TOP_K
        
        # Get more results and filter in Python
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k * 3
        )
        
        results = []
        for doc, score in docs_with_scores:
            if doc.metadata.get('document_type') != 'confluence_doc':
                continue
            
            if module and doc.metadata.get('module') != module:
                continue
            
            result = self._format_result(doc, score)
            result["metadata"]["source"] = "Confluence"
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results

    def search_test_cases(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Search for Zephyr test cases"""
        top_k = top_k or config.SEARCH_TOP_K
        
        # Get more results and filter in Python
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k * 3
        )
        
        results = []
        for doc, score in docs_with_scores:
            if doc.metadata.get('document_type') != 'test_case':
                continue
            
            if module and doc.metadata.get('module') != module:
                continue
            
            result = self._format_result(doc, score)
            result["metadata"]["source"] = "Zephyr"
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results

    def find_test_cases_by_stories(self, story_ids: List[str]) -> Dict[str, List[str]]:
        """Find linked test cases for given stories"""
        
        story_test_map = {}
        
        for story_id in story_ids:
            # Search for the story
            docs = self.vector_store.similarity_search(
                query=story_id,
                k=20  # Get more to find the exact one
            )
            
            # Find the matching story by ID
            for doc in docs:
                if (doc.metadata.get('document_type') == 'jira_story' and 
                    doc.metadata.get('document_id') == story_id):
                    try:
                        content = json.loads(doc.metadata.get('content', '{}'))
                        linked_issues = content.get('linked_issues', {})
                        tested_by = linked_issues.get('tested_by', [])
                        story_test_map[story_id] = tested_by
                        break
                    except:
                        story_test_map[story_id] = []
            
            if story_id not in story_test_map:
                story_test_map[story_id] = []
        
        return story_test_map

    def batch_retrieve_by_ids(
        self,
        ids: List[str],
        source: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve documents by exact IDs"""
        
        results = []
        
        # Map source to document type
        doc_type_map = {
            "JIRA": "jira_story",
            "Confluence": "confluence_doc",
            "Zephyr": "test_case"
        }
        
        for doc_id in ids:
            # Do a broad search
            docs = self.vector_store.similarity_search(
                query=doc_id,
                k=50  # Get many results
            )
            
            # Find exact match
            for doc in docs:
                if doc.metadata.get('document_id') == doc_id:
                    # Check source if specified
                    if source:
                        if doc.metadata.get('document_type') != doc_type_map.get(source):
                            continue
                    
                    result = self._format_result(doc, 1.0)
                    results.append(result)
                    break
        
        return results

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        collection_info = self.client.get_collection(config.QDRANT_COLLECTION_NAME)
        
        # Handle different vector config structures
        try:
            if isinstance(collection_info.config.params.vectors, dict):
                vector_size = list(collection_info.config.params.vectors.values())[0].size
            else:
                vector_size = collection_info.config.params.vectors.size
        except:
            vector_size = 384  # Default
        
        return {
            "total_documents": collection_info.points_count,
            "collection_name": config.QDRANT_COLLECTION_NAME,
            "vector_size": vector_size
        }


# Singleton instance
rag_tools = RAGTools()


# Convenience functions
def search_stories(query: str, module: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]:
    return rag_tools.search_stories(query, module, top_k)


def search_documentation(query: str, module: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]:
    return rag_tools.search_documentation(query, module, top_k)


def search_test_cases(query: str, module: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]:
    return rag_tools.search_test_cases(query, module, top_k)


def find_test_cases_by_stories(story_ids: List[str]) -> Dict[str, List[str]]:
    return rag_tools.find_test_cases_by_stories(story_ids)


def batch_retrieve_by_ids(ids: List[str], source: Optional[str] = None) -> List[Dict]:
    return rag_tools.batch_retrieve_by_ids(ids, source)