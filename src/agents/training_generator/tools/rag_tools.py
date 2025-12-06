from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
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
    """
    RAG tools for LangChain Qdrant with NESTED metadata (from from_documents()).
    Keys: metadata.source, metadata.type, metadata.module, metadata.id
    Content: page_content
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            check_compatibility=False,
        )

        # LangChain stores: {"page_content": "...", "metadata": {...}}
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=config.QDRANT_COLLECTION_NAME,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

    # ------------------ search helpers ------------------

    def _search(
        self,
        query: str,
        must: List[FieldCondition],
        top_k: int,
    ):
        """Search using LangChain QdrantVectorStore."""
        search_filter = Filter(must=must)
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=search_filter,
        )

    def search_stories(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Search for JIRA user stories."""
        top_k = top_k or config.SEARCH_TOP_K

        # FIXED: Nested metadata structure from LangChain from_documents()
        must = [
            FieldCondition(key="metadata.source", match=MatchValue(value="JIRA")),
            FieldCondition(key="metadata.type", match=MatchValue(value="user_story")),
        ]
        if module:
            must.append(FieldCondition(key="metadata.module", match=MatchValue(value=module)))

        docs_with_scores = self._search(query, must, top_k)

        results: List[Dict] = []
        for doc, score in docs_with_scores:
            if score < config.MIN_RELEVANCE_SCORE:
                continue
            meta = doc.metadata or {}
            results.append(
                {
                    "id": meta.get("id", meta.get("story_id", "")),
                    "content": doc.page_content,
                    "metadata": meta,
                    "score": score,
                }
            )
        return results

    def search_documentation(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Search for Confluence documentation."""
        top_k = top_k or config.SEARCH_TOP_K

        must = [
            FieldCondition(key="metadata.source", match=MatchValue(value="Confluence")),
            FieldCondition(key="metadata.type", match=MatchValue(value="documentation")),
        ]
        if module:
            must.append(FieldCondition(key="metadata.module", match=MatchValue(value=module)))

        docs_with_scores = self._search(query, must, top_k)

        results: List[Dict] = []
        for doc, score in docs_with_scores:
            if score < config.MIN_RELEVANCE_SCORE:
                continue
            meta = doc.metadata or {}
            results.append(
                {
                    "id": meta.get("id", meta.get("doc_id", "")),
                    "content": doc.page_content,
                    "metadata": meta,
                    "score": score,
                }
            )
        return results

    def search_test_cases(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Search for Zephyr test cases."""
        top_k = top_k or config.SEARCH_TOP_K

        must = [
            FieldCondition(key="metadata.source", match=MatchValue(value="Zephyr")),
            FieldCondition(key="metadata.type", match=MatchValue(value="test_case")),
        ]
        if module:
            must.append(FieldCondition(key="metadata.module", match=MatchValue(value=module)))

        docs_with_scores = self._search(query, must, top_k)

        results: List[Dict] = []
        for doc, score in docs_with_scores:
            if score < config.MIN_RELEVANCE_SCORE:
                continue
            meta = doc.metadata or {}
            results.append(
                {
                    "id": meta.get("id", meta.get("test_id", "")),
                    "content": doc.page_content,
                    "metadata": meta,
                    "score": score,
                }
            )
        return results

    # ------------------ relationships & batch ------------------

    def find_test_cases_by_stories(self, story_ids: List[str]) -> Dict[str, List[str]]:
        """Find Zephyr test case IDs linked to given JIRA story IDs."""
        stories = self.batch_retrieve_by_ids(story_ids, source="JIRA")
        mapping: Dict[str, List[str]] = {}
        for story in stories:
            sid = story["id"]
            tested_by = story.get("_relationships", {}).get("tested_by", [])
            mapping[sid] = tested_by
        return mapping

    def batch_retrieve_by_ids(
        self,
        ids: List[str],
        source: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve documents by their IDs using Qdrant scroll."""
        if not ids:
            return []

        results: List[Dict] = []
        for doc_id in ids:
            # FIXED: Use nested metadata.id
            must = [FieldCondition(key="metadata.id", match=MatchValue(value=doc_id))]
            if source:
                must.append(FieldCondition(key="metadata.source", match=MatchValue(value=source)))
            search_filter = Filter(must=must)

            hits, _ = self.client.scroll(
                collection_name=config.QDRANT_COLLECTION_NAME,
                scroll_filter=search_filter,
                limit=1,
                with_payload=True,
            )
            if not hits:
                continue

            hit = hits[0]
            payload = hit.payload or {}
            # Extract nested metadata
            meta = payload.get("metadata", {})

            results.append(
                {
                    "id": meta.get("id"),
                    "content": payload.get("page_content", ""),
                    "metadata": meta,
                    "_relationships": {
                        "tested_by": meta.get("tested_by", []),
                        "relates_to": meta.get("relates_to", []),
                        "blocks": meta.get("blocks", []),
                        "depends_on": meta.get("depends_on", []),
                        "linked_jira_issues": meta.get("linked_jira_issues", []),
                        "linked_test_cases": meta.get("linked_test_cases", []),
                        "linked_stories": meta.get("linked_stories", []),
                        "linked_requirements": meta.get("linked_requirements", []),
                    },
                }
            )
        return results

    def get_collection_stats(self) -> Dict:
        info = self.client.get_collection(config.QDRANT_COLLECTION_NAME)
        return {
            "total_documents": info.points_count,
            "collection_name": config.QDRANT_COLLECTION_NAME,
            "vector_size": info.config.params.vectors.size,
        }


# Global instance
rag_tools = RAGTools()


# Export functions for test_rag_tools.py
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
