"""
Index JIRA / Confluence / Zephyr examples into Qdrant using LangChain Documents.
"""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.agents.training_generator.config import config


# ---------- Adapters: raw JSON -> Document ----------

def jira_to_document(story: dict) -> Document:
    text_parts = [
        story.get("title", ""),
        story.get("description", ""),
        "Acceptance Criteria:",
    ]
    text_parts.extend(story.get("acceptance_criteria", []))
    page_content = "\n\n".join(p for p in text_parts if p)

    metadata = {
        "id": story.get("story_id"),
        "source": "JIRA",
        "type": "user_story",
        "module": story.get("module"),
        "title": story.get("title"),
        "status": story.get("status"),
        "priority": story.get("priority"),
        "epic": story.get("epic"),
        "story_points": story.get("story_points"),
        "labels": story.get("labels", []),
        "tested_by": story.get("linked_issues", {}).get("tested_by", []),
        "relates_to": story.get("linked_issues", {}).get("relates_to", []),
        "blocks": story.get("linked_issues", {}).get("blocks", []),
        "depends_on": story.get("linked_issues", {}).get("depends_on", []),
    }
    return Document(page_content=page_content, metadata=metadata)


def confluence_to_document(doc: dict) -> Document:
    text_parts = [
        doc.get("title", ""),
        doc.get("content", ""),
    ]
    page_content = "\n\n".join(p for p in text_parts if p)

    metadata = {
        "id": doc.get("doc_id"),
        "source": "Confluence",
        "type": "documentation",
        "module": doc.get("module"),
        "title": doc.get("title"),
        "doc_type": doc.get("type"),
        "labels": doc.get("labels", []),
        "page_url": doc.get("page_url", ""),
        "linked_jira_issues": doc.get("linked_jira_issues", []),
        "linked_test_cases": doc.get("linked_test_cases", []),
    }
    return Document(page_content=page_content, metadata=metadata)


def zephyr_to_document(test: dict) -> Document:
    steps_text = "\n".join(
        f"Step {s['step_number']}: {s['action']}\nExpected: {s['expected_result']}"
        for s in test.get("test_steps", [])
    )

    text_parts = [
        test.get("title", ""),
        f"Objective: {test.get('objective', '')}",
        "Test Steps:",
        steps_text,
    ]
    page_content = "\n\n".join(p for p in text_parts if p)

    metadata = {
        "id": test.get("test_id"),
        "source": "Zephyr",
        "type": "test_case",
        "module": test.get("module"),
        "title": test.get("title"),
        "objective": test.get("objective"),
        "priority": test.get("priority"),
        "test_type": test.get("test_type"),
        "automation_status": test.get("automation_status"),
        "linked_stories": test.get("linked_stories", []),
        "linked_requirements": test.get("linked_requirements", []),
    }
    return Document(page_content=page_content, metadata=metadata)


# ---------- Helpers ----------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base = Path("test_data")  # adjust if needed

    jira_raw = load_json(base / "jira_examples.json")
    conf_raw = load_json(base / "confluence_examples.json")
    zep_raw = load_json(base / "zephyr_examples.json")

    jira_docs: List[Document] = [jira_to_document(s) for s in jira_raw]
    conf_docs: List[Document] = [confluence_to_document(d) for d in conf_raw]
    zep_docs: List[Document] = [zephyr_to_document(t) for t in zep_raw]

    all_docs = jira_docs + conf_docs + zep_docs

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Ensure collection exists with correct vector size
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    if not client.collection_exists(config.QDRANT_COLLECTION_NAME):
        sample_vec = embeddings.embed_query("sample text")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_vec),
                distance=Distance.COSINE,
            ),
        )

    # Index documents via LangChain
    QdrantVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )

    info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    print("✅ Indexing complete")
    print(f"Total points: {info.points_count}")
    print(f"Collection: {config.QDRANT_COLLECTION_NAME}")
    print(f"Vector size: {info.config.params.vectors.size}")


if __name__ == "__main__":
    main()
