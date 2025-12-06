#!/usr/bin/env python3
"""
Index JIRA / Confluence / Zephyr examples into Qdrant using LangChain Documents.
Supports multiple files: jira_examples*.json, zephyr_examples*.json, confluence_examples*.json
"""

import sys
from pathlib import Path
import json
from typing import List

# Fix imports when running script directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.training_generator.config import config
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from tqdm import tqdm

print("🚀 Starting data indexing pipeline...")

# ---------- Adapters: raw JSON -> Document ----------

def jira_to_document(story: dict) -> Document:
    """Convert JIRA story JSON to LangChain Document."""
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

def zephyr_to_document(test: dict) -> Document:
    """Convert Zephyr test case JSON to LangChain Document."""
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

def confluence_to_document(doc: dict) -> Document:
    """Convert Confluence doc JSON to LangChain Document."""
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

# ---------- Helpers ----------

def load_json(path: Path):
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Multi-file indexing ----------

def index_jira_files(base: Path) -> List[Document]:
    """Index all jira_examples*.json files in test_data folder."""
    jira_docs: List[Document] = []
    jira_files = list(base.glob("jira_examples*.json"))
    
    print(f"📂 Found {len(jira_files)} JIRA files in test_data/")
    
    for path in jira_files:
        print(f"  Indexing {path.name}...")
        data = load_json(path)
        for story in data:
            jira_docs.append(jira_to_document(story))
    
    return jira_docs

def index_zephyr_files(base: Path) -> List[Document]:
    """Index all zephyr_examples*.json files in test_data folder."""
    zephyr_docs: List[Document] = []
    zephyr_files = list(base.glob("zephyr_examples*.json"))
    
    print(f"📂 Found {len(zephyr_files)} Zephyr files in test_data/")
    
    for path in zephyr_files:
        print(f"  Indexing {path.name}...")
        data = load_json(path)
        for test in data:
            zephyr_docs.append(zephyr_to_document(test))
    
    return zephyr_docs

def index_confluence_files(base: Path) -> List[Document]:
    """Index confluence_examples*.json files (optional)."""
    conf_files = list(base.glob("confluence_examples*.json"))
    if not conf_files:
        print("ℹ️  No Confluence files found in test_data/ (skipping)")
        return []
    
    conf_docs: List[Document] = []
    print(f"📂 Found {len(conf_files)} Confluence files in test_data/")
    
    for path in conf_files:
        print(f"  Indexing {path.name}...")
        data = load_json(path)
        for doc in data:
            conf_docs.append(confluence_to_document(doc))
    
    return conf_docs

# ---------- Main ----------

def main():
    base = Path("test_data")  # Look INSIDE test_data folder
    
    if not base.exists():
        print(f"❌ test_data folder not found at: {base.absolute()}")
        print("Please create test_data/ folder with your JSON files.")
        return
    
    print(f"🔍 Searching for JSON files in: {base.absolute()}")
    
    # Load all documents
    jira_docs = index_jira_files(base)
    zephyr_docs = index_zephyr_files(base)
    conf_docs = index_confluence_files(base)
    
    all_docs = jira_docs + conf_docs + zephyr_docs
    
    print(f"\n📊 Loaded {len(all_docs)} total documents:")
    print(f"  - JIRA stories: {len(jira_docs)}")
    print(f"  - Zephyr tests: {len(zephyr_docs)}")
    print(f"  - Confluence docs: {len(conf_docs)}")
    
    if not all_docs:
        print("❌ No documents found! Make sure JSON files matching 'jira_examples*.json' or 'zephyr_examples*.json' are in test_data/")
        return
    
    # Setup embeddings and Qdrant
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    client = QdrantClient(
        url=config.QDRANT_URL, 
        api_key=config.QDRANT_API_KEY,
        check_compatibility=False,
    )
    
    # Delete and recreate collection
    collection_name = config.QDRANT_COLLECTION_NAME
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"🗑️  Deleted existing collection: {collection_name}")
    
    # Create new collection
    sample_vec = embeddings.embed_query("sample text")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(sample_vec),
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ Collection '{collection_name}' created")
    
    # Index documents
    print("\n🔄 Indexing documents...")
    vectorstore = QdrantVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=collection_name,
    )
    
    # Verify
    info = client.get_collection(collection_name)
    print("\n🎉 Indexing complete!")
    print(f"📊 Total documents: {info.points_count}")
    print(f"📋 Collection: {collection_name}")
    print(f"🧬 Vector size: {info.config.params.vectors.size}")

if __name__ == "__main__":
    main()
