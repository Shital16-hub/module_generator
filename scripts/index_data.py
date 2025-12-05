"""
Index test data into Qdrant vector database
"""

import json
from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(url="http://localhost:6333")
collection_name = "tasconnect_knowledge_base"


def create_collection():
    """Create Qdrant collection"""
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2 dimension
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{collection_name}' created")


def index_jira_stories(file_path: str):
    """Index JIRA stories"""
    with open(file_path, 'r', encoding='utf-8') as f:
        stories = json.load(f)
    
    points = []
    for story in tqdm(stories, desc="Indexing JIRA stories"):
        # Create searchable content
        content = f"""
{story['title']}

{story['description']}

Acceptance Criteria:
{chr(10).join(f'- {ac}' for ac in story['acceptance_criteria'])}

Module: {story['module']}
Status: {story['status']}
Priority: {story['priority']}
"""
        
        # Generate embedding
        embedding = model.encode(content).tolist()
        
        # Create point with rich metadata
        point = PointStruct(
            id=hash(story['story_id']) & 0x7FFFFFFF,  # Positive int ID
            vector=embedding,
            payload={
                "id": story['story_id'],
                "source": "JIRA",
                "type": "user_story",
                "module": story['module'],
                "title": story['title'],
                "description": story['description'],
                "acceptance_criteria": story['acceptance_criteria'],
                "status": story['status'],
                "priority": story['priority'],
                "epic": story['epic'],
                "story_points": story['story_points'],
                "labels": story['labels'],
                "content": content,  # Full searchable content
                # Store relationships
                "tested_by": story['linked_issues'].get('tested_by', []),
                "relates_to": story['linked_issues'].get('relates_to', []),
                "blocks": story['linked_issues'].get('blocks', []),
                "depends_on": story['linked_issues'].get('depends_on', [])
            }
        )
        points.append(point)
    
    # Batch upload
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Indexed {len(points)} JIRA stories")


def index_confluence_docs(file_path: str):
    """Index Confluence documentation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    points = []
    for doc in tqdm(docs, desc="Indexing Confluence docs"):
        # Create searchable content
        content = f"""
{doc['title']}

{doc['content']}

Module: {doc['module']}
Type: {doc['type']}
"""
        
        # Generate embedding
        embedding = model.encode(content).tolist()
        
        # Create point
        point = PointStruct(
            id=hash(doc['doc_id']) & 0x7FFFFFFF,
            vector=embedding,
            payload={
                "id": doc['doc_id'],
                "source": "Confluence",
                "type": "documentation",
                "module": doc['module'],
                "title": doc['title'],
                "content": doc['content'][:5000],  # Truncate very long content
                "full_content": doc['content'],
                "doc_type": doc['type'],
                "labels": doc['labels'],
                "page_url": doc['page_url'],
                # Store relationships
                "linked_jira_issues": doc['linked_jira_issues'],
                "linked_test_cases": doc['linked_test_cases']
            }
        )
        points.append(point)
    
    # Batch upload
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Indexed {len(points)} Confluence docs")


def index_zephyr_tests(file_path: str):
    """Index Zephyr test cases"""
    with open(file_path, 'r', encoding='utf-8') as f:
        tests = json.load(f)
    
    points = []
    for test in tqdm(tests, desc="Indexing Zephyr tests"):
        # Create searchable content with test steps
        steps_text = "\n".join([
            f"Step {step['step_number']}: {step['action']}\nExpected: {step['expected_result']}"
            for step in test['test_steps']
        ])
        
        content = f"""
{test['title']}

Objective: {test['objective']}

Test Steps:
{steps_text}

Module: {test['module']}
Priority: {test['priority']}
Type: {test['test_type']}
"""
        
        # Generate embedding
        embedding = model.encode(content).tolist()
        
        # Create point
        point = PointStruct(
            id=hash(test['test_id']) & 0x7FFFFFFF,
            vector=embedding,
            payload={
                "id": test['test_id'],
                "source": "Zephyr",
                "type": "test_case",
                "module": test['module'],
                "title": test['title'],
                "objective": test['objective'],
                "priority": test['priority'],
                "test_type": test['test_type'],
                "automation_status": test['automation_status'],
                "test_steps": test['test_steps'],
                "preconditions": test['preconditions'],
                "content": content,
                # Store relationships
                "linked_stories": test['linked_stories'],
                "linked_requirements": test.get('linked_requirements', [])
            }
        )
        points.append(point)
    
    # Batch upload
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Indexed {len(points)} Zephyr test cases")


def main():
    """Main indexing pipeline"""
    print("🚀 Starting data indexing pipeline...\n")
    
    # Paths to test data
    data_dir = Path("test_data")
    jira_file = data_dir / "jira_examples.json"
    confluence_file = data_dir / "confluence_examples.json"
    zephyr_file = data_dir / "zephyr_examples.json"
    
    # Create collection
    create_collection()
    print()
    
    # Index all data
    index_jira_stories(str(jira_file))
    index_confluence_docs(str(confluence_file))
    index_zephyr_tests(str(zephyr_file))
    
    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"\n✅ Total indexed: {collection_info.points_count} documents")
    print(f"📊 Collection: {collection_name}")
    print("🎉 Indexing complete!")


if __name__ == "__main__":
    main()