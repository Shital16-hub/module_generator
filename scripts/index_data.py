"""
Index test data into Qdrant - Updated for separate files
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import config
from rich.console import Console
from rich.progress import Progress
import uuid

console = Console()

def load_all_documents(base_dir: Path):
    """Load all JSON documents from separate files"""
    documents = []
    
    # Load JIRA stories
    jira_dir = base_dir / "jira"
    if jira_dir.exists():
        for file_path in sorted(jira_dir.glob("jira_*.json")):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(list(jira_dir.glob('jira_*.json')))} JIRA files")
    
    # Load Confluence docs
    confluence_dir = base_dir / "confluence"
    if confluence_dir.exists():
        for file_path in sorted(confluence_dir.glob("confluence_*.json")):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(list(confluence_dir.glob('confluence_*.json')))} Confluence files")
    
    # Load Zephyr tests
    zephyr_dir = base_dir / "zephyr"
    if zephyr_dir.exists():
        for file_path in sorted(zephyr_dir.glob("zephyr_*.json")):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(list(zephyr_dir.glob('zephyr_*.json')))} Zephyr files")
    
    return documents

def create_text_for_embedding(doc: dict) -> str:
    """Create searchable text from document"""
    doc_type = doc.get('type', '')
    
    if doc_type == "User Story":
        return f"{doc.get('title', '')} {doc.get('description', '')} {doc.get('module', '')} {' '.join(doc.get('acceptance_criteria', []))}"
    elif doc_type in ["technical_documentation", "user_guide", "compliance_documentation", "operational_guide", "configuration_guide", "api_documentation", "analytics_guide"]:
        return f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('module', '')}"
    else:  # Test case
        return f"{doc.get('title', '')} {doc.get('objective', '')} {doc.get('module', '')}"

def index_documents():
    """Index all documents into Qdrant"""
    
    console.print("\n[bold cyan]🚀 Starting Indexing Process[/bold cyan]\n")
    
    # Initialize
    client = QdrantClient(url=config.QDRANT_URL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Load documents
    base_dir = project_root / "test_data"
    documents = load_all_documents(base_dir)
    
    console.print(f"\n📊 Total documents to index: [bold green]{len(documents)}[/bold green]\n")
    
    # Recreate collection
    console.print(f"🗑️  Deleting existing collection: {config.QDRANT_COLLECTION_NAME}")
    try:
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
    except:
        pass
    
    console.print(f"✨ Creating collection: {config.QDRANT_COLLECTION_NAME}")
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Index documents with progress bar
    points = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Indexing documents...", total=len(documents))
        
        for doc in documents:
            # Create embedding
            text = create_text_for_embedding(doc)
            embedding = model.encode(text).tolist()
            
            # Determine document type and ID
            if 'story_id' in doc:
                doc_type = "jira_story"
                doc_id = doc['story_id']
            elif 'doc_id' in doc:
                doc_type = "confluence_doc"
                doc_id = doc['doc_id']
            elif 'test_id' in doc:
                doc_type = "test_case"
                doc_id = doc['test_id']
            else:
                doc_type = "unknown"
                doc_id = str(uuid.uuid4())
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "document_id": doc_id,
                    "document_type": doc_type,
                    "module": doc.get('module', ''),
                    "content": json.dumps(doc)
                }
            )
            points.append(point)
            progress.update(task, advance=1)
    
    # Upload to Qdrant
    console.print("\n📤 Uploading to Qdrant...")
    client.upsert(
        collection_name=config.QDRANT_COLLECTION_NAME,
        points=points
    )
    
    # Verify
    collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    
    console.print("\n[bold green]✅ Indexing Complete![/bold green]\n")
    console.print(f"📊 Collection: {config.QDRANT_COLLECTION_NAME}")
    console.print(f"📈 Total vectors: {collection_info.points_count}")
    console.print(f"🎯 Vector size: {collection_info.config.params.vectors.size}")
    
    # Summary by type
    jira_count = len([d for d in documents if 'story_id' in d])
    conf_count = len([d for d in documents if 'doc_id' in d])
    test_count = len([d for d in documents if 'test_id' in d])
    
    console.print("\n📋 Document Breakdown:")
    console.print(f"  JIRA Stories:      {jira_count}")
    console.print(f"  Confluence Docs:   {conf_count}")
    console.print(f"  Zephyr Tests:      {test_count}")
    
    console.print("\n✨ Ready for queries!\n")

if __name__ == "__main__":
    index_documents()