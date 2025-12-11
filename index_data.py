"""
Index test data into Qdrant - Pure LangChain Implementation
Easy to swap Qdrant with other vector stores (Pinecone, Chroma, etc.)
"""

import json
from pathlib import Path
from typing import List, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from src.agents.training_generator.config import config
from rich.console import Console
from rich.progress import Progress

console = Console()


def load_all_documents(base_dir: Path) -> List[Dict]:
    """Load all JSON documents from separate files"""
    documents = []
    
    # Load JIRA stories
    jira_dir = base_dir / "jira"
    if jira_dir.exists():
        jira_files = list(jira_dir.glob("jira_*.json"))
        for file_path in sorted(jira_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(jira_files)} JIRA files")
    else:
        console.print(f"⚠️  JIRA directory not found: {jira_dir}")
    
    # Load Confluence docs
    confluence_dir = base_dir / "confluence"
    if confluence_dir.exists():
        conf_files = list(confluence_dir.glob("confluence_*.json"))
        for file_path in sorted(conf_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(conf_files)} Confluence files")
    else:
        console.print(f"⚠️  Confluence directory not found: {confluence_dir}")
    
    # Load Zephyr tests
    zephyr_dir = base_dir / "zephyr"
    if zephyr_dir.exists():
        zephyr_files = list(zephyr_dir.glob("zephyr_*.json"))
        for file_path in sorted(zephyr_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.extend(data)
        console.print(f"✅ Loaded {len(zephyr_files)} Zephyr files")
    else:
        console.print(f"⚠️  Zephyr directory not found: {zephyr_dir}")
    
    return documents


def create_text_for_embedding(doc: dict) -> str:
    """Create searchable text from document"""
    doc_type = doc.get('type', '')
    
    if doc_type == "User Story":
        text_parts = [
            doc.get('title', ''),
            doc.get('description', ''),
            doc.get('module', ''),
            ' '.join(doc.get('acceptance_criteria', []))
        ]
        return ' '.join(filter(None, text_parts))
    
    elif doc_type in ["technical_documentation", "user_guide", "compliance_documentation", 
                      "operational_guide", "configuration_guide", "api_documentation", 
                      "analytics_guide"]:
        text_parts = [
            doc.get('title', ''),
            doc.get('content', ''),
            doc.get('module', '')
        ]
        return ' '.join(filter(None, text_parts))
    
    else:  # Test case
        text_parts = [
            doc.get('title', ''),
            doc.get('objective', ''),
            doc.get('module', '')
        ]
        return ' '.join(filter(None, text_parts))


def convert_to_langchain_documents(documents: List[Dict]) -> List[Document]:
    """
    Convert raw documents to LangChain Document objects
    
    LangChain Document structure:
    - page_content: The main searchable text
    - metadata: Dictionary of all metadata fields
    """
    langchain_docs = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Converting to LangChain documents...", total=len(documents))
        
        for doc in documents:
            # Create searchable text
            page_content = create_text_for_embedding(doc)
            
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
                doc_id = "unknown"
            
            # Create metadata dictionary
            metadata = {
                "document_id": doc_id,
                "document_type": doc_type,
                "module": doc.get('module', ''),
                "content": json.dumps(doc)  # Store full document as JSON
            }
            
            # Create LangChain Document
            langchain_doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            
            langchain_docs.append(langchain_doc)
            progress.update(task, advance=1)
    
    return langchain_docs


def index_documents():
    """
    Index documents using pure LangChain
    
    This implementation uses only LangChain abstractions, making it easy to:
    - Swap Qdrant with Pinecone, Chroma, Weaviate, etc.
    - Change embedding models
    - Modify indexing strategy
    
    Just change QdrantVectorStore to another vector store class!
    """
    
    console.print("\n[bold cyan]🚀 Starting Indexing Process (Pure LangChain)[/bold cyan]\n")
    
    # Configuration
    console.print(f"⚙️  Configuration:")
    console.print(f"   Vector Store: Qdrant")
    console.print(f"   Qdrant URL: {config.QDRANT_URL}")
    console.print(f"   Collection: {config.QDRANT_COLLECTION_NAME}")
    console.print(f"   Embedding Model: sentence-transformers/{config.EMBEDDING_MODEL_NAME}")
    
    # Initialize embeddings
    console.print(f"\n📥 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{config.EMBEDDING_MODEL_NAME}",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    console.print("✅ Embedding model loaded")
    
    # Test Qdrant connection
    console.print(f"\n🔌 Testing Qdrant connection...")
    try:
        test_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        collections = test_client.get_collections()
        console.print(f"✅ Connected to Qdrant (found {len(collections.collections)} collections)")
        
        # Delete existing collection if it exists
        try:
            test_client.delete_collection(config.QDRANT_COLLECTION_NAME)
            console.print(f"🗑️  Deleted existing collection: {config.QDRANT_COLLECTION_NAME}")
        except:
            console.print(f"ℹ️  No existing collection to delete")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to connect to Qdrant: {e}[/bold red]")
        console.print("\n💡 Make sure Qdrant is running:")
        console.print("   docker run -p 6333:6333 qdrant/qdrant")
        return
    
    # Load documents
    base_dir = Path("test_data")
    if not base_dir.exists():
        console.print(f"[bold red]❌ test_data directory not found![/bold red]")
        console.print(f"Expected location: {base_dir.absolute()}")
        console.print("\n💡 Run generate_test_data.py first to create test data")
        return
    
    console.print(f"\n📂 Loading documents from {base_dir.absolute()}...")
    raw_documents = load_all_documents(base_dir)
    
    if len(raw_documents) == 0:
        console.print("[bold red]❌ No documents found to index![/bold red]")
        return
    
    console.print(f"\n📊 Total documents loaded: [bold green]{len(raw_documents)}[/bold green]")
    
    # Convert to LangChain documents
    console.print(f"\n🔄 Converting to LangChain Document format...")
    langchain_documents = convert_to_langchain_documents(raw_documents)
    console.print(f"✅ Converted {len(langchain_documents)} documents")
    
    # Create vector store and index documents (Pure LangChain!)
    console.print(f"\n📤 Creating vector store and indexing documents...")
    console.print(f"   This may take a few moments...")
    
    try:
        # Using LangChain's from_documents - handles everything!
        vector_store = QdrantVectorStore.from_documents(
            documents=langchain_documents,
            embedding=embeddings,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME,
            force_recreate=True  # Recreate collection
        )
        console.print("✅ Indexing complete!")
    except Exception as e:
        console.print(f"[bold red]❌ Indexing failed: {e}[/bold red]")
        return
    
    # Verify indexing
    console.print(f"\n🔍 Verifying indexing...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    
    console.print("\n[bold green]✅ Indexing Complete![/bold green]\n")
    console.print(f"📊 Collection: {config.QDRANT_COLLECTION_NAME}")
    console.print(f"📈 Total vectors: {collection_info.points_count}")
    
    # Fix: Handle different vector config structures
    try:
        # Try new API format (dict with vector names)
        if isinstance(collection_info.config.params.vectors, dict):
            vector_size = list(collection_info.config.params.vectors.values())[0].size
        else:
            # Old API format (single VectorParams object)
            vector_size = collection_info.config.params.vectors.size
        console.print(f"🎯 Vector size: {vector_size}")
    except:
        console.print(f"🎯 Vector size: 384 (default)")
    
    # Summary by type
    jira_count = len([d for d in raw_documents if 'story_id' in d])
    conf_count = len([d for d in raw_documents if 'doc_id' in d])
    test_count = len([d for d in raw_documents if 'test_id' in d])
    
    console.print("\n📋 Document Breakdown:")
    console.print(f"  JIRA Stories:      {jira_count}")
    console.print(f"  Confluence Docs:   {conf_count}")
    console.print(f"  Zephyr Tests:      {test_count}")
    
    # Module breakdown
    modules = {}
    for doc in raw_documents:
        module = doc.get('module', 'Unknown')
        modules[module] = modules.get(module, 0) + 1
    
    console.print("\n📦 Modules Indexed:")
    for module, count in sorted(modules.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {module}: {count} documents")
    
    # Test a quick search
    console.print("\n🧪 Testing search functionality...")
    try:
        results = vector_store.similarity_search("payment processing", k=3)
        console.print(f"✅ Search working! Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            console.print(f"   {i}. {doc.metadata.get('document_id')} - {doc.metadata.get('module')}")
    except Exception as e:
        console.print(f"⚠️  Search test failed: {e}")
    
    console.print("\n✨ Ready for queries!\n")
    console.print("🚀 Next steps:")
    console.print("   1. python diagnose_embeddings.py  # Verify metadata structure")
    console.print("   2. python test_rag_tools.py       # Test RAG tools")
    console.print("   3. streamlit run app.py           # Run UI")
    
    console.print("\n💡 To swap vector stores in the future:")
    console.print("   Replace 'QdrantVectorStore' with:")
    console.print("   - PineconeVectorStore (Pinecone)")
    console.print("   - Chroma (Chroma)")
    console.print("   - FAISS (Facebook AI)")
    console.print("   - Weaviate (Weaviate)")
    console.print()


if __name__ == "__main__":
    index_documents()