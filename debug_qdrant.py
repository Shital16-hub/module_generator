"""
Debug script to inspect actual Qdrant payload structure
Run this to see exactly how your data is stored
"""

from qdrant_client import QdrantClient
from src.agents.training_generator.config import config

client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
    check_compatibility=False,
)

# Get a sample of all points to inspect payload structure
points, _ = client.scroll(
    collection_name=config.QDRANT_COLLECTION_NAME,
    limit=10,
    with_payload=True,
    with_vectors=False,
)

print(f"Total points inspected: {len(points)}")
print("=" * 80)

for i, point in enumerate(points):
    print(f"\n--- Point {i+1} (ID: {point.id}) ---")
    payload = point.payload or {}
    
    # Show top-level keys
    print(f"Top-level keys: {list(payload.keys())}")
    
    # Check if metadata is nested or flat
    if "metadata" in payload:
        meta = payload["metadata"]
        print(f"metadata keys: {list(meta.keys()) if isinstance(meta, dict) else meta}")
        print(f"  source: {meta.get('source', 'N/A')}")
        print(f"  type: {meta.get('type', 'N/A')}")
        print(f"  module: {meta.get('module', 'N/A')}")
    else:
        # Flat structure
        print(f"  source (flat): {payload.get('source', 'N/A')}")
        print(f"  type (flat): {payload.get('type', 'N/A')}")
        print(f"  module (flat): {payload.get('module', 'N/A')}")
    
    # Show page_content preview
    content = payload.get("page_content", "")
    if content:
        print(f"  page_content preview: {content[:100]}...")
    
    print("-" * 40)
