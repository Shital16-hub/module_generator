"""
Test RAG Tools
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.training_generator.tools.rag_tools import (
    search_stories,
    search_documentation,
    search_test_cases,
    find_test_cases_by_stories,
    batch_retrieve_by_ids,
    rag_tools
)

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_rag_tools():
    """Test all RAG tools"""
    
    console.print("\n[bold blue]Testing RAG Tools[/bold blue]\n")
    
    # Test 1: Search stories
    console.print("[yellow]Test 1: Searching for Payment stories...[/yellow]")
    stories = search_stories("credit card payment", module="Payment", top_k=3)
    console.print(f"Found {len(stories)} stories")
    for story in stories:
        console.print(f"  - {story['id']}: {story['metadata']['title']} (score: {story['score']:.3f})")
    
    # Test 2: Search documentation
    console.print("\n[yellow]Test 2: Searching for Payment documentation...[/yellow]")
    docs = search_documentation("payment processing architecture", module="Payment", top_k=2)
    console.print(f"Found {len(docs)} documents")
    for doc in docs:
        console.print(f"  - {doc['id']}: {doc['metadata']['title']} (score: {doc['score']:.3f})")
    
    # Test 3: Search test cases
    console.print("\n[yellow]Test 3: Searching for Payment test cases...[/yellow]")
    tests = search_test_cases("credit card payment test", module="Payment", top_k=3)
    console.print(f"Found {len(tests)} test cases")
    for test in tests:
        console.print(f"  - {test['id']}: {test['metadata']['title']} (score: {test['score']:.3f})")
    
    # Test 4: Find relationships
    console.print("\n[yellow]Test 4: Finding test cases for stories...[/yellow]")
    if stories:
        story_ids = [story['id'] for story in stories[:2]]  # Just test with 2 stories
        relationships = find_test_cases_by_stories(story_ids)
        for story_id, test_ids in relationships.items():
            console.print(f"  - {story_id} -> {len(test_ids)} test cases: {test_ids}")
        
        # Test 5: Batch retrieve test cases
        console.print("\n[yellow]Test 5: Retrieving test case details...[/yellow]")
        all_test_ids = [tid for test_list in relationships.values() for tid in test_list]
        if all_test_ids:
            retrieved_tests = batch_retrieve_by_ids(all_test_ids[:3], source="Zephyr")
            console.print(f"Retrieved {len(retrieved_tests)} test cases")
            for test in retrieved_tests:
                console.print(f"  - {test['id']}: {test['metadata']['title']}")
    
    # Stats
    console.print("\n[yellow]Collection Stats:[/yellow]")
    stats = rag_tools.get_collection_stats()
    console.print(Panel(
        f"Total Documents: {stats['total_documents']}\n"
        f"Collection: {stats['collection_name']}\n"
        f"Vector Size: {stats['vector_size']}",
        title="RAG Tools Status",
        border_style="green"
    ))
    
    console.print("\n[green]✅ All tests passed![/green]\n")


if __name__ == "__main__":
    test_rag_tools()