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
    rag_tools,
)

from rich.console import Console
from rich.panel import Panel

console = Console()


def _print_hits(label: str, items):
    console.print(f"Found {len(items)} {label}")
    for item in items:
        meta = item["metadata"]
        console.print(
            f"  - {item['id']}: {meta.get('title', '')} "
            f"[source={meta.get('source')}, module={meta.get('module')}] "
            f"(score: {item['score']:.3f})"
        )


def test_rag_tools():
    """Test all RAG tools"""

    console.print("\n[bold blue]Testing RAG Tools[/bold blue]\n")

    # ------------------------------------------------------------------
    # Test 1: Stories - use real modules (Inventory, Search)
    # ------------------------------------------------------------------
    console.print("[yellow]Test 1a: Searching stories (no module filter)...[/yellow]")
    stories_any = search_stories("inventory updates", module=None, top_k=5)
    _print_hits("stories (no module filter)", stories_any)

    console.print("\n[yellow]Test 1b: Searching stories for module='Inventory'...[/yellow]")
    stories_inventory = search_stories("inventory updates", module="Inventory", top_k=5)
    _print_hits("stories (module=Inventory)", stories_inventory)

    console.print("\n[yellow]Test 1c: Searching stories for module='Search'...[/yellow]")
    stories_search = search_stories("product search filters", module="Search", top_k=5)
    _print_hits("stories (module=Search)", stories_search)

    # ------------------------------------------------------------------
    # Test 2: Documentation
    # ------------------------------------------------------------------
    console.print("\n[yellow]Test 2a: Searching documentation (no module filter)...[/yellow]")
    docs_any = search_documentation("inventory architecture", module=None, top_k=5)
    _print_hits("documents (no module filter)", docs_any)

    console.print("\n[yellow]Test 2b: Searching documentation for module='Inventory'...[/yellow]")
    docs_inventory = search_documentation("inventory architecture", module="Inventory", top_k=5)
    _print_hits("documents (module=Inventory)", docs_inventory)

    console.print("\n[yellow]Test 2c: Searching documentation for module='Search'...[/yellow]")
    docs_search = search_documentation("advanced product search", module="Search", top_k=5)
    _print_hits("documents (module=Search)", docs_search)

    # ------------------------------------------------------------------
    # Test 3: Test cases
    # ------------------------------------------------------------------
    console.print("\n[yellow]Test 3a: Searching test cases (no module filter)...[/yellow]")
    tests_any = search_test_cases("inventory decrement test", module=None, top_k=5)
    _print_hits("test cases (no module filter)", tests_any)

    console.print("\n[yellow]Test 3b: Searching test cases for module='Inventory'...[/yellow]")
    tests_inventory = search_test_cases("inventory decrement test", module="Inventory", top_k=5)
    _print_hits("test cases (module=Inventory)", tests_inventory)

    console.print("\n[yellow]Test 3c: Searching test cases for module='Search'...[/yellow]")
    tests_search = search_test_cases("product search filters", module="Search", top_k=5)
    _print_hits("test cases (module=Search)", tests_search)

    # ------------------------------------------------------------------
    # Test 4: Relationships (only if we got some stories)
    # ------------------------------------------------------------------
    console.print("\n[yellow]Test 4: Finding test cases for stories...[/yellow]")
    base_stories = stories_any or stories_inventory or stories_search
    if base_stories:
        story_ids = [story["id"] for story in base_stories[:2]]
        relationships = find_test_cases_by_stories(story_ids)
        for story_id, test_ids in relationships.items():
            console.print(f"  - {story_id} -> {len(test_ids)} test cases: {test_ids}")

        # ------------------------------------------------------------------
        # Test 5: Batch retrieve test detail
        # ------------------------------------------------------------------
        console.print("\n[yellow]Test 5: Retrieving test case details...[/yellow]")
        all_test_ids = [tid for test_list in relationships.values() for tid in test_list]
        if all_test_ids:
            retrieved_tests = batch_retrieve_by_ids(all_test_ids[:3], source="Zephyr")
            console.print(f"Retrieved {len(retrieved_tests)} test cases")
            for test in retrieved_tests:
                console.print(f"  - {test['id']}: {test['metadata'].get('title', '')}")
        else:
            console.print("  - No linked test cases to retrieve")
    else:
        console.print("  - No stories found (skipping relationship tests)")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    console.print("\n[yellow]Collection Stats:[/yellow]")
    stats = rag_tools.get_collection_stats()
    console.print(
        Panel(
            f"Total Documents: {stats['total_documents']}\n"
            f"Collection: {stats['collection_name']}\n"
            f"Vector Size: {stats['vector_size']}",
            title="RAG Tools Status",
            border_style="green",
        )
    )

    console.print("\n[green]✅ Test run completed (see counts above).[/green]\n")


if __name__ == "__main__":
    test_rag_tools()
