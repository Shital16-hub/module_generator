"""
Test RAG Tools with Broader Queries
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
from rich.table import Table

console = Console()


def _print_hits(label: str, items):
    if not items:
        console.print(f"  [dim]Found 0 {label}[/dim]")
        return
        
    console.print(f"  [cyan]Found {len(items)} {label}[/cyan]")
    for item in items:
        meta = item["metadata"]
        console.print(
            f"    ✓ [bold]{item['id']}[/bold]: {meta.get('title', 'N/A')[:60]}... "
            f"[dim](score: {item['score']:.3f})[/dim]"
        )


def test_rag_tools():
    """Test RAG with simpler, broader queries"""

    console.print("\n[bold magenta]═══════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]    Testing RAG Tools - Full Suite    [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════[/bold magenta]\n")

    all_results = []

    # ========================================================================
    # Test 1: JIRA Stories (Broad Queries)
    # ========================================================================
    
    console.print("[yellow]━━━ Test 1: JIRA User Stories ━━━[/yellow]\n")
    
    # 1a: Payment module
    console.print("  [bold]1a. Payment Module[/bold]")
    stories_payment = search_stories("payment", module="Payment", top_k=10)
    _print_hits("payment stories", stories_payment)
    all_results.extend(stories_payment)
    
    # 1b: Inventory module
    console.print("\n  [bold]1b. Inventory Module[/bold]")
    stories_inventory = search_stories("inventory", module="Inventory", top_k=10)
    _print_hits("inventory stories", stories_inventory)
    all_results.extend(stories_inventory)
    
    # 1c: Search module
    console.print("\n  [bold]1c. Search Module[/bold]")
    stories_search = search_stories("search", module="Search", top_k=10)
    _print_hits("search stories", stories_search)
    all_results.extend(stories_search)
    
    # 1d: Authentication
    console.print("\n  [bold]1d. Authentication Module[/bold]")
    stories_auth = search_stories("authentication", module="Authentication", top_k=10)
    _print_hits("auth stories", stories_auth)
    all_results.extend(stories_auth)
    
    # 1e: Order Management
    console.print("\n  [bold]1e. Order Management Module[/bold]")
    stories_order = search_stories("order", module="Order Management", top_k=10)
    _print_hits("order stories", stories_order)
    all_results.extend(stories_order)
    
    # 1f: All modules (no filter)
    console.print("\n  [bold]1f. All Modules (No Filter)[/bold]")
    stories_all = search_stories("user story", module=None, top_k=10)
    _print_hits("all stories", stories_all)

    # ========================================================================
    # Test 2: Confluence Documentation (Broad Queries)
    # ========================================================================
    
    console.print("\n\n[yellow]━━━ Test 2: Confluence Documentation ━━━[/yellow]\n")
    
    # 2a: Payment docs
    console.print("  [bold]2a. Payment Docs[/bold]")
    docs_payment = search_documentation("payment", module="Payment", top_k=10)
    _print_hits("payment docs", docs_payment)
    all_results.extend(docs_payment)
    
    # 2b: Inventory docs
    console.print("\n  [bold]2b. Inventory Docs[/bold]")
    docs_inventory = search_documentation("inventory", module="Inventory", top_k=10)
    _print_hits("inventory docs", docs_inventory)
    all_results.extend(docs_inventory)
    
    # 2c: Search docs
    console.print("\n  [bold]2c. Search Docs[/bold]")
    docs_search = search_documentation("search", module="Search", top_k=10)
    _print_hits("search docs", docs_search)
    all_results.extend(docs_search)
    
    # 2d: Authentication docs
    console.print("\n  [bold]2d. Authentication Docs[/bold]")
    docs_auth = search_documentation("authentication", module="Authentication", top_k=10)
    _print_hits("auth docs", docs_auth)
    all_results.extend(docs_auth)
    
    # 2e: All docs (no filter)
    console.print("\n  [bold]2e. All Docs (No Filter)[/bold]")
    docs_all = search_documentation("documentation guide", module=None, top_k=10)
    _print_hits("all docs", docs_all)

    # ========================================================================
    # Test 3: Zephyr Test Cases (Broad Queries)
    # ========================================================================
    
    console.print("\n\n[yellow]━━━ Test 3: Zephyr Test Cases ━━━[/yellow]\n")
    
    # 3a: Payment tests
    console.print("  [bold]3a. Payment Tests[/bold]")
    tests_payment = search_test_cases("payment", module="Payment", top_k=10)
    _print_hits("payment tests", tests_payment)
    all_results.extend(tests_payment)
    
    # 3b: Inventory tests
    console.print("\n  [bold]3b. Inventory Tests[/bold]")
    tests_inventory = search_test_cases("inventory", module="Inventory", top_k=10)
    _print_hits("inventory tests", tests_inventory)
    all_results.extend(tests_inventory)
    
    # 3c: Search tests
    console.print("\n  [bold]3c. Search Tests[/bold]")
    tests_search = search_test_cases("search", module="Search", top_k=10)
    _print_hits("search tests", tests_search)
    all_results.extend(tests_search)
    
    # 3d: Authentication tests
    console.print("\n  [bold]3d. Authentication Tests[/bold]")
    tests_auth = search_test_cases("authentication", module="Authentication", top_k=10)
    _print_hits("auth tests", tests_auth)
    all_results.extend(tests_auth)
    
    # 3e: All tests (no filter)
    console.print("\n  [bold]3e. All Tests (No Filter)[/bold]")
    tests_all = search_test_cases("test verify", module=None, top_k=10)
    _print_hits("all tests", tests_all)

    # ========================================================================
    # Test 4: Relationships
    # ========================================================================
    
    console.print("\n\n[yellow]━━━ Test 4: Story-Test Relationships ━━━[/yellow]\n")
    
    all_stories = stories_payment + stories_inventory + stories_search + stories_auth + stories_order
    if all_stories:
        # Take up to 5 stories
        story_ids = [s["id"] for s in all_stories[:5]]
        console.print(f"  Looking up test cases for stories: [cyan]{story_ids}[/cyan]\n")
        
        relationships = find_test_cases_by_stories(story_ids)
        
        table = Table(title="Story → Test Case Relationships", show_header=True, header_style="bold magenta")
        table.add_column("Story ID", style="cyan", width=15)
        table.add_column("Test Cases", style="green")
        
        for story_id, test_ids in relationships.items():
            if test_ids:
                table.add_row(story_id, f"{len(test_ids)} tests: {', '.join(test_ids[:3])}...")
            else:
                table.add_row(story_id, "[dim]No linked tests[/dim]")
        
        console.print(table)
    else:
        console.print("  [red]No stories found to test relationships[/red]")

    # ========================================================================
    # Test 5: Batch Retrieval
    # ========================================================================
    
    console.print("\n\n[yellow]━━━ Test 5: Batch Retrieval by ID ━━━[/yellow]\n")
    
    # Test with known IDs from various modules
    known_ids = [
        "TC-PAY-001", "TC-PAY-002", "TC-PAY-030",
        "TC-SRCH-020", "TC-INV-025", "TC-AUTH-020",
        "TC-ORD-050", "TC-NOTIF-030", "TC-USR-040"
    ]
    console.print(f"  Attempting to retrieve {len(known_ids)} test cases by ID...\n")
    
    retrieved = batch_retrieve_by_ids(known_ids, source="Zephyr")
    
    table = Table(title="Retrieved Test Cases", show_header=True, header_style="bold magenta")
    table.add_column("Test ID", style="cyan", width=15)
    table.add_column("Title", style="green")
    table.add_column("Module", style="yellow", width=15)
    
    for test in retrieved:
        table.add_row(
            test['id'],
            test['metadata'].get('title', 'N/A')[:50] + "...",
            test['metadata'].get('module', 'N/A')
        )
    
    console.print(table)
    console.print(f"\n  [green]✓ Successfully retrieved {len(retrieved)}/{len(known_ids)} test cases[/green]")

    # ========================================================================
    # Collection Stats
    # ========================================================================
    
    console.print("\n\n[yellow]━━━ Collection Statistics ━━━[/yellow]\n")
    stats = rag_tools.get_collection_stats()
    
    stats_panel = Panel(
        f"[bold cyan]Total Documents:[/bold cyan] {stats['total_documents']}\n"
        f"[bold cyan]Collection Name:[/bold cyan] {stats['collection_name']}\n"
        f"[bold cyan]Vector Dimension:[/bold cyan] {stats['vector_size']}\n"
        f"[bold cyan]Embedding Model:[/bold cyan] all-MiniLM-L6-v2",
        title="📊 Qdrant Collection Info",
        border_style="green",
        padding=(1, 2)
    )
    console.print(stats_panel)

    # ========================================================================
    # Final Summary
    # ========================================================================
    
    console.print("\n[yellow]━━━ Test Summary ━━━[/yellow]\n")
    
    # Count unique results
    unique_ids = set(item['id'] for item in all_results)
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green", justify="right")
    
    summary_table.add_row("JIRA Stories Found", str(len([r for r in all_results if r['metadata'].get('source') == 'JIRA'])))
    summary_table.add_row("Confluence Docs Found", str(len([r for r in all_results if r['metadata'].get('source') == 'Confluence'])))
    summary_table.add_row("Zephyr Tests Found", str(len([r for r in all_results if r['metadata'].get('source') == 'Zephyr'])))
    summary_table.add_row("Total Items Retrieved", str(len(all_results)))
    summary_table.add_row("Unique Documents", str(len(unique_ids)))
    summary_table.add_row("Documents in Qdrant", str(stats['total_documents']))
    
    console.print(summary_table)
    
    # Success criteria
    success_rate = (len(unique_ids) / stats['total_documents']) * 100 if stats['total_documents'] > 0 else 0
    
    if success_rate >= 60:
        console.print(f"\n[bold green]✅ RAG System Working Well! ({success_rate:.1f}% document retrieval)[/bold green]")
    elif success_rate >= 30:
        console.print(f"\n[bold yellow]⚠️  RAG System Partially Working ({success_rate:.1f}% document retrieval)[/bold yellow]")
        console.print("[yellow]   Consider lowering MIN_RELEVANCE_SCORE in config.py[/yellow]")
    else:
        console.print(f"\n[bold red]❌ RAG System Needs Tuning ({success_rate:.1f}% document retrieval)[/bold red]")
        console.print("[red]   Check indexing and query relevance[/red]")
    
    console.print("\n[dim]Tip: Lower MIN_RELEVANCE_SCORE in config.py if you want more results[/dim]\n")


if __name__ == "__main__":
    test_rag_tools()