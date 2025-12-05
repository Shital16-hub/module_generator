"""
Test Step 2: Planner and Tools nodes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.training_generator.agent import training_agent
from agents.training_generator.state import create_initial_state

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_agent():
    """Test the complete agent flow"""
    
    console.print("\n[bold blue]Testing Training Generator Agent[/bold blue]\n")
    
    # Create initial state
    initial_state = create_initial_state(
        user_request="Create training for Payment Module",
        module_name="Payment"
    )
    
    console.print("[yellow]Running agent...[/yellow]\n")
    
    # Run agent
    result = training_agent.invoke(initial_state)
    
    # Display results
    console.print("[green]✅ Agent completed![/green]\n")
    
    summary = f"""
Final State:
- Iterations: {result['iteration']}
- Stories: {len(result['stories'])}
- Docs: {len(result['documentation'])}
- Tests: {len(result['test_cases'])}
- Markdown generated: {'Yes' if result['markdown_output'] else 'No'}
    """
    
    console.print(Panel(summary, title="Results", border_style="green"))
    
    if result['markdown_output']:
        console.print("\n[bold cyan]Generated Training Module:[/bold cyan]\n")
        console.print(result['markdown_output'])


if __name__ == "__main__":
    test_agent()