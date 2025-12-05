"""
Test Azure OpenAI configuration and connectivity
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Change to project root directory
os.chdir(project_root)

from agents.training_generator.config import config
from agents.training_generator.llm import get_llm, get_structured_llm
from agents.training_generator.models import PlannerDecision

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_config():
    """Test configuration validation"""
    console.print("\n[bold blue]Testing Configuration...[/bold blue]")
    
    is_valid, errors = config.validate()
    
    if is_valid:
        console.print("[green]✅ Configuration is valid![/green]")
        
        # Display config summary
        summary = f"""
Azure OpenAI Configuration:
  • Endpoint: {config.AZURE_OPENAI_ENDPOINT}
  • Deployment: {config.AZURE_OPENAI_DEPLOYMENT_NAME}
  • API Version: {config.OPENAI_API_VERSION}
  • API Key: {config.AZURE_OPENAI_API_KEY[:10]}... (hidden)
  
Qdrant Configuration:
  • URL: {config.QDRANT_URL}
  • Collection: {config.QDRANT_COLLECTION_NAME}
  
Agent Configuration:
  • Max Iterations: {config.MAX_ITERATIONS}
  • Search Top-K: {config.SEARCH_TOP_K}
  • Temperature: {config.LLM_TEMPERATURE}
        """
        console.print(Panel(summary, title="Configuration Summary", border_style="green"))
    else:
        console.print("[red]❌ Configuration validation failed![/red]")
        for error in errors:
            console.print(f"  [red]• {error}[/red]")
        return False
    
    return True


def test_llm_basic():
    """Test basic LLM invocation"""
    console.print("\n[bold blue]Testing Basic LLM...[/bold blue]")
    
    try:
        llm = get_llm()
        console.print("[yellow]Calling Azure OpenAI...[/yellow]")
        
        response = llm.invoke("Say 'Hello from Azure OpenAI!' and nothing else.")
        
        console.print("[green]✅ LLM invocation successful![/green]")
        console.print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ LLM test failed: {str(e)}[/red]")
        import traceback
        console.print("[red]" + traceback.format_exc() + "[/red]")
        return False


def test_structured_output():
    """Test structured output with Pydantic"""
    console.print("\n[bold blue]Testing Structured Output...[/bold blue]")
    
    try:
        llm = get_structured_llm(PlannerDecision, temperature=0.0)
        
        prompt = """
You are a training generator planner. Based on the current state, decide the next action.

Current State:
- Module: Payment
- Stories collected: 0
- Documentation collected: 0
- Test cases collected: 0

What should be the next action? Provide your response as structured JSON.
        """
        
        console.print("[yellow]Calling Azure OpenAI with structured output...[/yellow]")
        
        decision = llm.invoke(prompt)
        
        console.print("[green]✅ Structured output successful![/green]")
        console.print(f"Action: {decision.action}")
        console.print(f"Reasoning: {decision.reasoning}")
        console.print(f"Query: {decision.query}")
        console.print(f"Confidence: {decision.confidence}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Structured output test failed: {str(e)}[/red]")
        import traceback
        console.print("[red]" + traceback.format_exc() + "[/red]")
        return False


def main():
    """Run all tests"""
    console.print("\n[bold magenta]╔══════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Azure OpenAI Configuration Tests          ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════════╝[/bold magenta]\n")
    
    results = {
        "Configuration": test_config(),
        "Basic LLM": test_llm_basic(),
        "Structured Output": test_structured_output(),
    }
    
    # Summary
    console.print("\n[bold blue]═══════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]Test Summary[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════[/bold blue]\n")
    
    for test_name, passed in results.items():
        status = "[green]✅ PASSED[/green]" if passed else "[red]❌ FAILED[/red]"
        console.print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        console.print("\n[bold green]🎉 All tests passed! Ready to proceed.[/bold green]")
    else:
        console.print("\n[bold red]⚠️  Some tests failed. Please check configuration.[/bold red]")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)