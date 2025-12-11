"""
Planner node for Training Generator Agent

Uses LLM to decide next action based on current state.
"""

from langgraph.types import Command
from typing import Literal

from ..state import TrainingGeneratorState
from ..models import PlannerDecision
from ..llm import get_structured_llm
from ..prompts.planner_prompt import get_planner_prompt


def planner(state: TrainingGeneratorState) -> Command[Literal["tools", "__end__"]]:
    """
    Planner node - decides next action based on current state.
    
    Workflow:
    1. Check if markdown already generated → END
    2. Check max iterations → END with error
    3. Analyze state and decide next action
    4. Route to tools or end
    
    Args:
        state: Current training generator state
        
    Returns:
        Command with next node and state updates
    """
    
    # ========================================================================
    # Early Exit: Markdown Generated
    # ========================================================================
    
    if state['markdown_output']:
        print("  ✅ Training module already generated")
        return Command(
            goto="__end__",
            update={
                "current_action": "complete",
                "reasoning": "Training module successfully generated",
                "gathering_complete": True
            }
        )
    
    # ========================================================================
    # Early Exit: Max Iterations
    # ========================================================================
    
    if state['iteration'] >= state['max_iterations']:
        print(f"  ⚠️  Reached max iterations ({state['max_iterations']})")
        
        # Try to generate with what we have
        if state['stories'] or state['documentation']:
            print("  📝 Generating training module with available data...")
            return Command(
                goto="tools",
                update={
                    "current_action": "generate_markdown",
                    "reasoning": "Max iterations reached, generating with available data"
                }
            )
        else:
            return Command(
                goto="__end__",
                update={
                    "current_action": "complete",
                    "reasoning": "Max iterations reached without sufficient data",
                    "error_message": "Failed to collect sufficient training data"
                }
            )
    
    # ========================================================================
    # Rule-Based Decision Logic (More Deterministic)
    # ========================================================================
    
    # Step 1: Search stories (if none collected)
    if len(state['stories']) == 0:
        print(f"  🎯 Step 1: Searching for stories (iteration {state['iteration'] + 1})")
        return Command(
            goto="tools",
            update={
                "current_action": "search_stories",
                "reasoning": f"Need to collect user stories for {state['module_name']} module",
                "iteration": state["iteration"] + 1
            }
        )
    
    # Step 2: Search documentation (if none collected)
    if len(state['documentation']) == 0:
        print(f"  🎯 Step 2: Searching for documentation (iteration {state['iteration'] + 1})")
        return Command(
            goto="tools",
            update={
                "current_action": "search_docs",
                "reasoning": f"Need to collect documentation for {state['module_name']} module",
                "iteration": state["iteration"] + 1
            }
        )
    
    # Step 3: Find relationships (if not yet mapped)
    if not state['story_test_map'] and state['stories']:
        print(f"  🎯 Step 3: Finding story-test relationships (iteration {state['iteration'] + 1})")
        return Command(
            goto="tools",
            update={
                "current_action": "find_relationships",
                "reasoning": "Need to find test cases linked to collected stories",
                "iteration": state["iteration"] + 1
            }
        )
    
    # Step 4: Fetch test details (if relationships found but no tests)
    if state['story_test_map'] and len(state['test_cases']) == 0:
        total_test_ids = sum(len(tests) for tests in state['story_test_map'].values())
        print(f"  🎯 Step 4: Fetching {total_test_ids} test cases (iteration {state['iteration'] + 1})")
        return Command(
            goto="tools",
            update={
                "current_action": "fetch_test_details",
                "reasoning": f"Need to retrieve details for {total_test_ids} linked test cases",
                "iteration": state["iteration"] + 1
            }
        )
    
    # Step 5: Generate markdown (if sufficient data collected)
    min_artifacts = 3  # At least 3 artifacts to generate useful training
    if state['total_artifacts_found'] >= min_artifacts:
        print(f"  🎯 Step 5: Generating training module (iteration {state['iteration'] + 1})")
        print(f"      📊 Collected: {len(state['stories'])} stories, {len(state['documentation'])} docs, {len(state['test_cases'])} tests")
        return Command(
            goto="tools",
            update={
                "current_action": "generate_markdown",
                "reasoning": f"Sufficient data collected ({state['total_artifacts_found']} artifacts)",
                "iteration": state["iteration"] + 1
            }
        )
    
    # ========================================================================
    # Fallback: Use LLM Decision (for complex cases)
    # ========================================================================
    
    print(f"  🤔 Using LLM decision (iteration {state['iteration'] + 1})")
    
    # Get LLM with structured output
    llm = get_structured_llm(PlannerDecision, temperature=0.0)
    
    # Generate prompt
    prompt = get_planner_prompt(state)
    
    # Get decision from LLM
    decision = llm.invoke(prompt)
    
    print(f"      LLM Decision: {decision.action}")
    print(f"      Reasoning: {decision.reasoning[:80]}...")
    
    # Route based on decision
    if decision.action == "complete":
        next_node = "__end__"
    else:
        next_node = "tools"
    
    # Update state
    updates = {
        "current_action": decision.action,
        "reasoning": decision.reasoning,
        "iteration": state["iteration"] + 1
    }
    
    # Add query to queries_made list
    if decision.query:
        updates["queries_made"] = [decision.query]
    
    return Command(
        goto=next_node,
        update=updates
    )