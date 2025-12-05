"""
Planner node for Training Generator Agent
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
    
    Args:
        state: Current training generator state
        
    Returns:
        Command with next node and state updates
    """
    
    # Check if we're done
    if state['markdown_output']:
        return Command(
            goto="__end__",
            update={
                "current_action": "complete",
                "reasoning": "Training module generated successfully"
            }
        )
    
    # Check max iterations
    if state['iteration'] >= state['max_iterations']:
        return Command(
            goto="__end__",
            update={
                "current_action": "complete",
                "reasoning": "Maximum iterations reached",
                "error_message": "Reached max iterations without completing"
            }
        )
    
    # Get LLM with structured output
    llm = get_structured_llm(PlannerDecision, temperature=0.0)
    
    # Generate prompt
    prompt = get_planner_prompt(state)
    
    # Get decision from LLM
    decision = llm.invoke(prompt)
    
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