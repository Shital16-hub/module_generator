"""
Main LangGraph agent for Training Generator
"""

from langgraph.graph import StateGraph, END

from .state import TrainingGeneratorState
from .nodes.planner import planner
from .nodes.tools import tools


def create_training_agent():
    """
    Create the training generator agent graph.
    
    Returns:
        Compiled LangGraph agent
    """
    
    # Create graph
    workflow = StateGraph(TrainingGeneratorState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("tools", tools)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    # Tools always routes back to planner for next decision
    workflow.add_edge("tools", "planner")
    
    # Compile
    app = workflow.compile()
    
    return app


# Create singleton instance
training_agent = create_training_agent()