"""
State Schema for Training Generator Agent

This module defines the state structure for the training module generator agent.
Uses TypedDict for type safety and Annotated for custom reducers.

Based on LangGraph v0.2+ state management patterns.
"""

from typing import TypedDict, List, Dict, Annotated, Optional
from datetime import datetime
import operator


# ============================================================================
# STATE SCHEMA
# ============================================================================

class TrainingGeneratorState(TypedDict):
    """
    State schema for the Training Generator Agent.
    
    This state is passed between the planner and tools nodes, maintaining
    all information gathered during the training module generation process.
    
    State Flow:
    1. User provides request → module_name extracted
    2. Planner decides actions → current_action set
    3. Tools execute → data collected (stories, docs, test_cases)
    4. Repeat until gathering_complete = True
    5. Generate markdown → markdown_output populated
    """
    
    # ========================================================================
    # INPUT FIELDS
    # ========================================================================
    
    user_request: str
    """Original user request (e.g., 'Create training for Payment Module')"""
    
    module_name: str
    """Extracted module name (e.g., 'Payment', 'Login', 'Checkout')"""
    
    # ========================================================================
    # CONTROL FLOW FIELDS
    # ========================================================================
    
    iteration: int
    """Current iteration count (starts at 0)"""
    
    max_iterations: int
    """Maximum allowed iterations (default: 8)"""
    
    current_action: str
    """
    Current action being executed. Possible values:
    - 'initialize' : Initial state
    - 'search_stories' : Search for JIRA stories
    - 'search_docs' : Search for Confluence documentation
    - 'find_relationships' : Query knowledge graph for relationships
    - 'fetch_test_details' : Batch retrieve test case details
    - 'generate_markdown' : Generate final markdown output
    - 'complete' : Process complete
    """
    
    reasoning: str
    """Planner's reasoning for the current action"""
    
    gathering_complete: bool
    """Flag indicating all data gathering is complete"""
    
    # ========================================================================
    # COLLECTED DATA (Using operator.add for list accumulation)
    # ========================================================================
    
    stories: Annotated[List[Dict], operator.add]
    """
    Collected JIRA stories/epics.
    Each story is a dict with:
    {
        'id': str,
        'content': str,
        'metadata': {
            'source': 'JIRA',
            'type': 'user_story' | 'epic',
            'title': str,
            'epic': str,
            ...
        },
        'score': float  # relevance score from vector search
    }
    """
    
    documentation: Annotated[List[Dict], operator.add]
    """
    Collected Confluence documentation.
    Each doc is a dict with:
    {
        'id': str,
        'content': str,
        'metadata': {
            'source': 'Confluence',
            'type': 'documentation',
            'title': str,
            'page_id': str,
            ...
        },
        'score': float
    }
    """
    
    test_cases: Annotated[List[Dict], operator.add]
    """
    Collected Zephyr test cases.
    Each test case is a dict with:
    {
        'id': str,
        'content': str,
        'metadata': {
            'source': 'Zephyr',
            'type': 'test_case',
            'title': str,
            'steps': List[str],
            'expected_result': str,
            ...
        },
        'score': float
    }
    """
    
    # ========================================================================
    # RELATIONSHIP MAPPING (From Knowledge Graph)
    # ========================================================================
    
    story_test_map: Dict[str, List[str]]
    """
    Mapping of story IDs to test case IDs.
    Example: {'PAY-001': ['TC-045', 'TC-046'], 'PAY-002': ['TC-047']}
    """
    
    story_doc_map: Dict[str, List[str]]
    """
    Mapping of story IDs to documentation IDs.
    Example: {'PAY-001': ['CONF-123'], 'PAY-002': ['CONF-124']}
    """
    
    # ========================================================================
    # SEARCH TRACKING
    # ========================================================================
    
    queries_made: Annotated[List[str], operator.add]
    """List of all search queries made (for debugging and audit)"""
    
    # ========================================================================
    # OUTPUT FIELDS
    # ========================================================================
    
    markdown_output: str
    """Final generated training module in markdown format"""
    
    # ========================================================================
    # METADATA
    # ========================================================================
    
    generation_timestamp: str
    """ISO format timestamp of when generation started"""
    
    total_artifacts_found: int
    """Total number of artifacts collected (stories + docs + tests)"""
    
    error_message: Optional[str]
    """Error message if generation fails"""


# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def create_initial_state(user_request: str, module_name: str) -> TrainingGeneratorState:
    """
    Create initial state for training generation.
    
    Args:
        user_request: User's original request
        module_name: Extracted module name
        
    Returns:
        Initialized TrainingGeneratorState
    """
    return TrainingGeneratorState(
        # Input
        user_request=user_request,
        module_name=module_name,
        
        # Control flow
        iteration=0,
        max_iterations=8,
        current_action="initialize",
        reasoning="Starting training generation process",
        gathering_complete=False,
        
        # Collected data (empty lists)
        stories=[],
        documentation=[],
        test_cases=[],
        
        # Relationships (empty dicts)
        story_test_map={},
        story_doc_map={},
        
        # Search tracking
        queries_made=[],
        
        # Output
        markdown_output="",
        
        # Metadata
        generation_timestamp=datetime.now().isoformat(),
        total_artifacts_found=0,
        error_message=None
    )


# ============================================================================
# STATE HELPER FUNCTIONS
# ============================================================================

def get_state_summary(state: TrainingGeneratorState) -> str:
    """
    Get a human-readable summary of the current state.
    
    Args:
        state: Current training generator state
        
    Returns:
        Formatted summary string
    """
    return f"""
Training Generation State Summary
{'='*50}
Module: {state['module_name']}
Iteration: {state['iteration']}/{state['max_iterations']}
Current Action: {state['current_action']}
Gathering Complete: {state['gathering_complete']}

Data Collected:
- Stories: {len(state['stories'])}
- Documentation: {len(state['documentation'])}
- Test Cases: {len(state['test_cases'])}
- Total Artifacts: {state['total_artifacts_found']}

Relationships:
- Story-Test Mappings: {len(state['story_test_map'])}
- Story-Doc Mappings: {len(state['story_doc_map'])}

Output Status: {'Generated' if state['markdown_output'] else 'Pending'}
Queries Made: {len(state['queries_made'])}
{'='*50}
    """.strip()


def validate_state(state: TrainingGeneratorState) -> tuple[bool, Optional[str]]:
    """
    Validate state integrity.
    
    Args:
        state: State to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check iteration bounds
    if state['iteration'] > state['max_iterations']:
        return False, f"Iteration count {state['iteration']} exceeds max {state['max_iterations']}"
    
    # Check module name
    if not state['module_name']:
        return False, "Module name is empty"
    
    # Check action validity
    valid_actions = {
        'initialize', 'search_stories', 'search_docs', 
        'find_relationships', 'fetch_test_details', 
        'generate_markdown', 'complete'
    }
    if state['current_action'] not in valid_actions:
        return False, f"Invalid action: {state['current_action']}"
    
    # Check data consistency
    if state['gathering_complete'] and not state['stories']:
        return False, "Gathering complete but no stories collected"
    
    return True, None


def should_continue_gathering(state: TrainingGeneratorState) -> bool:
    """
    Determine if we should continue gathering data.
    
    Args:
        state: Current state
        
    Returns:
        True if should continue, False otherwise
    """
    # Stop if already complete
    if state['gathering_complete']:
        return False
    
    # Stop if max iterations reached
    if state['iteration'] >= state['max_iterations']:
        return False
    
    # Stop if markdown already generated
    if state['markdown_output']:
        return False
    
    return True


# ============================================================================
# EXAMPLE USAGE (for testing)
# ============================================================================

if __name__ == "__main__":
    # Create initial state
    state = create_initial_state(
        user_request="Create training module for Payment Processing",
        module_name="Payment"
    )
    
    # Print summary
    print(get_state_summary(state))
    
    # Validate
    is_valid, error = validate_state(state)
    print(f"\nState Valid: {is_valid}")
    if error:
        print(f"Error: {error}")
    
    # Test state update (simulating data collection)
    state['stories'].append({
        'id': 'PAY-001',
        'content': 'As a user, I want to pay with credit card',
        'metadata': {'source': 'JIRA', 'type': 'user_story', 'title': 'CC Payment'},
        'score': 0.92
    })
    state['iteration'] = 1
    state['total_artifacts_found'] = 1
    
    print("\n" + "="*50)
    print("After Update:")
    print(get_state_summary(state))