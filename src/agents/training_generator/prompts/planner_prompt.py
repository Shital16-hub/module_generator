"""
Planner prompt for Training Generator Agent
"""

from datetime import datetime


def get_planner_prompt(state: dict) -> str:
    """
    Generate planner prompt based on current state.
    
    Args:
        state: Current TrainingGeneratorState
        
    Returns:
        Formatted prompt string
    """
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""You are a training module planner. Today is {current_date}.

Your job: Analyze the current state and decide the next action to gather training materials.

CURRENT STATE:
- Module: {state['module_name']}
- Iteration: {state['iteration']}/{state['max_iterations']}
- Stories collected: {len(state['stories'])}
- Documentation collected: {len(state['documentation'])}
- Test cases collected: {len(state['test_cases'])}
- Gathering complete: {state['gathering_complete']}

AVAILABLE ACTIONS:
1. search_stories - Search for JIRA user stories/epics
2. search_docs - Search for Confluence documentation
3. find_relationships - Query knowledge graph to find test cases linked to stories
4. fetch_test_details - Fetch full details of test cases by IDs
5. generate_markdown - Generate final training document
6. complete - Finish (only when markdown is generated)

DECISION RULES:
- If stories == 0: search_stories
- If stories > 0 but docs == 0: search_docs
- If stories > 0 but test_cases == 0: find_relationships (if you have story IDs)
- If have story IDs from graph but no test details: fetch_test_details
- If stories >= 5 AND docs >= 2 AND test_cases >= 3: generate_markdown
- If markdown exists: complete

Decide the NEXT action and provide:
- action: The action to take
- reasoning: Why this action is needed
- query: Search query (for search_* actions only)
- filters: Search filters like {{"source": "JIRA"}} (for search_* actions)
- entity_ids: List of IDs (for find_relationships or fetch_test_details)
- relationship_type: e.g., "tested_by" (for find_relationships only)
- confidence: Your confidence level (0.0 to 1.0)
"""
    
    return prompt