"""
Tools node for Training Generator Agent
"""

from ..state import TrainingGeneratorState


def tools(state: TrainingGeneratorState) -> dict:
    """
    Tools node - executes the action decided by planner.
    
    Args:
        state: Current training generator state
        
    Returns:
        State updates
    """
    
    action = state['current_action']
    
    # For now, return mock data to test the flow
    # We'll implement real tools in Step 3
    
    if action == "search_stories":
        # Mock: Return fake stories
        mock_stories = [
            {
                'id': 'PAY-001',
                'content': 'As a user, I want to pay with credit card',
                'metadata': {'source': 'JIRA', 'type': 'user_story'},
                'score': 0.95
            },
            {
                'id': 'PAY-002',
                'content': 'As a user, I want to pay with PayPal',
                'metadata': {'source': 'JIRA', 'type': 'user_story'},
                'score': 0.90
            }
        ]
        return {
            "stories": mock_stories,
            "total_artifacts_found": state['total_artifacts_found'] + 2
        }
    
    elif action == "search_docs":
        # Mock: Return fake docs
        mock_docs = [
            {
                'id': 'CONF-123',
                'content': 'Payment Processing Guide: The payment system handles...',
                'metadata': {'source': 'Confluence', 'type': 'documentation'},
                'score': 0.88
            }
        ]
        return {
            "documentation": mock_docs,
            "total_artifacts_found": state['total_artifacts_found'] + 1
        }
    
    elif action == "find_relationships":
        # Mock: Return fake relationships
        return {
            "story_test_map": {
                "PAY-001": ["TC-001", "TC-002"],
                "PAY-002": ["TC-003"]
            }
        }
    
    elif action == "fetch_test_details":
        # Mock: Return fake test cases
        mock_tests = [
            {
                'id': 'TC-001',
                'content': 'Verify successful credit card payment',
                'metadata': {
                    'source': 'Zephyr',
                    'type': 'test_case',
                    'steps': ['Open checkout', 'Enter card', 'Submit', 'Verify']
                },
                'score': 1.0
            }
        ]
        return {
            "test_cases": mock_tests,
            "total_artifacts_found": state['total_artifacts_found'] + 1
        }
    
    elif action == "generate_markdown":
        # Mock: Return simple markdown
        markdown = f"""# {state['module_name']} - Training Module

## Overview
Training materials for {state['module_name']}.

## Stories Covered
Total: {len(state['stories'])}

## Documentation
Total: {len(state['documentation'])}

## Test Cases
Total: {len(state['test_cases'])}
"""
        return {
            "markdown_output": markdown,
            "gathering_complete": True
        }
    
    # Default: no update
    return {}