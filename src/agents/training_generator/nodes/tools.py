"""
Tools node for Training Generator Agent

Fully LLM-driven - No hardcoded module selection logic.
"""

from ..state import TrainingGeneratorState
from ..tools.rag_tools import (
    search_stories,
    search_documentation,
    search_test_cases,
    find_test_cases_by_stories,
    batch_retrieve_by_ids
)
from ..llm import get_llm
import json
import re


def llm_filter_results(
    user_query: str,
    search_results: list,
    result_type: str,
    max_results: int = 10
) -> tuple[list, str]:
    """
    Use LLM to intelligently filter search results.
    
    Args:
        user_query: What the user asked for (e.g., "Reviews", "Payment Processing")
        search_results: Raw results from semantic search
        result_type: "stories", "documentation", or "tests"
        max_results: Maximum results to return
        
    Returns:
        (filtered_results, detected_module)
    """
    
    if not search_results:
        return [], None
    
    # Prepare results summary for LLM
    results_for_llm = []
    for idx, item in enumerate(search_results[:30]):  # Check top 30
        results_for_llm.append({
            "index": idx,
            "id": item.get('id', 'unknown'),
            "module": item.get('module', 'Unknown'),
            "title": item.get('metadata', {}).get('title', 'N/A')[:150],
            "description": item.get('metadata', {}).get('description', '')[:200],
            "score": round(item.get('score', 1.0), 3)
        })
    
    # Create intelligent prompt
    prompt = f"""You are an intelligent document filter for a training generation system.

**User's Request:** "{user_query}"

**Your Task:** 
Analyze the search results below and identify which {result_type} are ACTUALLY relevant to "{user_query}".

**Search Results:**
```json
{json.dumps(results_for_llm, indent=2)}
```

**Instructions:**
1. Look at module names, titles, and descriptions
2. Identify which results genuinely relate to "{user_query}"
3. Ignore results from unrelated modules
4. If "{user_query}" mentions a specific feature (e.g., "Payment Processing"), match by functionality, not just exact module name
5. Determine the PRIMARY module these results belong to

**Response Format (JSON only):**
{{
  "relevant_indices": [0, 2, 5, ...],
  "detected_module": "The correct module name",
  "reasoning": "Brief explanation of why you chose these results"
}}

Respond with ONLY valid JSON, no other text."""

    # Get LLM decision
    llm = get_llm(temperature=0.0)
    response = llm.invoke(prompt)
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*"relevant_indices"[^{}]*\}', response.content, re.DOTALL)
        if not json_match:
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        
        if json_match:
            decision = json.loads(json_match.group())
            
            relevant_indices = decision.get('relevant_indices', [])
            detected_module = decision.get('detected_module', None)
            reasoning = decision.get('reasoning', '')
            
            print(f"  🤖 LLM Analysis: {reasoning[:100]}...")
            print(f"  🎯 Detected Module: '{detected_module}'")
            print(f"  ✅ Selected {len(relevant_indices)} relevant {result_type}")
            
            # Filter results based on LLM decision
            filtered = []
            for idx in relevant_indices:
                if idx < len(search_results):
                    filtered.append(search_results[idx])
                if len(filtered) >= max_results:
                    break
            
            return filtered, detected_module
    
    except Exception as e:
        print(f"  ⚠️  LLM filtering failed: {e}")
    
    # Fallback: Use top results by score
    print(f"  ⚠️  Falling back to score-based selection")
    sorted_results = sorted(search_results, key=lambda x: x.get('score', 1.0))
    detected_module = sorted_results[0].get('module') if sorted_results else None
    return sorted_results[:max_results], detected_module


def tools(state: TrainingGeneratorState) -> dict:
    """
    Tools node - fully LLM-driven execution.
    
    No hardcoded logic. LLM decides what's relevant.
    """
    
    action = state['current_action']
    user_module = state['module_name']
    
    # ========================================================================
    # ACTION: Search Stories
    # ========================================================================
    
    if action == "search_stories":
        print(f"  🔍 Searching stories for: '{user_module}'")
        
        # Build focused query (just the module/feature name)
        query = user_module
        
        # Get broad semantic search results
        all_stories = search_stories(query, module=None, top_k=30)
        
        print(f"  📊 Semantic search returned {len(all_stories)} candidate stories")
        
        # Let LLM filter intelligently
        filtered_stories, detected_module = llm_filter_results(
            user_query=user_module,
            search_results=all_stories,
            result_type="stories",
            max_results=10
        )
        
        if filtered_stories:
            # Use LLM-detected module
            actual_module = detected_module or user_module
            
            updates = {
                "stories": filtered_stories,
                "total_artifacts_found": state['total_artifacts_found'] + len(filtered_stories)
            }
            
            # Update module name if LLM detected a better one
            if detected_module and detected_module != user_module:
                print(f"  ℹ️  Module refined: '{user_module}' → '{detected_module}'")
                updates["module_name"] = detected_module
            
            return updates
        else:
            print(f"  ⚠️  No relevant stories found")
            return {
                "stories": [],
                "total_artifacts_found": state['total_artifacts_found']
            }
    
    # ========================================================================
    # ACTION: Search Documentation
    # ========================================================================
    
    elif action == "search_docs":
        actual_module = state['module_name']
        
        print(f"  📚 Searching documentation for: '{actual_module}'")
        
        # Focused query
        query = f"{actual_module} documentation guide"
        
        # Broad search
        all_docs = search_documentation(query, module=None, top_k=30)
        
        print(f"  📊 Semantic search returned {len(all_docs)} candidate docs")
        
        # LLM filters
        filtered_docs, _ = llm_filter_results(
            user_query=actual_module,
            search_results=all_docs,
            result_type="documentation",
            max_results=10
        )
        
        return {
            "documentation": filtered_docs,
            "total_artifacts_found": state['total_artifacts_found'] + len(filtered_docs)
        }
    
    # ========================================================================
    # ACTION: Search Test Cases
    # ========================================================================
    
    elif action == "search_test_cases":
        actual_module = state['module_name']
        
        print(f"  🧪 Searching test cases for: '{actual_module}'")
        
        # Focused query
        query = f"{actual_module} test verify"
        
        # Broad search
        all_tests = search_test_cases(query, module=None, top_k=30)
        
        print(f"  📊 Semantic search returned {len(all_tests)} candidate tests")
        
        # LLM filters
        filtered_tests, _ = llm_filter_results(
            user_query=actual_module,
            search_results=all_tests,
            result_type="tests",
            max_results=10
        )
        
        return {
            "test_cases": filtered_tests,
            "total_artifacts_found": state['total_artifacts_found'] + len(filtered_tests),
            "gathering_complete": True
        }
    
    # ========================================================================
    # ACTION: Find Relationships
    # ========================================================================
    
    elif action == "find_relationships":
        story_ids = [s['id'] for s in state['stories']]
        
        if not story_ids:
            print("  ⚠️  No stories available")
            return {"story_test_map": {}}
        
        story_test_map = find_test_cases_by_stories(story_ids)
        
        total_tests = sum(len(tests) for tests in story_test_map.values())
        print(f"  🔗 Found relationships: {len(story_test_map)} stories → {total_tests} test cases")
        
        return {"story_test_map": story_test_map}
    
    # ========================================================================
    # ACTION: Fetch Test Details
    # ========================================================================
    
    elif action == "fetch_test_details":
        test_ids = []
        for test_list in state['story_test_map'].values():
            test_ids.extend(test_list)
        
        test_ids = list(set(test_ids))
        
        if not test_ids:
            # Fallback to semantic search
            return tools({**state, 'current_action': 'search_test_cases'})
        
        print(f"  📋 Fetching {len(test_ids)} test cases by ID...")
        
        test_cases = batch_retrieve_by_ids(test_ids, source="Zephyr")
        
        print(f"  ✅ Retrieved {len(test_cases)}/{len(test_ids)} test cases")
        
        return {
            "test_cases": test_cases,
            "total_artifacts_found": state['total_artifacts_found'] + len(test_cases)
        }
    
    # ========================================================================
    # ACTION: Generate Markdown
    # ========================================================================
    
    elif action == "generate_markdown":
        markdown = generate_training_markdown(state)
        
        print(f"  📝 Generated markdown: {len(markdown)} characters")
        
        return {
            "markdown_output": markdown,
            "gathering_complete": True
        }
    
    return {}


def generate_training_markdown(state: TrainingGeneratorState) -> str:
    """Generate training module (same as before)"""
    
    module = state['module_name']
    stories = state['stories']
    docs = state['documentation']
    tests = state['test_cases']
    timestamp = state['generation_timestamp']
    
    markdown = f"""# {module} Module - Training Package

**Generated:** {timestamp}  
**Total Artifacts:** {state['total_artifacts_found']}

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [User Stories](#user-stories) ({len(stories)} items)
3. [Documentation](#documentation) ({len(docs)} items)
4. [Test Cases](#test-cases) ({len(tests)} items)

---

## Overview

This training module covers the **{module}** module.

### Learning Objectives

- Understand business requirements for {module}
- Learn technical implementation details
- Review test scenarios and acceptance criteria

---

## User Stories

Total: **{len(stories)}**

"""
    
    if stories:
        for idx, story in enumerate(stories, 1):
            meta = story.get('metadata', {})
            markdown += f"""### {idx}. {story.get('id')}: {meta.get('title', 'N/A')}

**Priority:** {meta.get('priority')} | **Status:** {meta.get('status')} | **Points:** {meta.get('story_points')}

**Description:** {meta.get('description', '')[:300]}...

**Acceptance Criteria:**
"""
            for criterion in meta.get('acceptance_criteria', [])[:5]:
                markdown += f"- {criterion}\n"
            markdown += "\n"
    
    markdown += f"\n---\n\n## Documentation\n\nTotal: **{len(docs)}**\n\n"
    
    if docs:
        for idx, doc in enumerate(docs, 1):
            meta = doc.get('metadata', {})
            markdown += f"""### {idx}. {doc.get('id')}: {meta.get('title')}

{meta.get('content', '')[:400]}...

"""
    
    markdown += f"\n---\n\n## Test Cases\n\nTotal: **{len(tests)}**\n\n"
    
    if tests:
        for idx, test in enumerate(tests, 1):
            meta = test.get('metadata', {})
            markdown += f"""### {idx}. {test.get('id')}: {meta.get('title')}

**Objective:** {meta.get('objective')}  
**Priority:** {meta.get('priority')}

"""
    
    markdown += "\n---\n\n*End of Training Module*\n"
    
    return markdown