"""
Pydantic Models for Training Generator Agent

Updated for Azure OpenAI structured output compatibility.
All optional fields use Optional[] to meet Azure OpenAI requirements.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
from datetime import datetime


# ============================================================================
# PLANNER OUTPUT MODELS
# ============================================================================

class PlannerDecision(BaseModel):
    """
    Structured output from the planner node.
    
    Note: All optional fields use Optional[] for Azure OpenAI compatibility
    """
    
    action: Literal[
        'search_stories',
        'search_docs', 
        'find_relationships',
        'fetch_test_details',
        'generate_markdown',
        'complete'
    ] = Field(
        description="Next action to execute"
    )
    
    reasoning: str = Field(
        description="Detailed reasoning explaining why this action is needed based on current state"
    )
    
    query: Optional[str] = Field(
        default=None,
        description="Search query for Qdrant searches (only for search_* actions)"
    )
    
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Filters for search (e.g., {'source': 'JIRA', 'type': 'user_story'})"
    )
    
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description="Entity IDs for knowledge graph queries or batch retrieval"
    )
    
    relationship_type: Optional[str] = Field(
        default=None,
        description="Type of relationship to traverse in knowledge graph"
    )
    
    confidence: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in this decision (0.0 to 1.0)"
    )


class PlannerAnalysis(BaseModel):
    """Initial analysis before making a decision."""
    
    current_state_summary: str = Field(
        description="Summary of what data has been collected so far"
    )
    
    what_is_missing: List[str] = Field(
        description="List of data types still needed"
    )
    
    next_logical_step: str = Field(
        description="The most logical next step to take"
    )
    
    readiness_to_generate: float = Field(
        ge=0.0,
        le=1.0,
        description="How ready we are to generate the final output"
    )


# ============================================================================
# TRAINING STRUCTURE MODELS
# ============================================================================

class TrainingSection(BaseModel):
    """Structure for a single section in the training module."""
    
    section_id: int = Field(description="Sequential section number")
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown format")
    subsections: Optional[List[str]] = Field(default=None, description="List of subsection titles")


class QuizQuestion(BaseModel):
    """Structure for a quiz question."""
    
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of answer options")
    correct_answer: str = Field(description="The correct option")
    explanation: Optional[str] = Field(default=None, description="Explanation")


class TrainingModuleStructure(BaseModel):
    """Complete structure for the training module."""
    
    module_name: str = Field(description="Name of the module")
    overview: str = Field(description="High-level overview")
    learning_objectives: List[str] = Field(description="List of learning objectives")
    sections: List[TrainingSection] = Field(description="Main content sections")
    quiz_questions: Optional[List[QuizQuestion]] = Field(default=None, description="Knowledge check quiz")
    total_stories: int = Field(description="Number of stories covered")
    total_docs: int = Field(description="Number of documentation pages")
    total_test_cases: int = Field(description="Number of test cases included")
    generation_timestamp: str = Field(description="ISO format timestamp")


# ============================================================================
# SEARCH RESULT MODELS
# ============================================================================

class SearchResult(BaseModel):
    """Structure for a single search result from Qdrant."""
    
    id: str = Field(description="Unique identifier")
    content: str = Field(description="Text content")
    metadata: Dict = Field(description="Associated metadata")
    score: float = Field(description="Relevance score")


class SearchResponse(BaseModel):
    """Aggregated search response."""
    
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="List of search results")
    total_found: int = Field(description="Total number of results found")
    search_timestamp: Optional[str] = Field(default=None, description="When this search was performed")


# ============================================================================
# KNOWLEDGE GRAPH MODELS
# ============================================================================

class GraphRelationship(BaseModel):
    """Structure for a relationship in the knowledge graph."""
    
    source_id: str = Field(description="Source entity ID")
    target_id: str = Field(description="Target entity ID")
    relationship_type: str = Field(description="Type of relationship")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class GraphQueryResponse(BaseModel):
    """Response from a knowledge graph query."""
    
    relationships: List[GraphRelationship] = Field(description="List of discovered relationships")
    target_ids: List[str] = Field(description="All unique target IDs")
    query_timestamp: Optional[str] = Field(default=None, description="When query was performed")


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    """Structured error response."""
    
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Human-readable error message")
    details: Optional[Dict] = Field(default=None, description="Additional error details")
    timestamp: Optional[str] = Field(default=None, description="When error occurred")
    recoverable: bool = Field(default=True, description="Whether error is recoverable")


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_planner_decision(decision: PlannerDecision) -> tuple[bool, Optional[str]]:
    """Validate a planner decision for consistency."""
    
    # Check search actions have queries
    if decision.action in ['search_stories', 'search_docs']:
        if not decision.query:
            return False, f"Action '{decision.action}' requires a query"
    
    # Check graph actions have entity IDs
    if decision.action == 'find_relationships':
        if not decision.entity_ids or len(decision.entity_ids) == 0:
            return False, "Action 'find_relationships' requires entity_ids"
        if not decision.relationship_type:
            return False, "Action 'find_relationships' requires relationship_type"
    
    # Check batch retrieve has entity IDs
    if decision.action == 'fetch_test_details':
        if not decision.entity_ids or len(decision.entity_ids) == 0:
            return False, "Action 'fetch_test_details' requires entity_ids"
    
    return True, None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example planner decision
    decision = PlannerDecision(
        action="search_stories",
        reasoning="Need to find JIRA stories for the Payment module",
        query="payment module user stories epics",
        filters={"source": "JIRA", "type": "user_story"},
        confidence=0.95
    )
    
    print("Planner Decision:")
    print(decision.model_dump_json(indent=2))
    
    # Validate
    is_valid, error = validate_planner_decision(decision)
    print(f"\nValid: {is_valid}")
    if error:
        print(f"Error: {error}")