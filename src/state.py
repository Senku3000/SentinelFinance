"""State management for LangGraph workflow"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class FinancialAdvisoryState(TypedDict):
    """State definition for the financial advisory graph"""
    
    # User input
    user_query: str
    user_id: str
    
    # Intent analysis
    intent: Optional[Dict[str, Any]]  # Parsed intent from Router
    needs_research: bool
    needs_calculation: bool
    needs_user_profile: bool
    
    # Research results
    hypotheses: List[Dict[str, Any]]  # Testable hypotheses
    research_results: List[Dict[str, Any]]  # Evidence from Vector DB
    market_data: Dict[str, Any]  # Real-time market data from Search Tool
    document_scores: List[Dict[str, float]]  # Relevance scores for documents
    
    # Calculations
    calculations: List[Dict[str, Any]]  # Results from Math Tool
    calculation_formulas: List[str]  # Formulas used for transparency
    
    # User profile
    user_profile: Optional[Dict[str, Any]]  # Income, expenses, goals, etc.
    
    # Synthesis
    intermediate_analysis: Optional[str]  # Analysis from Strategist
    confidence_scores: Dict[str, float]  # Confidence for each recommendation
    constraints_violated: List[str]  # Any constraint violations found
    
    # Final output
    final_recommendation: Optional[str]
    recommendation_breakdown: Optional[Dict[str, Any]]  # Detailed breakdown
    
    # Control flow
    iteration_count: int
    max_iterations: int
    current_node: str  # Track which node is executing
    next_node: Optional[str]  # Next node to route to
    
    # Messages for LLM
    messages: Annotated[List[Any], add_messages]
    
    # Audit trail
    tool_calls: List[Dict[str, Any]]  # Log of all tool calls
    errors: List[str]  # Any errors encountered


def create_initial_state(
    user_query: str,
    user_id: str = "default_user",
    max_iterations: int = 10
) -> FinancialAdvisoryState:
    """Create initial state for the graph"""
    return FinancialAdvisoryState(
        user_query=user_query,
        user_id=user_id,
        intent=None,
        needs_research=False,
        needs_calculation=False,
        needs_user_profile=False,
        hypotheses=[],
        research_results=[],
        market_data={},
        document_scores=[],
        calculations=[],
        calculation_formulas=[],
        user_profile=None,
        intermediate_analysis=None,
        confidence_scores={},
        constraints_violated=[],
        final_recommendation=None,
        recommendation_breakdown=None,
        iteration_count=0,
        max_iterations=max_iterations,
        current_node="router",
        next_node=None,
        messages=[],
        tool_calls=[],
        errors=[]
    )
