"""LangGraph workflow construction"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import FinancialAdvisoryState, create_initial_state
from .nodes import (
    router_node,
    researcher_node,
    analyst_node,
    strategist_node,
    clarifier_node,
    evidence_scorer_node,
    should_continue,
)
from .config import Config
from .ingestion.user_embedder import UserEmbedder


def create_financial_advisory_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow
    """
    # Create graph
    workflow = StateGraph(FinancialAdvisoryState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("evidence_scorer", evidence_scorer_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("strategist", strategist_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "clarifier": "clarifier",
            "researcher": "researcher",
            "analyst": "analyst",
            "strategist": "strategist",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "clarifier",
        should_continue,
        {
            "router": "router",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "evidence_scorer": "evidence_scorer",
            "researcher": "researcher",
            "analyst": "analyst",
            "strategist": "strategist",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "evidence_scorer",
        should_continue,
        {
            "clarifier": "clarifier",
            "analyst": "analyst",
            "strategist": "strategist",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "analyst": "analyst",
            "strategist": "strategist",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "strategist",
        should_continue,
        {
            "router": "router",  # Can loop back if more info needed
            "end": END
        }
    )
    
    # Compile graph with checkpointing
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_query(
    query: str,
    user_id: str = "default_user",
    config: dict = None,
    user_profile: dict = None,
    asked_clarifications: list = None
) -> dict:
    """
    Run a financial query through the graph
    
    Args:
        query: User's financial question
        user_id: User identifier
        config: Optional runtime config
        
    Returns:
        Final state with recommendation
    """
    # Validate config
    Config.validate()
    
    # Create graph
    app = create_financial_advisory_graph()
    
    # Create initial state
    initial_state = create_initial_state(
        query,
        user_id,
        Config.MAX_ITERATIONS,
        user_profile=user_profile,
        asked_clarifications=asked_clarifications
    )

    # Load user's document metadata so nodes know what's available
    try:
        user_embedder = UserEmbedder()
        initial_state["user_documents"] = user_embedder.list_user_documents(user_id)
    except Exception:
        initial_state["user_documents"] = []
    
    # Run graph
    config = config or {"configurable": {"thread_id": user_id}}
    
    # Use invoke for simpler execution (returns final state directly)
    try:
        final_state = app.invoke(initial_state, config)
        state_data = final_state
    except Exception as e:
        # Fallback: try streaming
        final_state = None
        for state in app.stream(initial_state, config):
            # State is a dict with node names as keys
            if isinstance(state, dict):
                # Get the last node's output
                node_names = list(state.keys())
                if node_names:
                    final_state = state[node_names[-1]]
                else:
                    final_state = list(state.values())[-1] if state.values() else {}
            else:
                final_state = state
        
        state_data = final_state if final_state else {}
    
    recommendation = state_data.get("final_recommendation")
    if not recommendation:
        questions = state_data.get("clarification_questions") or []
        if questions:
            recommendation = "I need a bit more information:\n- " + "\n- ".join(questions[:3])
        else:
            recommendation = "No recommendation generated"
    return {
        "query": query,
        "recommendation": recommendation,
        "confidence": state_data.get("confidence_scores", {}).get("overall", 0.5),
        "calculations": state_data.get("calculations", []),
        "research_results": state_data.get("research_results", []),
        "clarification_questions": state_data.get("clarification_questions", []),
        "asked_clarifications": state_data.get("asked_clarifications", []),
        "errors": state_data.get("errors", []),
        "tool_calls": state_data.get("tool_calls", [])
    }
