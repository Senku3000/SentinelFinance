"""Core nodes for LangGraph workflow"""

import json
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .state import FinancialAdvisoryState
from .config import Config
from .tools import MathTool, SearchTool, VectorDBTool, UserVaultTool


# Initialize LLM
llm = ChatGroq(
    model=Config.GROQ_MODEL,
    groq_api_key=Config.GROQ_API_KEY,
    temperature=0.3
)


def router_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Router node: Analyzes user intent and determines routing
    """
    state["current_node"] = "router"
    state["iteration_count"] += 1
    
    # Check max iterations
    if state["iteration_count"] >= state["max_iterations"]:
        state["next_node"] = "end"
        state["errors"].append("Maximum iterations reached")
        return state
    
    user_query = state["user_query"]
    
    # Intent analysis prompt
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial advisory router. Analyze the user's query and determine:
1. Does this query need research from knowledge base? (tax laws, regulations, principles)
2. Does this query need calculations? (EMI, SIP, tax calculations, ROI)
3. Does this query need user profile data? (income, expenses, goals)

Respond in JSON format:
{{
    "needs_research": true/false,
    "needs_calculation": true/false,
    "needs_user_profile": true/false,
    "reasoning": "brief explanation",
    "query_type": "tax_planning/investment/loan/retirement/etc"
}}"""),
        ("human", "User query: {query}")
    ])
    
    try:
        # Get intent analysis
        messages = intent_prompt.format_messages(query=user_query)
        response = llm.invoke(messages)
        
        # Parse response (simple extraction - could be improved)
        response_text = response.content
        
        # Extract JSON from response
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            intent_data = json.loads(json_match.group())
        else:
            # Fallback: infer from response text
            intent_data = {
                "needs_research": "research" in response_text.lower() or "knowledge" in response_text.lower(),
                "needs_calculation": "calculation" in response_text.lower() or "calculate" in response_text.lower(),
                "needs_user_profile": "profile" in response_text.lower() or "income" in response_text.lower(),
                "reasoning": response_text,
                "query_type": "general"
            }
        
        # Update state
        state["intent"] = intent_data
        state["needs_research"] = intent_data.get("needs_research", False)
        state["needs_calculation"] = intent_data.get("needs_calculation", False)
        state["needs_user_profile"] = intent_data.get("needs_user_profile", False)
        
        # Determine next node
        if state["needs_research"]:
            state["next_node"] = "researcher"
        elif state["needs_calculation"]:
            state["next_node"] = "analyst"
        elif state["needs_user_profile"] and not state.get("user_profile"):
            # Need to fetch user profile first
            state["next_node"] = "researcher"  # Will fetch profile in researcher
        else:
            state["next_node"] = "strategist"
        
        # Log tool call
        state["tool_calls"].append({
            "node": "router",
            "tool": "intent_analysis",
            "input": user_query,
            "output": intent_data
        })
        
    except Exception as e:
        state["errors"].append(f"Router error: {str(e)}")
        # Default routing: go to researcher
        state["next_node"] = "researcher"
        state["needs_research"] = True
    
    return state


def researcher_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Researcher node: Generates hypotheses and retrieves evidence
    """
    state["current_node"] = "researcher"
    
    user_query = state["user_query"]
    
    # Initialize tools
    vector_db_tool = VectorDBTool()
    search_tool = SearchTool()
    user_vault_tool = UserVaultTool()
    
    # Step 1: Generate hypotheses
    hypothesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial researcher. Generate 2-3 testable hypotheses about the user's financial query.
Each hypothesis should be specific and testable with data from knowledge base or market data.

Format as JSON array:
[
    {{"hypothesis": "specific testable statement", "needs_vector_db": true/false, "needs_market_data": true/false}},
    ...
]"""),
        ("human", "User query: {query}")
    ])
    
    try:
        # Generate hypotheses
        messages = hypothesis_prompt.format_messages(query=user_query)
        response = llm.invoke(messages)
        
        # Parse hypotheses (simplified)
        json_match = re.search(r'\[[^\]]*\]', response.content, re.DOTALL)
        if json_match:
            hypotheses = json.loads(json_match.group())
        else:
            # Fallback: create simple hypothesis
            hypotheses = [{
                "hypothesis": f"User needs information about: {user_query}",
                "needs_vector_db": True,
                "needs_market_data": False
            }]
        
        state["hypotheses"] = hypotheses
        
        # Step 2: Fetch user profile if needed
        if state["needs_user_profile"] and not state.get("user_profile"):
            profile_result = user_vault_tool._run(
                "get_profile",
                state["user_id"]
            )
            if profile_result.get("success"):
                state["user_profile"] = profile_result["profile"]
                state["tool_calls"].append({
                    "node": "researcher",
                    "tool": "user_vault_tool",
                    "operation": "get_profile",
                    "output": profile_result
                })
        
        # Step 3: Retrieve from vector DB
        research_results = []
        for hypothesis in hypotheses:
            if hypothesis.get("needs_vector_db", True):
                # Search vector DB
                search_query = hypothesis.get("hypothesis", user_query)
                db_result = vector_db_tool._run(search_query, k=3)
                
                if db_result.get("success"):
                    research_results.extend(db_result["results"])
                    state["document_scores"].extend([
                        {"score": score, "source": "vector_db"}
                        for score in db_result.get("scores", [])
                    ])
                    
                    state["tool_calls"].append({
                        "node": "researcher",
                        "tool": "vector_db_tool",
                        "query": search_query,
                        "results_count": len(db_result["results"])
                    })
        
        state["research_results"] = research_results
        
        # Step 4: Fetch market data if needed
        market_data = {}
        for hypothesis in hypotheses:
            if hypothesis.get("needs_market_data", False):
                # Determine what market data is needed
                if "gold" in user_query.lower():
                    gold_data = search_tool._run("gold_rate")
                    if gold_data.get("success"):
                        market_data["gold"] = gold_data
                elif "fd" in user_query.lower() or "fixed deposit" in user_query.lower():
                    fd_data = search_tool._run("fd_rates")
                    if fd_data.get("success"):
                        market_data["fd"] = fd_data
        
        state["market_data"] = market_data
        
        # Determine next node
        if state["needs_calculation"]:
            state["next_node"] = "analyst"
        else:
            state["next_node"] = "strategist"
        
    except Exception as e:
        state["errors"].append(f"Researcher error: {str(e)}")
        state["next_node"] = "strategist"  # Continue to strategist even on error
    
    return state


def analyst_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Analyst node: Performs financial calculations
    """
    state["current_node"] = "analyst"
    
    user_query = state["user_query"]
    user_profile = state.get("user_profile", {})
    
    math_tool = MathTool()
    
    # Determine what calculations are needed
    calculation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst. Determine what calculations are needed for the user's query.
Generate Python code to perform the calculations. Use the user profile data if available.

IMPORTANT RULES:
1. Code must be valid Python syntax
2. Use only simple expressions or statements
3. For EMI: EMI = (P * R * (1+R)**N) / ((1+R)**N - 1) where P=principal, R=monthly_rate, N=tenure_months
4. For SIP: FV = P * (((1+R)**N - 1) / R) where P=monthly_payment, R=monthly_rate, N=months
5. Return the result as a number or string
6. Do NOT use print statements - just calculate and return the value

User Profile: {profile}

Respond with JSON:
{{
    "calculations": [
        {{
            "description": "what this calculates",
            "code": "result = <valid python expression>",
            "variables": {{"var_name": value}}
        }}
    ]
}}"""),
        ("human", "User query: {query}\n\nResearch context: {context}")
    ])
    
    try:
        # Prepare context
        context = "\n".join([
            r.get("content", "")[:200] for r in state.get("research_results", [])[:3]
        ])
        
        profile_str = str(user_profile) if user_profile else "Not available"
        
        messages = calculation_prompt.format_messages(
            query=user_query,
            context=context,
            profile=profile_str
        )
        response = llm.invoke(messages)
        
        # Parse calculations needed
        json_match = re.search(r'\{[^{}]*"calculations"[^{}]*\}', response.content, re.DOTALL)
        
        if json_match:
            calc_data = json.loads(json_match.group())
            calculations = calc_data.get("calculations", [])
        else:
            # Fallback: try to extract Python code from response
            code_blocks = re.findall(r'```python\n(.*?)\n```', response.content, re.DOTALL)
            if code_blocks:
                calculations = [{
                    "description": "Financial calculation",
                    "code": code_blocks[0],
                    "variables": {}
                }]
            else:
                calculations = []
        
        # Execute calculations
        calc_results = []
        formulas = []
        
        for calc in calculations:
            code = calc.get("code", "").strip()
            description = calc.get("description", "Financial calculation")
            
            # Replace variables in code
            variables = calc.get("variables", {})
            for var_name, var_value in variables.items():
                code = code.replace(f"{{{var_name}}}", str(var_value))
            
            # Clean up code - remove markdown code blocks if present
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0].strip()
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0].strip()
            
            # Ensure code is a valid expression
            if not code.startswith("result =") and not code.startswith("result="):
                # If it's just an expression, wrap it
                if "=" not in code:
                    code = f"result = {code}"
            
            # Execute calculation
            result = math_tool._run(code, description)
            calc_results.append(result)
            formulas.append(code)
            
            state["tool_calls"].append({
                "node": "analyst",
                "tool": "math_tool",
                "description": description,
                "code": code,
                "result": result
            })
        
        state["calculations"] = calc_results
        state["calculation_formulas"] = formulas
        
        # Route to strategist
        state["next_node"] = "strategist"
        
    except Exception as e:
        state["errors"].append(f"Analyst error: {str(e)}")
        state["next_node"] = "strategist"  # Continue even on error
    
    return state


def strategist_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Strategist node: Synthesizes information and generates recommendations
    """
    state["current_node"] = "strategist"
    
    user_query = state["user_query"]
    user_profile = state.get("user_profile", {})
    research_results = state.get("research_results", [])
    calculations = state.get("calculations", [])
    market_data = state.get("market_data", {})
    
    # Synthesis prompt
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial strategist. Synthesize all available information to provide personalized financial advice.

Consider:
1. Research findings from knowledge base
2. Calculation results
3. Current market data
4. User's financial profile and constraints

Generate a comprehensive recommendation with:
- Clear answer to user's query
- Specific actionable steps
- Confidence level (high/medium/low)
- Any constraints or warnings

Format your response as a detailed financial recommendation."""),
        ("human", """User Query: {query}

User Profile:
{profile}

Research Findings:
{research}

Calculations:
{calculations}

Market Data:
{market_data}

Provide your financial recommendation:""")
    ])
    
    try:
        # Format inputs
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not available"
        
        research_str = "\n".join([
            f"- {r.get('content', '')[:300]}..." 
            for r in research_results[:5]
        ]) if research_results else "No research results available"
        
        calc_str = "\n".join([
            f"- {calc.get('description', 'Calculation')}: {calc.get('result', 'N/A')}"
            for calc in calculations
        ]) if calculations else "No calculations performed"
        
        market_str = json.dumps(market_data, indent=2) if market_data else "No market data"
        
        messages = synthesis_prompt.format_messages(
            query=user_query,
            profile=profile_str,
            research=research_str,
            calculations=calc_str,
            market_data=market_str
        )
        
        response = llm.invoke(messages)
        recommendation = response.content
        
        # Extract confidence if mentioned
        confidence = 0.7  # Default
        if "high confidence" in recommendation.lower():
            confidence = 0.9
        elif "low confidence" in recommendation.lower():
            confidence = 0.5
        
        # Check for constraint violations
        constraints_violated = []
        if user_profile:
            income = user_profile.get("income", {}).get("monthly", 0)
            expenses = user_profile.get("expenses", {}).get("monthly", 0)
            if expenses > income * 0.8:  # More than 80% of income
                constraints_violated.append("High expense-to-income ratio")
        
        state["final_recommendation"] = recommendation
        state["confidence_scores"] = {"overall": confidence}
        state["constraints_violated"] = constraints_violated
        state["next_node"] = "end"
        
        # Log final synthesis
        state["tool_calls"].append({
            "node": "strategist",
            "tool": "synthesis",
            "output_length": len(recommendation),
            "confidence": confidence
        })
        
    except Exception as e:
        state["errors"].append(f"Strategist error: {str(e)}")
        state["final_recommendation"] = f"Error generating recommendation: {str(e)}"
        state["next_node"] = "end"
    
    return state


def should_continue(state: FinancialAdvisoryState) -> str:
    """Determine next node based on state"""
    if state.get("next_node") == "end":
        return "end"
    
    next_node = state.get("next_node", "router")
    
    if next_node == "researcher":
        return "researcher"
    elif next_node == "analyst":
        return "analyst"
    elif next_node == "strategist":
        return "strategist"
    else:
        return "end"
