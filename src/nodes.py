"""Core nodes for LangGraph workflow"""

import json
import re
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .state import FinancialAdvisoryState
from .config import Config
from .tools import MathTool, SearchTool, VectorDBTool, UserVaultTool
from .ingestion.user_embedder import UserEmbedder


# Initialize LLM
llm = ChatGroq(
    model=Config.GROQ_MODEL,
    groq_api_key=Config.GROQ_API_KEY,
    temperature=0.3
)

def _extract_json_payload(text: str) -> Optional[str]:
    """Extract the first JSON object or array from text using bracket balancing."""
    if not text:
        return None
    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        return None
    if start_obj == -1 or (start_arr != -1 and start_arr < start_obj):
        start = start_arr
        open_ch, close_ch = "[", "]"
    else:
        start = start_obj
        open_ch, close_ch = "{", "}"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _load_json_from_text(text: str) -> Optional[Any]:
    """Best-effort JSON parsing from model output."""
    payload = _extract_json_payload(text)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


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
    "needs_clarification": true/false,
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
        
        intent_data = _load_json_from_text(response_text)
        if not isinstance(intent_data, dict):
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
        state["needs_clarification"] = intent_data.get("needs_clarification", False)
        
        # Determine next node
        if state.get("needs_clarification"):
            state["next_node"] = "clarifier"
        elif state["needs_research"]:
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
        parsed = _load_json_from_text(response.content)
        if isinstance(parsed, list):
            hypotheses = parsed
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
        elif state.get("user_profile"):
            # Profile already present (e.g., from UI)
            state["needs_user_profile"] = False
        
        # Step 3: Retrieve from vector DB
        research_results = []
        for hypothesis in hypotheses:
            if hypothesis.get("needs_vector_db", True):
                # Search vector DB
                search_query = hypothesis.get("hypothesis", user_query)
                db_result = vector_db_tool._run(search_query, k=3)
                
                if db_result.get("success"):
                    for item in db_result["results"]:
                        item["retrieval_type"] = "general_finance"
                        research_results.append(item)
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
        
        # Step 3b: Search user's personal document index
        user_doc_results = []
        try:
            user_embedder = UserEmbedder()
            if user_embedder.has_documents(state["user_id"]):
                search_query = state["user_query"]
                user_results = user_embedder.search_user_documents(
                    state["user_id"], search_query, k=3
                )
                user_doc_results = user_results
                research_results.extend(user_results)

                state["tool_calls"].append({
                    "node": "researcher",
                    "tool": "user_document_search",
                    "query": search_query,
                    "results_count": len(user_results)
                })
        except Exception as e:
            state["errors"].append(f"User document search error: {str(e)}")

        state["user_doc_results"] = user_doc_results
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
            state["next_node"] = "evidence_scorer"
        
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
        calc_data = _load_json_from_text(response.content)
        if isinstance(calc_data, dict):
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


def clarifier_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Clarifier node: asks targeted questions when critical info is missing
    """
    state["current_node"] = "clarifier"
    
    user_query = state["user_query"]
    user_profile = state.get("user_profile") or {}
    intent = state.get("intent") or {}
    
    # Check what info is available from user documents
    user_doc_results = state.get("user_doc_results", [])
    user_docs_meta = state.get("user_documents", [])
    doc_context = ""
    if user_doc_results:
        doc_snippets = [r.get("content", "")[:200] for r in user_doc_results[:3]]
        doc_context = "\nUser's uploaded documents contain:\n" + "\n".join(f"- {s}" for s in doc_snippets)
    if user_docs_meta:
        doc_names = [d.get("filename", "") for d in user_docs_meta]
        doc_context += f"\nUploaded files: {', '.join(doc_names)}"

    clarifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial clarification agent. Ask 1-3 short questions
only if critical information is missing. Avoid asking for info already present
in the user profile OR in the user's uploaded documents.

Return JSON:
{{
  "needs_clarification": true/false,
  "questions": ["question1", "question2", "question3"],
  "missing_fields": ["field1","field2"]
}}"""),
        ("human", """User query: {query}

Intent: {intent}
User profile: {profile}
{doc_context}
""")
    ])
    
    try:
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not available"
        messages = clarifier_prompt.format_messages(
            query=user_query,
            intent=json.dumps(intent),
            profile=profile_str,
            doc_context=doc_context
        )
        response = llm.invoke(messages)
        parsed = _load_json_from_text(response.content)
        
        if isinstance(parsed, dict):
            needs_clarification = bool(parsed.get("needs_clarification"))
            questions = parsed.get("questions", []) or []
            missing_fields = parsed.get("missing_fields", []) or []
        else:
            needs_clarification = False
            questions = []
            missing_fields = []
        
        asked = set(q.strip().lower() for q in state.get("asked_clarifications", []))
        profile = state.get("user_profile") or {}
        income_val = (profile.get("income") or {}).get("monthly", 0)
        expense_val = (profile.get("expenses") or {}).get("monthly", 0)
        risk_val = profile.get("risk_tolerance")
        goals_val = profile.get("goals") or []
        
        def _is_redundant(q: str) -> bool:
            ql = q.lower()
            if income_val and any(k in ql for k in ["income", "salary", "earn"]):
                return True
            if expense_val and any(k in ql for k in ["expense", "spend"]):
                return True
            if risk_val and "risk" in ql:
                return True
            if goals_val and "goal" in ql:
                return True
            return False
        filtered_questions = []
        for q in questions:
            q_norm = q.strip().lower()
            if q_norm and q_norm not in asked and not _is_redundant(q_norm):
                filtered_questions.append(q)
        
        state["needs_clarification"] = needs_clarification
        state["clarification_questions"] = filtered_questions[:3]
        state["asked_clarifications"].extend(filtered_questions[:3])
        
        if needs_clarification and filtered_questions:
            state["final_recommendation"] = (
                "I need a bit more information to answer accurately:\n- "
                + "\n- ".join(filtered_questions[:3])
            )
            state["next_node"] = "end"
        else:
            state["next_node"] = "strategist"
        
        if missing_fields:
            state["assumptions"].append(
                f"Missing fields for analysis: {', '.join(missing_fields)}"
            )
        
        state["tool_calls"].append({
            "node": "clarifier",
            "tool": "clarifier_prompt",
            "output": {
                "needs_clarification": needs_clarification,
                "questions": questions[:3],
                "missing_fields": missing_fields
            }
        })
        
    except Exception as e:
        state["errors"].append(f"Clarifier error: {str(e)}")
        state["next_node"] = "router"
    
    return state


def evidence_scorer_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Evidence scorer: rates quality/coverage of retrieved evidence
    """
    state["current_node"] = "evidence_scorer"
    
    research_results = state.get("research_results", [])
    scores = [s.get("score", 0.0) for s in state.get("document_scores", [])]
    
    if not research_results:
        state["evidence_score"] = 0.0
        state["evidence_quality"] = "low"
        state["needs_clarification"] = True
        state["clarification_questions"] = [
            "I couldn't find enough relevant information. Can you provide more details or context?"
        ]
        state["next_node"] = "clarifier"
        return state
    
    # Simple heuristic: lower FAISS distance score is better, so invert for quality.
    if scores:
        avg_score = sum(scores) / len(scores)
        evidence_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_score)))
    else:
        evidence_score = 0.4
    
    if evidence_score >= 0.7:
        quality = "high"
    elif evidence_score >= 0.4:
        quality = "medium"
    else:
        quality = "low"
    
    state["evidence_score"] = evidence_score
    state["evidence_quality"] = quality
    
    if quality == "low":
        state["needs_clarification"] = True
        state["clarification_questions"] = [
            "I need a bit more detail to answer accurately (e.g., amounts, tenure, risk level)."
        ]
        state["next_node"] = "clarifier"
    else:
        state["next_node"] = "strategist"
    
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
    assumptions = state.get("assumptions", [])
    evidence_quality = state.get("evidence_quality", "low")
    
    user_doc_results = state.get("user_doc_results", [])

    # Synthesis prompt
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial strategist. Synthesize all available information to provide personalized financial advice.

Consider:
1. Research findings from knowledge base
2. Personal document findings (from user's own uploaded financial documents)
3. Calculation results
4. Current market data
5. User's financial profile and constraints

IMPORTANT: When personal document findings are available, prioritize them — they contain the user's actual financial data. Reference specific numbers from their documents when giving advice.

Generate a comprehensive recommendation with:
- Clear answer to user's query
- Specific actionable steps
- Confidence level (high/medium/low)
- Any constraints or warnings

Format your response as a detailed financial recommendation."""),
        ("human", """User Query: {query}

User Profile:
{profile}

Knowledge Base Findings:
{research}

Personal Document Findings:
{personal_docs}

Calculations:
{calculations}

Market Data:
{market_data}

Assumptions:
{assumptions}

Evidence Quality:
{evidence_quality}

Provide your financial recommendation:""")
    ])
    
    try:
        # Format inputs
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not available"
        
        # Separate knowledge base results from personal document results
        kb_results = [r for r in research_results if r.get("retrieval_type") != "personal_document"]
        research_str = "\n".join([
            f"- {r.get('content', '')[:300]}..."
            for r in kb_results[:5]
        ]) if kb_results else "No research results available"

        personal_doc_str = "\n".join([
            f"- [From {r.get('metadata', {}).get('file_name', 'document')}]: {r.get('content', '')[:300]}..."
            for r in user_doc_results[:5]
        ]) if user_doc_results else "No personal documents uploaded"
        
        calc_str = "\n".join([
            f"- {calc.get('description', 'Calculation')}: {calc.get('result', 'N/A')}"
            for calc in calculations
        ]) if calculations else "No calculations performed"
        
        market_str = json.dumps(market_data, indent=2) if market_data else "No market data"
        
        messages = synthesis_prompt.format_messages(
            query=user_query,
            profile=profile_str,
            research=research_str,
            personal_docs=personal_doc_str,
            calculations=calc_str,
            market_data=market_str,
            assumptions="\n".join(f"- {a}" for a in assumptions) if assumptions else "None",
            evidence_quality=evidence_quality
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
    elif next_node == "clarifier":
        return "clarifier"
    elif next_node == "evidence_scorer":
        return "evidence_scorer"
    else:
        return "end"
