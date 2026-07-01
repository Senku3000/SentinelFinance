"""Core nodes for LangGraph workflow"""

import json
import re
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .state import FinancialAdvisoryState
from .config import Config
from .llm import create_chat_llm
from .tools import MathTool, SearchTool, VectorDBTool, UserVaultTool
from .ingestion.user_embedder import UserEmbedder


llm = create_chat_llm(temperature=0.3)

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


def _format_indian_number(value: float, decimals: int = 0) -> str:
    """Format a number with Indian digit grouping."""
    rounded = round(float(value), decimals)
    sign = "-" if rounded < 0 else ""
    number = f"{abs(rounded):.{decimals}f}" if decimals else str(int(abs(rounded)))
    whole, _, fraction = number.partition(".")
    if len(whole) > 3:
        whole = ",".join([whole[:-3][max(i - 2, 0):i] for i in range(len(whole[:-3]) % 2 or 2, len(whole[:-3]) + 1, 2)] + [whole[-3:]])
    return f"{sign}{whole}{'.' + fraction if fraction else ''}"


def _format_inr(value: float, decimals: int = 0) -> str:
    """Format a rupee value for strategist context."""
    return f"₹{_format_indian_number(value, decimals)}"


def _format_lakh(value: float) -> str:
    """Format a rupee value in lakhs, e.g. 1800000 -> ₹18L."""
    lakhs = float(value) / 100000
    label = f"{lakhs:.2f}".rstrip("0").rstrip(".")
    return f"₹{label}L"


def _format_calculation_for_context(calc: Dict[str, Any]) -> str:
    """Create human-readable calculation context so the strategist doesn't re-parse raw numbers."""
    description = calc.get("description", "Calculation")
    result = calc.get("result")
    if not calc.get("success") or not isinstance(result, dict):
        return f"- {description}: {result if result is not None else 'Unavailable'}"

    if "future_value_percent_of_annual_salary" in result:
        return (
            f"- {description}: monthly SIP {_format_inr(result['monthly_sip'])}; "
            f"years {result['years']}; annual return {result['annual_return_percent']}%; "
            f"annual salary {_format_inr(result['annual_salary'])} ({_format_lakh(result['annual_salary'])}); "
            f"total invested {_format_inr(result['total_invested'])} ({_format_lakh(result['total_invested'])}); "
            f"future value {_format_inr(result['future_value'], 2)} ({_format_lakh(result['future_value'])}); "
            f"future value is {result['future_value_percent_of_annual_salary']}% of annual salary; "
            f"principal invested is {result['total_invested_percent_of_annual_salary']}% of annual salary."
        )

    if "months_needed_from_surplus" in result:
        return (
            f"- {description}: purchase cost {_format_inr(result['purchase_cost'])}; "
            f"monthly income {_format_inr(result['monthly_income'])}; "
            f"monthly expenses {_format_inr(result['monthly_expenses'])}; "
            f"monthly surplus {_format_inr(result['monthly_surplus'])}; "
            f"cost is {result['cost_percent_of_monthly_surplus']}% of one month's surplus; "
            f"months needed from surplus {result['months_needed_from_surplus']}; "
            f"affordable from one month surplus: {result['affordable_from_one_month_surplus']}."
        )

    if "current_monthly_investment" in result:
        return (
            f"- {description}: current monthly investment {_format_inr(result['current_monthly_investment'])}; "
            f"monthly income {_format_inr(result['monthly_income'])}; "
            f"investment is {result['investment_percent_of_monthly_income']}% of monthly income."
        )

    if "monthly_surplus" in result:
        return (
            f"- {description}: monthly income {_format_inr(result['monthly_income'])}; "
            f"monthly expenses {_format_inr(result['monthly_expenses'])}; "
            f"monthly surplus {_format_inr(result['monthly_surplus'])}; "
            f"expense ratio {result['expense_ratio_percent']}%."
        )

    return f"- {description}: {result}"


def router_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Router node: Analyzes user intent and determines routing
    """
    state["current_node"] = "router"
    state["iteration_count"] += 1
    
    if state["iteration_count"] >= state["max_iterations"]:
        state["next_node"] = "end"
        state["errors"].append("Maximum iterations reached")
        return state
    
    user_query = state["user_query"]
    
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
        messages = intent_prompt.format_messages(query=user_query)
        response = llm.invoke(messages)
        
        response_text = response.content
        
        intent_data = _load_json_from_text(response_text)
        if not isinstance(intent_data, dict):
            intent_data = {
                "needs_research": "research" in response_text.lower() or "knowledge" in response_text.lower(),
                "needs_calculation": "calculation" in response_text.lower() or "calculate" in response_text.lower(),
                "needs_user_profile": "profile" in response_text.lower() or "income" in response_text.lower(),
                "reasoning": response_text,
                "query_type": "general"
            }
        
        state["intent"] = intent_data
        state["needs_research"] = intent_data.get("needs_research", False)
        state["needs_calculation"] = intent_data.get("needs_calculation", False)
        state["needs_user_profile"] = intent_data.get("needs_user_profile", False)
        state["needs_clarification"] = intent_data.get("needs_clarification", False)

        if state["needs_user_profile"] and state.get("user_profile"):
            state["needs_user_profile"] = False

        profile = state.get("user_profile") or {}
        existing_investments = profile.get("existing_investments") or {}
        is_current_investment_query = (
            any(word in user_query.lower() for word in ["investing", "investment", "invested"])
            and any(word in user_query.lower() for word in ["current", "currently", "right now", "how much"])
        )
        if existing_investments and is_current_investment_query:
            state["needs_calculation"] = True
        
        if state["needs_research"] or state["needs_user_profile"]:
            state["next_node"] = "researcher"
        elif state["needs_calculation"]:
            state["next_node"] = "analyst"
        else:
            state["next_node"] = "researcher"  # Default: always research first
        
        state["tool_calls"].append({
            "node": "router",
            "tool": "intent_analysis",
            "input": user_query,
            "output": intent_data
        })
        
    except Exception as e:
        state["errors"].append(f"Router error: {str(e)}")
        state["next_node"] = "researcher"
        state["needs_research"] = True
    
    return state


def researcher_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Researcher node: Generates hypotheses and retrieves evidence
    """
    state["current_node"] = "researcher"
    
    user_query = state["user_query"]
    
    vector_db_tool = VectorDBTool()
    search_tool = SearchTool()
    user_vault_tool = UserVaultTool()
    
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
        messages = hypothesis_prompt.format_messages(query=user_query)
        response = llm.invoke(messages)
        
        parsed = _load_json_from_text(response.content)
        if isinstance(parsed, list):
            hypotheses = parsed
        else:
            hypotheses = [{
                "hypothesis": f"User needs information about: {user_query}",
                "needs_vector_db": True,
                "needs_market_data": False
            }]
        
        state["hypotheses"] = hypotheses
        
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
            state["needs_user_profile"] = False
        
        research_results = []
        for hypothesis in hypotheses:
            if hypothesis.get("needs_vector_db", True):
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

        market_data = {}

        if "gold" in user_query.lower():
            gold_data = search_tool._run("gold_rate")
            if gold_data.get("success"):
                market_data["gold"] = gold_data
        if "fd" in user_query.lower() or "fixed deposit" in user_query.lower():
            fd_data = search_tool._run("fd_rates")
            if fd_data.get("success"):
                market_data["fd"] = fd_data

        if not market_data:
            try:
                extract_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Extract a short web search query from the user's message. Focus on the product, price, or factual lookup needed. Return ONLY the search query, nothing else. If no search is needed, return 'NONE'."),
                    ("human", "{query}")
                ])
                search_response = llm.invoke(extract_prompt.format_messages(query=user_query))
                search_query = search_response.content.strip().strip('"')

                if search_query and search_query.upper() != "NONE":
                    web_data = search_tool._run(search_query)
                    if web_data.get("success"):
                        market_data["web_search"] = web_data
            except Exception:
                pass  # Web search is best-effort

        state["market_data"] = market_data
        
        if state["needs_calculation"]:
            state["next_node"] = "analyst"
        else:
            state["next_node"] = "evidence_scorer"
        
    except Exception as e:
        state["errors"].append(f"Researcher error: {str(e)}")
        if state["needs_calculation"]:
            state["next_node"] = "analyst"
        else:
            state["next_node"] = "strategist"  # Continue to strategist even on error
    
    return state


def analyst_node(state: FinancialAdvisoryState) -> FinancialAdvisoryState:
    """
    Analyst node: Performs financial calculations
    """
    state["current_node"] = "analyst"
    
    user_query = state["user_query"]
    user_profile = state.get("user_profile", {})
    user_query_lower = user_query.lower()
    
    math_tool = MathTool()

    def _parse_amount(value: str, suffix: Optional[str] = None) -> int:
        amount = float(value.replace(",", ""))
        suffix = suffix or ""
        if suffix in ("l", "lac", "lakh"):
            amount *= 100000
        elif suffix == "k":
            amount *= 1000
        return int(amount)

    def _extract_sip_amount(query: str) -> Optional[int]:
        amount = r"(\d[\d,]*(?:\.\d+)?)\s*(l|lac|lakh|k)?"
        patterns = [
            rf"\binvest(?:ing)?\b\D{{0,30}}?{amount}\D{{0,20}}?\bsip\b",
            rf"{amount}\s*(?:per\s+month|/month|monthly)?\D{{0,20}}?\bsip\b",
            rf"\bsip\b\s+(?:of|for)\s+{amount}",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return _parse_amount(match.group(1), match.group(2))
        return None

    def _extract_years(query: str) -> Optional[int]:
        match = re.search(r"(\d+)\s*(?:years|year|yrs|yr)\b", query)
        return int(match.group(1)) if match else None

    def _extract_annual_return(query: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:annual\s*)?(?:returns?|return|cagr)", query)
        if not match:
            match = re.search(r"(?:annual\s*)?(?:returns?|return|cagr)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", query)
        return float(match.group(1)) / 100 if match else None

    def _extract_purchase_amount(query: str) -> Optional[int]:
        amount = r"(\d[\d,]*(?:\.\d+)?)\s*(l|lac|lakh|k)?"
        patterns = [
            rf"\b(?:for|costs?|price|priced|worth)\b\D{{0,20}}?{amount}",
            rf"{amount}\s*(?:rupees?|rs|inr)?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                value, suffix = match.group(1), match.group(2)
                if suffix or float(value.replace(",", "")) >= 1000:
                    return _parse_amount(value, suffix)
        return None

    def _deterministic_calculations() -> list[Dict[str, Any]]:
        calculations = []
        income = (user_profile.get("income") or {}).get("monthly")
        expenses = (user_profile.get("expenses") or {}).get("monthly")
        annual_salary = (user_profile.get("income") or {}).get("annual")
        monthly_surplus = income - expenses if income and expenses is not None else None
        existing_investments = user_profile.get("existing_investments") or {}

        if income and expenses is not None and any(word in user_query_lower for word in ["income", "salary", "earn", "expense", "spend", "afford", "buy"]):
            calculations.append({
                "description": "Monthly surplus and expense ratio",
                "code": (
                    "income = {income}\n"
                    "expenses = {expenses}\n"
                    "monthly_surplus = income - expenses\n"
                    "expense_ratio_percent = (expenses / income) * 100\n"
                    "result = {\n"
                    "    'monthly_income': income,\n"
                    "    'monthly_expenses': expenses,\n"
                    "    'monthly_surplus': monthly_surplus,\n"
                    "    'expense_ratio_percent': round(expense_ratio_percent, 2),\n"
                    "}"
                ),
                "variables": {"income": income, "expenses": expenses}
            })

        purchase_amount = _extract_purchase_amount(user_query_lower)
        if (
            purchase_amount
            and income
            and expenses is not None
            and monthly_surplus
            and any(word in user_query_lower for word in ["afford", "buy", "purchase"])
        ):
            calculations.append({
                "description": "Purchase affordability using monthly surplus",
                "code": (
                    "purchase_cost = {purchase_cost}\n"
                    "income = {income}\n"
                    "expenses = {expenses}\n"
                    "monthly_surplus = income - expenses\n"
                    "months_needed = purchase_cost / monthly_surplus\n"
                    "result = {\n"
                    "    'purchase_cost': purchase_cost,\n"
                    "    'monthly_income': income,\n"
                    "    'monthly_expenses': expenses,\n"
                    "    'monthly_surplus': monthly_surplus,\n"
                    "    'cost_percent_of_monthly_surplus': round((purchase_cost / monthly_surplus) * 100, 2),\n"
                    "    'months_needed_from_surplus': round(months_needed, 2),\n"
                    "    'affordable_from_one_month_surplus': purchase_cost <= monthly_surplus,\n"
                    "}"
                ),
                "variables": {
                    "purchase_cost": purchase_amount,
                    "income": income,
                    "expenses": expenses,
                }
            })

        is_current_investment_query = (
            any(word in user_query_lower for word in ["investing", "investment", "invested"])
            and any(word in user_query_lower for word in ["current", "currently", "right now", "how much"])
        )
        investment_values = [
            value for value in existing_investments.values()
            if isinstance(value, (int, float)) and value > 0
        ]
        if income and investment_values and is_current_investment_query:
            calculations.append({
                "description": "Current monthly investment as percentage of monthly income",
                "code": (
                    "monthly_income = {monthly_income}\n"
                    "current_monthly_investment = {current_monthly_investment}\n"
                    "result = {\n"
                    "    'monthly_income': monthly_income,\n"
                    "    'current_monthly_investment': current_monthly_investment,\n"
                    "    'investment_percent_of_monthly_income': round((current_monthly_investment / monthly_income) * 100, 2),\n"
                    "}"
                ),
                "variables": {
                    "monthly_income": income,
                    "current_monthly_investment": sum(investment_values),
                }
            })

        sip_amount = _extract_sip_amount(user_query_lower)
        years = _extract_years(user_query_lower)
        annual_return = _extract_annual_return(user_query_lower)

        if sip_amount and years and annual_salary and annual_return is not None:
            calculations.append({
                "description": "SIP future value as percentage of annual salary",
                "code": (
                    "monthly_sip = {monthly_sip}\n"
                    "years = {years}\n"
                    "annual_salary = {annual_salary}\n"
                    "annual_return = {annual_return}\n"
                    "months = years * 12\n"
                    "monthly_rate = annual_return / 12\n"
                    "total_invested = monthly_sip * months\n"
                    "if monthly_rate == 0:\n"
                    "    future_value = total_invested\n"
                    "else:\n"
                    "    future_value = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate)\n"
                    "result = {\n"
                    "    'monthly_sip': monthly_sip,\n"
                    "    'years': years,\n"
                    "    'annual_return_percent': round(annual_return * 100, 2),\n"
                    "    'annual_salary': annual_salary,\n"
                    "    'total_invested': round(total_invested, 2),\n"
                    "    'future_value': round(future_value, 2),\n"
                    "    'future_value_percent_of_annual_salary': round((future_value / annual_salary) * 100, 2),\n"
                    "    'total_invested_percent_of_annual_salary': round((total_invested / annual_salary) * 100, 2),\n"
                    "}"
                ),
                "variables": {
                    "monthly_sip": sip_amount,
                    "years": years,
                    "annual_salary": annual_salary,
                    "annual_return": annual_return,
                }
            })

        return calculations
    
    calculation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst. Determine what calculations are needed for the user's query.
Generate Python code to perform the calculations. Use the user profile data if available.

IMPORTANT RULES:
1. Code must be valid Python syntax
2. Use only simple expressions or statements
3. For EMI: EMI = (P * R * (1+R)**N) / ((1+R)**N - 1) where P=principal, R=monthly_rate, N=tenure_months
4. For SIP: FV = P * (((1+R)**N - 1) / R) where P=monthly_payment, R=monthly_rate, N=months
5. For affordability: use monthly surplus = income - expenses; do not invent a percentage-of-income rule
6. Return the result as a number or string
7. Do NOT use print statements - just calculate and return the value

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
        context = "\n".join([
            r.get("content", "")[:200] for r in state.get("research_results", [])[:3]
        ])
        
        profile_str = str(user_profile) if user_profile else "Not available"

        calculations = _deterministic_calculations()
        if not calculations:
            messages = calculation_prompt.format_messages(
                query=user_query,
                context=context,
                profile=profile_str
            )
            response = llm.invoke(messages)

            calc_data = _load_json_from_text(response.content)
            if isinstance(calc_data, dict):
                calculations = calc_data.get("calculations", [])
            else:
                code_blocks = re.findall(r'```python\n(.*?)\n```', response.content, re.DOTALL)
                if code_blocks:
                    calculations = [{
                        "description": "Financial calculation",
                        "code": code_blocks[0],
                        "variables": {}
                    }]
                else:
                    calculations = []
        
        calc_results = []
        formulas = []
        
        for calc in calculations:
            code = calc.get("code", "").strip()
            description = calc.get("description", "Financial calculation")
            
            variables = calc.get("variables", {})
            variable_assignments = []
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                if placeholder not in code and re.fullmatch(r"[A-Za-z_]\w*", str(var_name)):
                    variable_assignments.append(f"{var_name} = {repr(var_value)}")
                code = code.replace(placeholder, repr(var_value))
            
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0].strip()
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0].strip()
            
            if not code.startswith("result =") and not code.startswith("result="):
                if "=" not in code:
                    code = f"result = {code}"

            if variable_assignments:
                code = "\n".join(variable_assignments + [code])
            
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
    else:
        if scores:
            avg_score = sum(scores) / len(scores)
            evidence_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_score)))
        else:
            evidence_score = 0.4

        if evidence_score >= 0.5:
            quality = "high"
        elif evidence_score >= 0.2:
            quality = "medium"
        else:
            quality = "low"

        state["evidence_score"] = evidence_score
        state["evidence_quality"] = quality

    if state["needs_calculation"] and not state.get("calculations"):
        state["next_node"] = "analyst"
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

    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sharp, honest personal financial adviser who knows the user well.

RULES:
- Be DIRECT. Answer the question first, then explain briefly. No fluff.
- Use the user's ACTUAL numbers (income, expenses, savings) in your answer. Don't say "consider your budget" - say "you make ₹1.1L and spend ₹30k, so you have ₹80k/month to work with."
- Give your real OPINION. If something is a bad financial decision, say so clearly and why. If it's fine, say that too.
- Think practically - compare costs to the user's monthly savings, not in abstract terms.
- Use the Calculations section for computed results. Never mention internal calculation/tool/pipeline failures to the user.
- If a required number is missing, ask for that user-facing input plainly instead of saying a calculation was not performed.
- Keep it SHORT. 3-5 sentences for simple questions. Only go longer if the user asked for a detailed breakdown.
- Don't add generic disclaimers like "consult a financial advisor" or "this is not financial advice." You ARE the advisor.
- If you don't have enough info, say what specific info you need - don't give a vague non-answer."""),
        ("human", """User's question: {query}

User Profile:
{profile}

Knowledge Base:
{research}

Personal Documents:
{personal_docs}

Calculations:
{calculations}

Market Data:
{market_data}

Answer directly:""")
    ])
    
    try:
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not available"
        
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
            _format_calculation_for_context(calc)
            for calc in calculations
        ]) if calculations else "No calculation results available"
        
        market_str = json.dumps(market_data, indent=2) if market_data else "No market data"
        
        messages = synthesis_prompt.format_messages(
            query=user_query,
            profile=profile_str,
            research=research_str,
            personal_docs=personal_doc_str,
            calculations=calc_str,
            market_data=market_str,
        )
        
        response = llm.invoke(messages)
        recommendation = response.content
        
        confidence = 0.7  # Default
        if "high confidence" in recommendation.lower():
            confidence = 0.9
        elif "low confidence" in recommendation.lower():
            confidence = 0.5
        
        constraints_violated = []
        if user_profile:
            income = user_profile.get("income", {}).get("monthly") or 0
            expenses = user_profile.get("expenses", {}).get("monthly") or 0
            if income > 0 and expenses > income * 0.8:
                constraints_violated.append("High expense-to-income ratio")
        
        state["final_recommendation"] = recommendation
        state["confidence_scores"] = {"overall": confidence}
        state["constraints_violated"] = constraints_violated
        state["next_node"] = "end"
        
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
