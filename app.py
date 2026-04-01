"""Streamlit UI for Sentinel Finance"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any
from src.graph import run_query
from src.config import Config
from src.tools.user_vault_tool import UserVaultTool
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.document_parser import DocumentParser
from src.ingestion.user_embedder import UserEmbedder
from src.ingestion.llm_extractor import LLMExtractor, merge_extracted_data

# Page config
st.set_page_config(
    page_title="SentinelFinance - AI Financial Adviser",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = Config.DEFAULT_USER_ID
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "asked_clarifications" not in st.session_state:
    st.session_state.asked_clarifications = []
if "user_documents" not in st.session_state:
    st.session_state.user_documents = []
if "pending_extraction" not in st.session_state:
    st.session_state.pending_extraction = None


def _merge_profiles(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge user profile data onto defaults."""
    merged = defaults.copy()
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profiles(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_user_profile(user_id: str) -> Dict[str, Any]:
    """Load user profile"""
    vault_tool = UserVaultTool()
    result = vault_tool._run("get_profile", user_id)
    if result.get("success"):
        defaults = vault_tool._get_default_profile(user_id)
        return _merge_profiles(defaults, result["profile"])
    return None


def save_user_profile(user_id: str, profile: Dict[str, Any]):
    """Save user profile"""
    vault_tool = UserVaultTool()
    vault_tool._run("update_profile", user_id, profile)

def _parse_amount(text: str) -> int:
    text = text.replace(",", "").strip()
    try:
        return int(float(text))
    except Exception:
        return 0

def _apply_multiplier(value: float, suffix: str) -> int:
    if not suffix:
        return int(value)
    s = suffix.lower()
    if s == "l" or s == "lac" or s == "lakh":
        return int(value * 100000)
    if s == "k":
        return int(value * 1000)
    return int(value)

def _update_profile_from_text(profile: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Best-effort extraction of income/expenses from user text."""
    import re
    updated = profile.copy()
    updated.setdefault("income", {}).setdefault("monthly", 0)
    updated.setdefault("expenses", {}).setdefault("monthly", 0)
    
    inc_match = re.search(r"(?:income|salary|earn)[^0-9]*([0-9]+(?:\\.[0-9]+)?)(\\s*[lk]|\\s*lakh|\\s*lac)?", text, re.I)
    exp_match = re.search(r"(?:expense|spend|spending)[^0-9]*([0-9]+(?:\\.[0-9]+)?)(\\s*[lk]|\\s*lakh|\\s*lac)?", text, re.I)
    
    if inc_match:
        val = float(inc_match.group(1))
        suffix = (inc_match.group(2) or "").strip().lower().replace("lakh", "l").replace("lac", "l")
        updated["income"]["monthly"] = _apply_multiplier(val, suffix)
        updated["income"]["annual"] = updated["income"]["monthly"] * 12
    
    if exp_match:
        val = float(exp_match.group(1))
        suffix = (exp_match.group(2) or "").strip().lower().replace("lakh", "l").replace("lac", "l")
        updated["expenses"]["monthly"] = _apply_multiplier(val, suffix)
    
    return updated


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("💰 SentinelFinance")
        st.markdown("---")
        
        # User Profile Section
        st.header("User Profile")
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # Load profile on startup (or when user clicks Load)
        if st.button("Load Profile"):
            profile = load_user_profile(user_id)
            if profile:
                st.session_state.user_profile = profile
                st.success("Profile loaded!")
            else:
                st.info("No profile found. Using defaults.")

        # Initialize profile if not loaded yet
        vault_tool = UserVaultTool()
        if not st.session_state.user_profile:
            st.session_state.user_profile = vault_tool._get_default_profile(user_id)

        st.markdown("---")

        # --- Document Upload (BEFORE profile editor so extracted data shows up) ---
        st.header("Your Documents")

        uploaded_files = st.file_uploader(
            "Upload financial documents",
            type=["pdf", "xlsx", "xls", "csv", "txt", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Salary slips, tax returns, expense sheets, bank statements, etc.",
        )

        if uploaded_files:
            user_embedder = UserEmbedder()
            extractor = LLMExtractor()

            for uploaded_file in uploaded_files:
                # Save to user's documents directory
                docs_dir = Config.get_user_documents_path(user_id)
                dest_path = docs_dir / uploaded_file.name

                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Parse & embed into user's FAISS index
                    num_chunks = user_embedder.ingest_user_document(user_id, dest_path)

                    if num_chunks > 0:
                        st.success(f"{uploaded_file.name}: {num_chunks} chunks indexed")

                        # LLM extraction
                        parser = DocumentParser()
                        chunks = parser.parse_file(dest_path)
                        doc_text = "\n".join(c.content for c in chunks)

                        with st.spinner(f"Extracting financial data from {uploaded_file.name}..."):
                            extracted = extractor.extract(doc_text)

                        if "error" not in extracted:
                            summary = extracted.pop("document_summary", "Financial document")
                            st.info(f"Detected: {summary}")

                            with st.expander(f"Extracted data from {uploaded_file.name}"):
                                # Show extracted fields (skip nulls)
                                display = {
                                    k: v for k, v in extracted.items()
                                    if v is not None and v != {}
                                }
                                st.json(display)

                            # Auto-merge into profile
                            st.session_state.user_profile = merge_extracted_data(
                                st.session_state.user_profile, extracted
                            )
                            save_user_profile(user_id, st.session_state.user_profile)
                            st.success("Profile updated with extracted data!")
                        else:
                            st.warning(f"Could not extract structured data: {extracted.get('error')}")
                    else:
                        st.warning(f"No content extracted from {uploaded_file.name}")

            # Refresh document list
            st.session_state.user_documents = user_embedder.list_user_documents(user_id)

        # Show uploaded documents
        if not st.session_state.user_documents:
            embedder_check = UserEmbedder()
            st.session_state.user_documents = embedder_check.list_user_documents(user_id)

        if st.session_state.user_documents:
            st.subheader("Uploaded Documents")
            for doc in st.session_state.user_documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{doc['filename']} ({doc['file_type']})")
                with col2:
                    if st.button("Delete", key=f"del_{doc['filename']}"):
                        user_embedder = UserEmbedder()
                        user_embedder.delete_user_document(user_id, doc["filename"])
                        st.session_state.user_documents = user_embedder.list_user_documents(user_id)
                        st.rerun()

        st.markdown("---")

        # --- Profile Editor (AFTER document upload so it reflects extracted values) ---
        st.header("Edit Profile")

        profile = _merge_profiles(
            vault_tool._get_default_profile(user_id),
            st.session_state.user_profile.copy()
        )

        # Income
        st.subheader("Income")
        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=0,
            value=int(profile.get("income", {}).get("monthly", 150000)),
            step=10000
        )
        profile["income"]["monthly"] = monthly_income
        profile["income"]["annual"] = monthly_income * 12

        # Expenses
        st.subheader("Expenses")
        monthly_expenses = st.number_input(
            "Monthly Expenses (₹)",
            min_value=0,
            value=int(profile.get("expenses", {}).get("monthly", 50000)),
            step=5000
        )
        profile["expenses"]["monthly"] = monthly_expenses

        # Risk Tolerance
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(
                profile.get("risk_tolerance", "moderate")
            )
        )
        profile["risk_tolerance"] = risk_tolerance

        # Keep latest sidebar values in session
        st.session_state.user_profile = profile

        if st.button("Save Profile"):
            save_user_profile(user_id, profile)
            st.session_state.user_profile = profile
            st.success("Profile saved!")

        st.markdown("---")

        # Knowledge Base Ingestion
        with st.expander("Knowledge Base"):
            st.caption("Ingest generic financial knowledge documents")
            if st.button("Ingest Knowledge Base"):
                with st.spinner("Ingesting documents..."):
                    try:
                        embedder = DocumentEmbedder()
                        num_chunks = embedder.ingest_documents()
                        st.success(f"Ingested {num_chunks} chunks!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.markdown("---")

        # Info
        st.info("""
        **How to use:**
        1. Upload your financial documents (salary slips, tax returns, expense sheets)
        2. Review extracted data and edit your profile
        3. Ask financial questions
        4. Get personalized advice based on YOUR data!
        """)


    # Main area
    st.title(" AI Financial Adviser")
    st.markdown("Ask me anything about your finances!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional info if available
            if "metadata" in message:
                with st.expander("Details"):
                    if "calculations" in message["metadata"]:
                        st.subheader("Calculations")
                        for calc in message["metadata"]["calculations"]:
                            if calc.get("success"):
                                st.code(calc.get("formula", ""))
                                st.success(f"Result: {calc.get('result')}")
                    
                    if "confidence" in message["metadata"]:
                        st.metric("Confidence", f"{message['metadata']['confidence']*100:.1f}%")
    
    # Chat input
    if prompt := st.chat_input("Ask a financial question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                try:
                    # Validate API key
                    Config.validate()
                    
                    # Run query through graph
                    # Update profile from chat text if possible
                    if st.session_state.user_profile:
                        st.session_state.user_profile = _update_profile_from_text(
                            st.session_state.user_profile,
                            prompt
                        )
                    
                    result = run_query(
                        prompt,
                        user_id=st.session_state.user_id,
                        user_profile=st.session_state.user_profile,
                        asked_clarifications=st.session_state.asked_clarifications
                    )
                    
                    # Display recommendation
                    recommendation = result.get("recommendation", "No recommendation available")
                    st.markdown(recommendation)
                    
                    # Show metadata
                    metadata = {}
                    
                    # Show calculations if any
                    calculations = result.get("calculations", [])
                    if calculations:
                        metadata["calculations"] = calculations
                        with st.expander("Calculations Performed"):
                            for calc in calculations:
                                if calc.get("success"):
                                    st.code(calc.get("formula", ""))
                                    st.success(f"Result: {calc.get('result')}")
                                else:
                                    st.error(f"Error: {calc.get('error')}")
                    
                    # Show confidence
                    confidence = result.get("confidence", 0.5)
                    metadata["confidence"] = confidence
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Show errors if any
                    errors = result.get("errors", [])
                    if errors:
                        with st.expander(" Errors"):
                            for error in errors:
                                st.error(error)
                    
                    # Show clarification questions if present
                    clarifications = result.get("clarification_questions", [])
                    if clarifications:
                        with st.expander(" Clarifications Needed"):
                            for q in clarifications:
                                st.info(q)
                    
                    # Persist asked clarifications across turns
                    st.session_state.asked_clarifications = result.get("asked_clarifications", [])
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": recommendation,
                        "metadata": metadata
                    })
                    
                except ValueError as e:
                    error_msg = f"Configuration error: {str(e)}"
                    st.error(error_msg)
                    st.info("Please set your GROQ_API_KEY in the .env file")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
