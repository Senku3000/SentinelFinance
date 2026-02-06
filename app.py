"""Streamlit UI for SentinelFinance"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any

from src.graph import run_query
from src.config import Config
from src.tools.user_vault_tool import UserVaultTool
from src.ingestion.embedder import DocumentEmbedder

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


def load_user_profile(user_id: str) -> Dict[str, Any]:
    """Load user profile"""
    vault_tool = UserVaultTool()
    result = vault_tool._run("get_profile", user_id)
    if result.get("success"):
        return result["profile"]
    return None


def save_user_profile(user_id: str, profile: Dict[str, Any]):
    """Save user profile"""
    vault_tool = UserVaultTool()
    vault_tool._run("update_profile", user_id, profile)


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
        
        if st.button("Load Profile"):
            profile = load_user_profile(user_id)
            if profile:
                st.session_state.user_profile = profile
                st.success("Profile loaded!")
            else:
                st.info("No profile found. Using defaults.")
        
        st.markdown("---")
        
        # Profile Editor
        st.header("Edit Profile")
        
        if st.session_state.user_profile:
            profile = st.session_state.user_profile.copy()
        else:
            vault_tool = UserVaultTool()
            profile = vault_tool._get_default_profile(user_id)
        
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
        
        if st.button("Save Profile"):
            save_user_profile(user_id, profile)
            st.session_state.user_profile = profile
            st.success("Profile saved!")
        
        st.markdown("---")
        
        # Document Ingestion
        st.header("Document Management")
        if st.button("Ingest Documents"):
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
        1. Set up your profile (income, expenses, risk tolerance)
        2. Ask financial questions
        3. Get personalized advice!
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
                    result = run_query(
                        prompt,
                        user_id=st.session_state.user_id
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
                        with st.expander("📊 Calculations Performed"):
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
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": recommendation,
                        "metadata": metadata
                    })
                    
                except ValueError as e:
                    error_msg = f"Configuration error: {str(e)}"
                    st.error(error_msg)
                    st.info("Please set your GOOGLE_API_KEY in the .env file")
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
