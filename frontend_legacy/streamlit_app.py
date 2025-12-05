"""
Streamlit Web Interface for Legal AI Agent
Quick prototype for testing and demonstration
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="LexiBot Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000/api/v1"
API_TOKEN = "your-secret-token"  # Replace with your actual token

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c5aa0;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.analysis-box {
    background-color: #f8f9fa;
    border-left: 5px solid #2c5aa0;
    padding: 1rem;
    margin: 1rem 0;
}
.risk-high { color: #dc3545; font-weight: bold; }
.risk-medium { color: #ffc107; font-weight: bold; }
.risk-low { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def call_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make API call with caching"""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", 
                               json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"success": False, "error": str(e)}

def display_analysis_result(result: Dict[str, Any]):
    """Display analysis results in formatted way"""
    if not result.get("success", False):
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    analysis = result.get("analysis", {})
    confidence = result.get("confidence_score", 0)
    processing_time = result.get("processing_time", 0)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence Score", f"{confidence:.2%}")
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col3:
        st.metric("Model Version", result.get("model_version", "N/A"))
    
    # Display analysis results
    for question, answer in analysis.items():
        with st.expander(f"📋 {question}", expanded=True):
            st.markdown(f'<div class="analysis-box">{answer}</div>', 
                       unsafe_allow_html=True)

def extract_risk_level(text: str) -> str:
    """Extract risk level from analysis text"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["high risk", "severe", "critical", "dangerous"]):
        return "High"
    elif any(word in text_lower for word in ["medium risk", "moderate", "caution"]):
        return "Medium"
    else:
        return "Low"

# Main App
def main():
    st.markdown('<h1 class="main-header">⚖️ LexiBot Legal AI Assistant</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", [
        "Contract Analysis", 
        "Contract Comparison", 
        "Clause Extraction",
        "AI Agent Chat",
        "Analytics Dashboard"
    ])
    
    if page == "Contract Analysis":
        contract_analysis_page()
    elif page == "Contract Comparison":
        contract_comparison_page()
    elif page == "Clause Extraction":
        clause_extraction_page()
    elif page == "AI Agent Chat":
        agent_chat_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()

def contract_analysis_page():
    st.markdown('<h2 class="sub-header">📄 Contract Analysis</h2>', 
                unsafe_allow_html=True)
    
    # Input methods
    input_method = st.radio("Choose input method:", 
                           ["Upload File", "Paste Text"], horizontal=True)
    
    contract_text = ""
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload Contract", 
                                       type=['txt', 'pdf', 'docx'],
                                       help="Supported formats: TXT, PDF, DOCX")
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                contract_text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("PDF/DOCX parsing coming soon! Please use TXT files for now.")
                return
    else:
        contract_text = st.text_area("Paste contract text here:", 
                                   height=300, 
                                   placeholder="Enter or paste your contract text...")
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox("Analysis Type:", 
                                   ["comprehensive", "quick", "specific"])
    with col2:
        custom_questions = st.checkbox("Add custom questions")
    
    questions = []
    if custom_questions:
        questions_text = st.text_area("Enter questions (one per line):",
                                    placeholder="Who are the parties?\nWhat are the key risks?")
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    # Run analysis
    if st.button("🔍 Analyze Contract", type="primary"):
        if not contract_text.strip():
            st.error("Please provide contract text to analyze.")
            return
        
        with st.spinner("Analyzing contract... This may take a few moments."):
            request_data = {
                "text": contract_text,
                "analysis_type": analysis_type,
                "questions": questions if questions else None
            }
            
            result = call_api("analyze-contract", request_data)
            display_analysis_result(result)
            
            # Save to session state for dashboard
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            
            st.session_state.analysis_history.append({
                "timestamp": datetime.now(),
                "type": "contract_analysis",
                "confidence": result.get("confidence_score", 0),
                "processing_time": result.get("processing_time", 0)
            })

def contract_comparison_page():
    st.markdown('<h2 class="sub-header">🔄 Contract Comparison</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contract A")
        contract_a = st.text_area("First contract:", height=300, key="contract_a")
    
    with col2:
        st.subheader("Contract B")
        contract_b = st.text_area("Second contract:", height=300, key="contract_b")
    
    # Focus areas
    focus_areas = st.multiselect("Focus on specific areas (optional):", [
        "Payment Terms", "Termination Clauses", "Liability", 
        "Intellectual Property", "Confidentiality", "Warranties"
    ])
    
    if st.button("🔍 Compare Contracts", type="primary"):
        if not contract_a.strip() or not contract_b.strip():
            st.error("Please provide both contracts to compare.")
            return
        
        with st.spinner("Comparing contracts..."):
            request_data = {
                "contract_a": contract_a,
                "contract_b": contract_b,
                "focus_areas": focus_areas if focus_areas else None
            }
            
            result = call_api("compare-contracts", request_data)
            display_analysis_result(result)

def clause_extraction_page():
    st.markdown('<h2 class="sub-header">📝 Clause Extraction</h2>', 
                unsafe_allow_html=True)
    
    contract_text = st.text_area("Contract text:", height=300)
    
    clause_types = st.multiselect("Select clause types to extract:", [
        "termination", "payment", "liability", "intellectual_property",
        "confidentiality", "warranties", "indemnification", "force_majeure"
    ], default=["termination", "liability"])
    
    if st.button("🔍 Extract Clauses", type="primary"):
        if not contract_text.strip() or not clause_types:
            st.error("Please provide contract text and select clause types.")
            return
        
        with st.spinner("Extracting clauses..."):
            request_data = {
                "text": contract_text,
                "clause_types": clause_types
            }
            
            result = call_api("extract-clauses", request_data)
            display_analysis_result(result)

def agent_chat_page():
    st.markdown('<h2 class="sub-header">🤖 AI Agent Chat</h2>', 
                unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask me anything about legal documents...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                request_data = {
                    "query": user_query,
                    "use_planning": True,
                    "context": {"chat_history": st.session_state.chat_history}
                }
                
                result = call_api("agent-query", request_data)
                
                if result.get("success", False):
                    response = result["analysis"].get("response", "I apologize, but I couldn't process your request.")
                    st.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                else:
                    error_msg = f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

def analytics_dashboard_page():
    st.markdown('<h2 class="sub-header">📊 Analytics Dashboard</h2>', 
                unsafe_allow_html=True)
    
    if "analysis_history" not in st.session_state or not st.session_state.analysis_history:
        st.info("No analysis history available. Perform some contract analyses to see statistics here.")
        return
    
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    with col3:
        avg_time = df['processing_time'].mean()
        st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    with col4:
        recent_analyses = len(df[df['timestamp'] > (datetime.now() - pd.Timedelta(days=1))])
        st.metric("Last 24h", recent_analyses)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig = px.histogram(df, x='confidence', nbins=10, 
                          title="Confidence Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time trend
        fig = px.scatter(df, x='timestamp', y='processing_time', 
                        color='type', title="Processing Time Over Time")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()