"""
Enterprise HR Analytics Platform
Main Streamlit Application with Multi-Page Navigation
"""

import streamlit as st
from attrition_analytics import render_attrition_dashboard
from hr_assistant import render_hr_assistant

# Page configuration
st.set_page_config(
    page_title="ACME Corp HR Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 0.5rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 2rem;
        font-weight: bold;
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .sidebar h1 {
        font-size: 1.2rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
    }
    .sidebar h2 {
        font-size: 0.9rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    .sidebar h3 {
        font-size: 0.85rem !important;
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        padding-top: 0 !important;
    }
    .sidebar p, .sidebar li {
        font-size: 0.75rem !important;
        margin-bottom: 0.1rem !important;
        line-height: 1.2 !important;
    }
    .sidebar .stRadio {
        font-size: 0.8rem !important;
    }
    .sidebar hr {
        margin: 0.3rem 0 !important;
    }
    .sidebar .stMetric {
        margin-bottom: 0.2rem !important;
    }
    .sidebar [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    .sidebar [data-testid="stMetricValue"] {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)
def main():
    """Main application with navigation"""
    # Initialize page in session state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# ACME Corp")
        st.markdown("### HR Analytics Platform")
        st.divider()
        st.markdown("## Navigation")
        # Get current page index for radio button
        page_options = ["Home", "HR Dashboard", "Ask HR Assistant"]
        current_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
        page = st.radio(
            "Select Module:",
            page_options,
            index=current_index,
            label_visibility="collapsed"
        )
        # Update session state if radio selection changes
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        st.divider()
        st.markdown("## Quick Links")
        if st.button("View Dashboard", use_container_width=True):
            st.session_state.page = "HR Dashboard"
            st.rerun()
        if st.button("Ask HR Bot", use_container_width=True):
            st.session_state.page = "Ask HR Assistant"
            st.rerun()
        st.divider()
        st.markdown("## Usage Stats")
        if 'chat_history' in st.session_state:
            st.metric("Questions Asked", len(st.session_state.chat_history))
        else:
            st.metric("Questions Asked", 0)
        st.divider()
        st.markdown("## Resources")
        st.markdown("- HR: hr@acmecorp.com\n- Help: +91-XXX-XXX-XXXX\n- Portal: portal.acmecorp.com")
    # Route to appropriate page based on session state
    if st.session_state.page == "Home":
        render_home_page()
    elif st.session_state.page == "HR Dashboard":
        render_attrition_dashboard()
    elif st.session_state.page == "Ask HR Assistant":
        render_hr_assistant()
def render_home_page():
    """Render home page"""
    # Custom CSS for compact layout
    st.markdown("""
        <style>
        .block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;}
        h1 {font-size: 1.6rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        h2 {font-size: 1.1rem !important; margin-bottom: 0.3rem !important; margin-top: 0.3rem !important;}
        h3 {font-size: 0.95rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        p {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stButton > button {font-size: 0.7rem !important; padding: 0.2rem 0.4rem !important; height: auto !important; margin-bottom: 0.2rem !important;}
        .stTextArea textarea {font-size: 0.85rem !important;}
        .stMarkdown {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        hr {margin: 0.3rem 0 !important;}
        div[data-testid="stExpander"] {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stAlert {padding: 0.4rem !important; font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
        div[data-testid="column"] {padding: 0 0.2rem !important;}
        </style>
    """, unsafe_allow_html=True)
    # Header
    st.markdown('<div class="main-header">ACME Corp HR Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enterprise-Level People Analytics & AI-Powered HR Solutions</div>', unsafe_allow_html=True)
    st.divider()
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### Welcome to Your HR Analytics Command Center
        This platform provides comprehensive insights into employee attrition patterns, 
        predictive analytics, and instant HR policy assistance powered by AI.
        """)
    st.divider()
    st.header("Platform Features")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("### HR Dashboard", expanded=False):
            st.markdown("""
            **Comprehensive diagnostic tools including:**
            - **Executive Summary** - Key metrics and costs
            - **Department Analysis** - Detailed departmental KPIs
            - **Role-Based Insights** - Job role attrition patterns
            - **Tenure Impact** - Retention by years at company
            - **Compensation Analysis** - Income vs. attrition correlation
            - **Satisfaction Drivers** - Key factors affecting retention
            - **Overtime Impact** - Work-life balance analysis
            - **Education Analysis** - Qualification-based insights
            - **Commute Distance** - Location impact on attrition
            - **Training ROI** - Learning investment effectiveness
            - **High-Risk Segments** - Critical employee groups
            - **Feature Importance** - Key attrition drivers
            - **Demographics** - Age and gender insights
            - **Correlations** - Feature relationship analysis
            - **Cohort Analysis** - Longitudinal trends
            - **Predictive Scoring** - ML-powered risk assessment
            - **Actionable Recommendations** - Data-driven strategies
            
            **Benefits:** Data-driven decision making, Proactive retention strategies, Cost optimization, Talent management insights
            """)
        if st.button("Launch Analytics Dashboard", type="primary", use_container_width=True):
            st.session_state.page = "HR Dashboard"
            st.rerun()
    with col2:
        with st.expander("### Ask HR Assistant", expanded=False):
            st.markdown("""
            **Intelligent policy guidance system:**
            - **Natural Language Q&A** - Ask in plain English
            - **Semantic Search** - Find relevant policies instantly
            - **Context-Aware** - Provides accurate, sourced answers
            - **Instant Responses** - No waiting for HR emails
            - **Policy Categories** - Leave, benefits, performance, etc.
            - **Common Questions** - Quick access to frequent queries
            - **Chat History** - Track your conversations
            - **Source References** - View original policy text
            
            **Covers topics:** Leave policies, Working hours, Remote work, Travel/expenses, Performance management, Learning & development, Benefits/insurance, Exit procedures, Dress code, Internal transfers, Grievance handling
            
            **Benefits:** 24/7 policy access, Reduced HR workload, Improved employee self-service, Consistent policy communication
            """)
        if st.button("Start Conversation with HR Bot", type="primary", use_container_width=True):
            st.session_state.page = "Ask HR Assistant"
            st.rerun()
    st.divider()
    st.header("Platform Capabilities")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Analytics Features", value="17", help="Comprehensive diagnostic tools and insights")
    with col2:
        st.metric(label="Visualization Charts", value="24+", help="Interactive charts and graphs")
    with col3:
        st.metric(label="HR Policy Topics", value="11", help="Covered by AI assistant")
    with col4:
        st.metric(label="AI Assistant Features", value="8", help="Intelligent guidance capabilities")
    st.divider()
    # Getting started
    st.header("Getting Started")
    st.markdown("""
    ### Quick Start Guide:
    1. **Explore Analytics Dashboard**
       - Click "Launch Analytics Dashboard" or use sidebar navigation
       - Review executive summary and key metrics
       - Drill down into specific analyses
       - Download reports for offline use
    2. **Use HR Assistant**
       - Click "Start Conversation with HR Bot" or use sidebar
       - Type your HR policy question or select a common question
       - Review the AI-generated answer with source references
       - Continue conversation for follow-up questions
    3. **Best Practices**
       - Review analytics regularly (weekly/monthly)
       - Act on high-risk employee alerts promptly
       - Share insights with department managers
       - Use HR bot for policy clarifications
       - Export data for presentations
    """)
    st.divider()
    # Footer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>ACME Corp HR Analytics Platform</strong></p>
            <p>Empowering HR decisions through data and AI</p>
            <p>© 2025 ACME Corporation. All rights reserved.</p>
        </div>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
