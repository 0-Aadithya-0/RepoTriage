import streamlit as st
import requests

# Set page config for a cleaner look
st.set_page_config(
    page_title="RepoTriage Dashboard",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Constants
API_URL = "http://127.0.0.1:8000/api/v1/analyze"

# Custom CSS for styling
st.markdown("""
<style>
    .issue-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .issue-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .priority-critical { background-color: #FF5252; color: white; }
    .priority-high { background-color: #FF9800; color: white; }
    .priority-medium { background-color: #FFC107; color: black; }
    .priority-low { background-color: #4CAF50; color: white; }
    
    .category-bug { border: 1px solid #FF5252; color: #FF5252; }
    .category-feature { border: 1px solid #4CAF50; color: #4CAF50; }
    .category-docs { border: 1px solid #2196F3; color: #2196F3; }
    .category-support { border: 1px solid #9C27B0; color: #9C27B0; }
    .category-other { border: 1px solid #9E9E9E; color: #9E9E9E; }
    
    a { text-decoration: none; color: inherit; }
    a:hover { color: #64B5F6; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 RepoTriage Dashboard")
st.markdown("Instantly triage and prioritize recent open issues from any public GitHub repository using AI.")

# Helper functions for styling
def get_priority_class(priority: str) -> str:
    return f"priority-{priority.lower()}"

def get_category_class(category: str) -> str:
    cat = category.lower()
    if "bug" in cat: return "category-bug"
    if "feature" in cat: return "category-feature"
    if "doc" in cat: return "category-docs"
    if "support" in cat: return "category-support"
    return "category-other"

# Input Form
with st.form("analyze_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        owner = st.text_input("Repository Owner", value="tiangolo", placeholder="e.g., tiangolo")
    with col2:
        repo = st.text_input("Repository Name", value="fastapi", placeholder="e.g., fastapi")
    with col3:
        limit = st.number_input("Limit", min_value=1, max_value=10, value=5)
        
    submitted = st.form_submit_button("Analyze Issues", use_container_width=True)

# Handle Submission
if submitted:
    if not owner or not repo:
        st.error("Please provide both an owner and a repository name.")
    else:
        with st.spinner("Fetching and analyzing issues... This usually takes 5–15 seconds."):
            try:
                response = requests.post(
                    API_URL, 
                    json={"owner": owner.strip(), "repo": repo.strip(), "limit": limit}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"Analysis complete for **{data['repository']}**")
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Issues Analyzed", data["analyzed_count"])
                    m2.metric("Failed to Analyze", data["failed_count"])
                    m3.metric("Total Requested", data["requested_limit"])
                    
                    st.divider()
                    
                    # Display Issues
                    if data["issues"]:
                        st.subheader("Analyzed Issues")
                        for issue in data["issues"]:
                            p_class = get_priority_class(issue['priority_level'])
                            c_class = get_category_class(issue['category'])
                            
                            st.markdown(f"""
                            <div class="issue-card">
                                <a href="{issue['html_url']}" target="_blank">
                                    <div class="issue-title">#{issue['issue_number']} - {issue['title']}</div>
                                </a>
                                <div>
                                    <span class="badge {p_class}">Priority: {issue['priority_level']}</span>
                                    <span class="badge {c_class}">Category: {issue['category']}</span>
                                </div>
                                <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #E0E0E0;">
                                    {issue['tldr_summary']}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Display Failures
                    if data["failures"]:
                        st.subheader("Analysis Failures")
                        for failure in data["failures"]:
                            st.error(f"Issue #{failure['issue_number']}: {failure['error']}")
                            
                else:
                    # Handle API Errors gracefully
                    try:
                        error_data = response.json()
                        st.error(f"**Error ({error_data.get('error', 'unknown')}):** {error_data.get('message', 'Unknown error')}")
                    except ValueError:
                        st.error(f"API Error {response.status_code}: Make sure the FastAPI server is running.")
                        
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API. Please ensure the FastAPI server is running on http://127.0.0.1:8000.")
