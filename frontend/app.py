"""
Streamlit Frontend for SHL Assessment Recommendation System
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        text-align: center;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy" and data.get("engine_status") == "healthy"
        return False
    except Exception as e:
        st.error(f"Cannot connect to API: {e}")
        return False


def get_recommendations(
    query: str,
    top_k: int = 10,
    balance_skills: bool = True,
    include_explanation: bool = False
) -> Dict[str, Any]:
    """Get recommendations from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={
                "query": query,
                "top_k": top_k,
                "balance_skills": balance_skills,
                "include_explanation": include_explanation
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            return {
                "success": False,
                "error": error_data.get("error", f"API returned status {response.status_code}")
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API. Please check if the API is running."}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


def display_recommendations(recommendations: List[Dict], query_analysis: Dict = None):
    """Display recommendations in a nice format"""
    
    if not recommendations:
        st.warning("No recommendations found for your query.")
        return
    
    # Display query analysis if available
    if query_analysis:
        st.subheader("📊 Query Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Technical Weight</h3>
                <h2>{query_analysis.get('technical_weight', 0):.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Behavioral Weight</h3>
                <h2>{query_analysis.get('behavioral_weight', 0):.0%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            skills = query_analysis.get('primary_skills', [])
            skills_text = ", ".join(skills[:3]) if skills else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Primary Skills</h3>
                <p>{skills_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Display recommendations
    st.subheader(f"🎯 Top {len(recommendations)} Recommendations")
    
    # Create DataFrame for table view
    df_data = []
    for i, rec in enumerate(recommendations, 1):
        df_data.append({
            "Rank": i,
            "Assessment Name": rec['assessment_name'],
            "Type": rec['test_type'],
            "Score": f"{rec['similarity_score']:.3f}",
            "Category": rec.get('category', 'N/A'),
            "URL": rec['assessment_url']
        })
    
    df = pd.DataFrame(df_data)
    
    # Display as table with clickable links
    st.dataframe(
        df,
        column_config={
            "URL": st.column_config.LinkColumn("Assessment URL"),
            "Score": st.column_config.NumberColumn(
                "Similarity Score",
                format="%.3f"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Display detailed cards
    st.markdown("---")
    st.subheader("📋 Detailed View")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['assessment_name']} (Score: {rec['similarity_score']:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Assessment Name:** {rec['assessment_name']}")
                st.markdown(f"**Test Type:** {rec['test_type']}")
                st.markdown(f"**Similarity Score:** {rec['similarity_score']:.3f}")
                if rec.get('category'):
                    st.markdown(f"**Category:** {rec['category']}")
                if rec.get('description'):
                    st.markdown(f"**Description:** {rec['description']}")
            
            with col2:
                st.markdown(f"**Assessment URL:**")
                st.markdown(f"[Open Assessment]({rec['assessment_url']})")
                st.markdown(f"**Rank:** #{i}")


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered recommendation system for SHL assessments</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API is healthy")
        else:
            st.error("❌ API is unavailable")
            st.info(f"API URL: {API_BASE_URL}")
        
        st.markdown("---")
        
        # Configuration
        st.subheader("Configuration")
        
        top_k = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of assessment recommendations to return"
        )
        
        balance_skills = st.checkbox(
            "Balance hard & soft skills",
            value=True,
            help="Automatically balance technical and behavioral assessments"
        )
        
        include_explanation = st.checkbox(
            "Include explanation",
            value=False,
            help="Generate AI explanation for recommendations (slower)"
        )
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.info(
            "This system uses AI to recommend relevant SHL assessments "
            "based on your job description or hiring needs."
        )
        
        st.markdown("---")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Java developers with collaboration skills",
            "Sales representatives with communication abilities",
            "Python programmers for data science",
            "Customer service with empathy",
            "Project managers with leadership"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.query = example
    
    # Main content
    st.markdown("---")
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong><br>
        1. Enter your job description or hiring requirements in the text area below<br>
        2. Adjust settings in the sidebar if needed<br>
        3. Click "Get Recommendations" to see relevant assessments<br>
        4. Review the recommendations and click on assessment URLs to learn more
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    st.subheader("📝 Enter Your Query")
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_area(
        "Job Description or Hiring Requirements",
        value=st.session_state.query,
        height=150,
        placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business team...",
        help="Enter a natural language description of the role or skills you're looking for"
    )
    
    # Alternative: URL input (optional)
    with st.expander("🔗 Or paste a job posting URL (optional)"):
        job_url = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/job-posting",
            help="Paste a URL to a job posting (feature coming soon)"
        )
        if job_url:
            st.info("URL parsing feature coming soon. Please use the text area above for now.")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button(
            "🚀 Get Recommendations",
            type="primary",
            use_container_width=True
        )
    
    # Process query
    if submit_button:
        if not query or not query.strip():
            st.error("Please enter a query before submitting.")
        else:
            # Show loading
            with st.spinner("🔍 Analyzing your query and finding the best assessments..."):
                result = get_recommendations(
                    query=query.strip(),
                    top_k=top_k,
                    balance_skills=balance_skills,
                    include_explanation=include_explanation
                )
            
            if result["success"]:
                data = result["data"]
                
                # Success message
                st.markdown(f"""
                <div class="success-message">
                    ✅ Found {data['total_found']} relevant assessments. 
                    Showing top {data['returned']} recommendations.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display explanation if available
                if data.get('explanation'):
                    st.subheader("💡 AI Explanation")
                    st.info(data['explanation'])
                    st.markdown("---")
                
                # Display recommendations
                display_recommendations(
                    data['recommendations'],
                    data.get('query_analysis')
                )
                
                # Download button
                st.markdown("---")
                st.subheader("📥 Export Results")
                
                # Create CSV
                csv_data = []
                for rec in data['recommendations']:
                    csv_data.append({
                        'Assessment Name': rec['assessment_name'],
                        'Test Type': rec['test_type'],
                        'Similarity Score': rec['similarity_score'],
                        'Category': rec.get('category', ''),
                        'Assessment URL': rec['assessment_url']
                    })
                
                df_export = pd.DataFrame(csv_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"shl_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                # Error message
                st.markdown(f"""
                <div class="error-message">
                    ❌ Error: {result['error']}
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "SHL Assessment Recommendation System | Powered by AI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
