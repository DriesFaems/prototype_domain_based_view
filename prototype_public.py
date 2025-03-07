import streamlit as st
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import matplotlib.pyplot as plt
from pydantic_ai import Agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq 
from langchain.agents import initialize_agent, Tool
import nest_asyncio
# Apply nest_asyncio to allow nested event loops
import asyncio
try:
    nest_asyncio.apply()
except RuntimeError:
    # Create a new event loop and set it as the current one if none exists
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()

# Page configuration
st.set_page_config(
    page_title="Company Classification Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .competitor-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .info-text {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("This tool analyzes whether a company is market-based or domain-based based on its description, mission, and vision.")
    
    st.markdown("### What does it do?")
    st.markdown("""
    - Analyzes your company's orientation
    - Identifies top competitors
    - Compares your company with competitors
    - Provides detailed explanations
    """)
    
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your company details
    2. Click 'Analyze Company'
    3. Review the results
    """)
    
    st.markdown("---")
    st.markdown("### Definitions")
    with st.expander("Market-based View"):
        st.markdown("""
        A **market-based view** is focused on the ambition to become a leader in a particular industry, 
        product category or market segment. Companies with this orientation typically:
        - Focus on existing market needs
        - Are reactive to market trends
        - Prioritize meeting current customer requirements
        """)
    
    with st.expander("Domain-based View"):
        st.markdown("""
        A **domain-based view** is focused on the ambition to address human-centric needs across different 
        industries, product categories or market segments. Companies with this orientation typically:
        - Develop deep expertise in specific domains
        - Create new markets through innovation
        - Are more proactive than reactive
        """)

# Main content
st.markdown("<h1 class='main-header'>Company Classification Tool</h1>", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2 = st.tabs(["Company Analysis", "Results"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Enter Company Information")
    st.markdown("Please provide detailed information about your company to get an accurate analysis.")
    
    # Input fields with better formatting
    company_name = st.text_input('Company Name', help='Enter the name of your company')
    
    col1, col2 = st.columns(2)
    with col1:
        company_description = st.text_area('Company Description', 
                                          help='Provide a detailed description of the company, including products/services, target market, and business model',
                                          height=150)
    with col2:
        company_mission = st.text_area('Company Mission', 
                                      help='Describe the core mission of the company - why it exists and what it aims to achieve',
                                      height=150)
    
    company_vision = st.text_area('Company Vision', 
                                 help='Describe the core vision of the company - what it aspires to become in the future',
                                 height=100)
    
    # Add example button to help users
    if st.button('Show Example'):
        st.info("""
        **Example Company:**
        - **Name:** TechSolutions Inc.
        - **Description:** TechSolutions is a software company that develops AI-powered productivity tools for businesses of all sizes. Our flagship product helps teams collaborate more effectively through smart task management and automated workflows.
        - **Mission:** To empower organizations to achieve more with less effort through intelligent software solutions.
        - **Vision:** To become the leading provider of AI-enhanced productivity tools that transform how work gets done.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # API key handling
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]

        serper_api_key = st.secrets["SERPER_API_KEY"]

        os.environ['GROQ_API_KEY'] = groq_api_key
        os.environ['SERPER_API_KEY'] = serper_api_key
    except Exception as e:
        st.error(f"Error loading API keys: {str(e)}")

    # Pydantic model for the analysis result
    class CompanyAnalysis(BaseModel):
        score: float = Field(description="Score from 0-10 where 10 is strongly aligned with the classification")
        explanation: str = Field(description="Detailed explanation for the classification and score")

    # Center the analyze button and make it more prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button('Analyze Company', use_container_width=True, type="primary")

    # Run analysis when button is clicked
    if analyze_button:
        if not company_name or not company_description or not company_mission or not company_vision:
            st.error("⚠️ Please fill in all fields to proceed with the analysis")
        else:
            # Store company name in session state
            if 'company_name' not in st.session_state:
                st.session_state.company_name = company_name
            
            # Switch to results tab
            st.session_state.active_tab = "Results"
            
            with tab2:
                st.markdown(f"## Analysis for {company_name}")
                
                # Create a progress bar for the analysis process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Company Analysis
                status_text.text("Analyzing company orientation...")
                progress_bar.progress(10)

                system_prompt = """The user will provide a description of the mission, vision and business strategy of the company.
                As a business analyst, you need to evalute these statements and provide an indication of whether the description reflects a market-based-view or a domain-based-view.
                A market-based-view is focused on the ambition to become a leader in a particular industry, product category or market segment. A domain-based view is focused on the ambition to address human-centric needs across different industries, product categories or market segments.
                You should provide a classification on a scale of 0 to 10 where 0 means that the company's statements are purely market-based and 10 means that the company's statements are purely domain-based."""
                user_prompt = f"""Description of the mission and vision of the company:
                Company: {company_name}
                Mission: {company_mission}
                Vision: {company_vision}
                """

                model = 'groq:qwen-2.5-32b'

                agent = Agent(model,
                            system_prompt=system_prompt,
                            result_type=CompanyAnalysis,
                            )

                result = agent.run_sync(user_prompt)
                json_result = result.data.model_dump()
                
                # Store the focal company score for later comparison
                st.session_state.focal_company_score = json_result['score']
                
                progress_bar.progress(30)
                status_text.text("Identifying competitors...")
            
                search = GoogleSerperAPIWrapper()
                llm = ChatGroq(temperature=0, model="qwen-2.5-32b")
                tools = [
                    Tool(
                        name="Search for company information",
                        func=search.run,
                        description="useful for when you need to ask with search"
                    )
                ]

                # generate the chain for the search agent

                self_ask_with_search = initialize_agent(tools, llm, verbose=True, handle_parsing_errors=True)
                
                competitor_system_prompt = """You are a business intelligence expert. Your task is to search the web to identify the three most important competitors of company""" + company_name + """.  When using the search_web tool:
1. First search for competitors to identify potential competitors
2. Then search for specific information about each competitor's mission and vision
3. Use separate search queries for each competitor

Consider factors like industry, product offerings, target market, and business model.
Provide a clear explanation of why you selected these competitors. For each competitor, provide information on the company's mission and vision. Remember to format your final answer as a valid JSON object that conforms to this schema:
{'competitorlist': [{'competitorname': str, 'competitordescription': str, 'competitor_mission': str, 'competitor_vision': str}]}"""

                competitor_json = self_ask_with_search.run(competitor_system_prompt)

                # Extract JSON from the response
                import json
                import re

                # Check if the response is already a string representation of JSON
                if isinstance(competitor_json, str):
                    # Try to find JSON within code blocks if present
                    json_match = re.search(r"```json\n(.*?)\n```", competitor_json, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no code blocks, use the entire string
                        json_str = competitor_json
                    
                    # Parse the string into a Python dictionary
                    try:
                        competitor_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to clean the string
                        # Remove any potential leading/trailing characters that aren't part of JSON
                        cleaned_str = re.search(r'(\{.*\})', json_str, re.DOTALL)
                        if cleaned_str:
                            try:
                                competitor_data = json.loads(cleaned_str.group(1))
                            except json.JSONDecodeError:
                                competitor_data = {"error": "Could not parse JSON from the response"}
                        else:
                            competitor_data = {"error": "Could not parse JSON from the response"}
                else:
                    # If it's already a dictionary, use it directly
                    competitor_data = competitor_json
                
                progress_bar.progress(60)
                status_text.text("Analyzing competitors...")
                
                # Pydantic model for competitor analysis
                class CompetitorAnalysis(BaseModel):
                    score: float = Field(..., description="Score from 0 to 10 indicating if the competitor is more market-based (0) or domain-based (10)")
                    explanation: str = Field(..., description="Explanation of the score and analysis")
                
                # Initialize competitor scores dictionary
                if 'competitor_scores' not in st.session_state:
                    st.session_state.competitor_scores = {}
                
                # Analyze each competitor
                for i, competitor in enumerate(competitor_data['competitorlist']):
                    progress_value = 60 + (i + 1) * 10
                    progress_bar.progress(min(progress_value, 90))
                    status_text.text(f"Analyzing competitor: {competitor['competitorname']}...")
                    
                    competitor_analysis_system_prompt = """The user will provide a description of the mission and vision of the company.
                    As a business analyst, you need to evalute these statements and provide an indication of whether the description reflects a market-based-view or a domain-based-view.
                A market-based-view is focused on the ambition to become a leader in a particular industry, product category or market segment. A domain-based view is focused on the ambition to address human-centric needs across different industries, product categories or market segments.
                You should provide a classification on a scale of 0 to 10 where 0 means that the company's statements are purely market-based and 10 means that the company's statements are purely domain-based."""
                
                
                    
                    competitor_analysis_user_prompt = f"""Company Name: {competitor['competitorname']}
                    Company Mission: {competitor['competitor_mission']}
                    Company Vision: {competitor['competitor_vision']}
                    Company Description: {competitor['competitordescription']}
                    
                    Please analyze this company and determine if they are more market-based or domain-based."""
                    
                    competitor_analysis_agent = Agent(model,
                                                    system_prompt=competitor_analysis_system_prompt,
                                                    result_type=CompetitorAnalysis,
                                                    )
                    
                    try:
                        competitor_analysis_result = competitor_analysis_agent.run_sync(competitor_analysis_user_prompt)
                        competitor_analysis_json = competitor_analysis_result.data.model_dump()
                        
                        # Save this competitor's score
                        st.session_state.competitor_scores[competitor['competitorname']] = competitor_analysis_json['score']
                        
                        # Store the full analysis for display
                        if 'competitor_analyses' not in st.session_state:
                            st.session_state.competitor_analyses = {}
                        
                        st.session_state.competitor_analyses[competitor['competitorname']] = {
                            'score': competitor_analysis_json['score'],
                            'explanation': competitor_analysis_json['explanation'],
                            'mission': competitor['competitor_mission'],
                            'vision': competitor['competitor_vision'],
                            'description': competitor['competitordescription']
                        }
                        
                    except Exception as e:
                        st.error(f"Error analyzing {competitor['competitorname']}: {str(e)}")
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Display results in a well-organized format
                st.markdown("<h2 class='sub-header'>Company Orientation Analysis</h2>", unsafe_allow_html=True)
                
                # Create a visual score indicator
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"### {company_name}")
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Create a gauge-like visualization
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='lightgray'))
                    ax.add_patch(plt.Circle((0.5, 0.5), 0.35, color='white'))
                    
                    # Add a colored arc based on the score
                    score = json_result['score']
                    if score < 3.33:
                        color = '#ef5350'  # Red for market-based
                        label = "Market-Based"
                    elif score < 6.67:
                        color = '#ffca28'  # Yellow for balanced
                        label = "Balanced"
                    else:
                        color = '#66bb6a'  # Green for domain-based
                        label = "Domain-Based"
                    
                    ax.text(0.5, 0.5, f"{score:.1f}", ha='center', va='center', fontsize=20, fontweight='bold')
                    ax.text(0.5, 0.3, f"{label}", ha='center', va='center', fontsize=10)
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown(f"**Score: {json_result['score']:.1f}/10**")
                    if json_result['score'] < 3.33:
                        st.markdown("**Classification: Market-Based**")
                    elif json_result['score'] < 6.67:
                        st.markdown("**Classification: Balanced Approach**")
                    else:
                        st.markdown("**Classification: Domain-Based**")
                    
                    st.markdown("### Analysis")
                    st.markdown(json_result['explanation'])
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Competitor Analysis Section
                st.markdown("<h2 class='sub-header'>Competitor Analysis</h2>", unsafe_allow_html=True)
                
                # Create tabs for each competitor
                competitor_tabs = st.tabs([comp['competitorname'] for comp in competitor_data['competitorlist']])
                
                for i, (tab, competitor) in enumerate(zip(competitor_tabs, competitor_data['competitorlist'])):
                    with tab:
                        st.markdown("<div class='competitor-card'>", unsafe_allow_html=True)
                        st.markdown(f"### {competitor['competitorname']}")
                        
                        # Display competitor info
                        with st.expander("Company Information", expanded=True):
                            st.markdown(f"**Mission:** {competitor['competitor_mission']}")
                            st.markdown(f"**Vision:** {competitor['competitor_vision']}")
                            st.markdown(f"**Description:** {competitor['competitordescription']}")
                        
                        # Display analysis if available
                        if competitor['competitorname'] in st.session_state.competitor_analyses:
                            analysis = st.session_state.competitor_analyses[competitor['competitorname']]
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                # Create a gauge-like visualization
                                fig, ax = plt.subplots(figsize=(3, 3))
                                ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='lightgray'))
                                ax.add_patch(plt.Circle((0.5, 0.5), 0.35, color='white'))
                                
                                # Add a colored arc based on the score
                                score = analysis['score']
                                if score < 3.33:
                                    color = '#ef5350'  # Red for market-based
                                    label = "Market-Based"
                                elif score < 6.67:
                                    color = '#ffca28'  # Yellow for balanced
                                    label = "Balanced"
                                else:
                                    color = '#66bb6a'  # Green for domain-based
                                    label = "Domain-Based"
                                
                                ax.text(0.5, 0.5, f"{score:.1f}", ha='center', va='center', fontsize=20, fontweight='bold')
                                ax.text(0.5, 0.3, f"{label}", ha='center', va='center', fontsize=10)
                                
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.axis('off')
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown(f"**Score: {analysis['score']:.1f}/10**")
                                if analysis['score'] < 3.33:
                                    st.markdown("**Classification: Market-Based**")
                                elif analysis['score'] < 6.67:
                                    st.markdown("**Classification: Balanced Approach**")
                                else:
                                    st.markdown("**Classification: Domain-Based**")
                                
                                with st.expander("View Detailed Analysis"):
                                    st.markdown(analysis['explanation'])
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Comparative Analysis
                st.markdown("<h2 class='sub-header'>Comparative Analysis</h2>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Prepare data for visualization
                companies = list(st.session_state.competitor_scores.keys())
                scores = list(st.session_state.competitor_scores.values())
                
                # Add the focal company
                companies.append(company_name)
                scores.append(st.session_state.focal_company_score)
                
                # Sort by score for better visualization
                sorted_data = sorted(zip(companies, scores), key=lambda x: x[1])
                companies_sorted = [x[0] for x in sorted_data]
                scores_sorted = [x[1] for x in sorted_data]
                
                # Create a horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(companies_sorted, scores_sorted, 
                              color=['#1f77b4' if company != company_name else '#ff7f0e' for company in companies_sorted])
                
                # Add a vertical line at 5 to indicate the middle point
                ax.axvline(x=5, color='gray', linestyle='--', alpha=0.7)
                
                # Add labels
                ax.text(0.5, -0.15, 'Market-Based', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.text(0.9, -0.15, 'Domain-Based', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
                # Add value labels to the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                           va='center', fontsize=10)
                
                ax.set_xlabel('Domain-Based Score (0 = Market-Based, 10 = Domain-Based)')
                ax.set_title('Domain vs Market Orientation Comparison')
                ax.set_xlim(0, 10)
                
                # Add a legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#ff7f0e', label=f'{company_name} (Your Company)'),
                    Patch(facecolor='#1f77b4', label='Competitors')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
                
                st.pyplot(fig)
                
                # Add explanation
                st.info("""
                This chart compares all companies on the market-based vs domain-based spectrum:
                - Scores closer to 0 indicate a more market-based approach
                - Scores closer to 10 indicate a more domain-based approach
                - Your company is highlighted in orange
                """)
                
                # Add insights section
                st.markdown("### Key Insights")
                
                # Calculate average competitor score
                avg_competitor_score = sum(st.session_state.competitor_scores.values()) / len(st.session_state.competitor_scores)
                
                # Determine if the company is more or less domain-based than competitors
                if st.session_state.focal_company_score > avg_competitor_score:
                    difference = st.session_state.focal_company_score - avg_competitor_score
                    st.markdown(f"- Your company is **more domain-based** than the average competitor by {difference:.1f} points")
                else:
                    difference = avg_competitor_score - st.session_state.focal_company_score
                    st.markdown(f"- Your company is **more market-based** than the average competitor by {difference:.1f} points")
                
                # Find the most similar competitor
                similarity_scores = {comp: abs(score - st.session_state.focal_company_score) 
                                    for comp, score in st.session_state.competitor_scores.items()}
                most_similar = min(similarity_scores.items(), key=lambda x: x[1])
                
                st.markdown(f"- The most similar competitor to your company is **{most_similar[0]}** with a difference of only {most_similar[1]:.1f} points")
                
                # Find the most different competitor
                most_different = max(similarity_scores.items(), key=lambda x: x[1])
                st.markdown(f"- The most different competitor from your company is **{most_different[0]}** with a difference of {most_different[1]:.1f} points")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                
                
                # Add download options
                st.markdown("<h2 class='sub-header'>Export Results</h2>", unsafe_allow_html=True)
                
                # Prepare data for export
                export_data = {
                    "company_name": company_name,
                    "company_description": company_description,
                    "company_mission": company_mission,
                    "company_vision": company_vision,
                    "analysis": {
                        "score": float(st.session_state.focal_company_score),
                        "explanation": json_result['explanation']
                    },
                    "competitors": {}
                }
                
                for comp_name, analysis in st.session_state.competitor_analyses.items():
                    export_data["competitors"][comp_name] = {
                        "score": float(analysis['score']),
                        "explanation": analysis['explanation'],
                        "mission": analysis['mission'],
                        "vision": analysis['vision'],
                        "description": analysis['description']
                    }
                
                # Convert to JSON for download
                json_str = json.dumps(export_data, indent=2)
                
                # Create download button
                st.download_button(
                    label="Download Analysis as JSON",
                    data=json_str,
                    file_name=f"{company_name.replace(' ', '_')}_analysis.json",
                    mime="application/json"
                )
               