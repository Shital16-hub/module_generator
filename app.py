"""
Streamlit UI for Training Generator Agent

Simple interface for generating training modules using RAG-powered AI agent.
Works dynamically for ANY module without hardcoding.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.training_generator.agent import training_agent
from agents.training_generator.state import create_initial_state

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Training Module Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 8px;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_module_name(query: str) -> str:
    """
    Intelligently extract module name from user query.
    
    Examples:
    - "Create training for Payment module" → "Payment"
    - "Generate training for Inventory Management" → "Inventory Management"
    - "Build training on Authentication" → "Authentication"
    - "Training for the Order Processing system" → "Order Processing"
    """
    
    # Clean up query
    query = query.strip()
    
    # Common patterns to extract module name
    patterns = [
        r'training (?:for|on|about) (?:the )?(.+?)(?:\s+module|\s+system|\s+feature)?$',
        r'create .+ for (?:the )?(.+?)(?:\s+module|\s+system)?$',
        r'generate .+ (?:for|on) (?:the )?(.+?)(?:\s+module)?$',
        r'build .+ (?:for|on) (?:the )?(.+?)(?:\s+module)?$',
        r'(?:for|on|about) (?:the )?(.+?)(?:\s+module|\s+system)?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            module = match.group(1).strip()
            # Capitalize each word
            return ' '.join(word.capitalize() for word in module.split())
    
    # Fallback: Look for capitalized words (likely proper nouns)
    words = query.split()
    capitalized_words = []
    for word in words:
        # Skip common words
        if word.lower() in ['create', 'generate', 'build', 'training', 'module', 'for', 'the', 'a', 'an']:
            continue
        if word and (word[0].isupper() or len(word) > 4):
            capitalized_words.append(word.capitalize())
    
    if capitalized_words:
        return ' '.join(capitalized_words[:3])  # Take up to 3 words
    
    # Last resort: Return "Module" as generic
    return "Module"


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🎓 Training Module Generator</h1>
    <p>AI-powered training materials from JIRA, Confluence, and Zephyr</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# User Input
st.subheader("📝 Enter Your Query")

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "What training module would you like to generate?",
        placeholder="e.g., Create training for Payment Processing",
        help="Describe the module you want training for - the agent will automatically detect it",
        label_visibility="collapsed"
    )

with col2:
    st.write("")  # Spacer
    generate_button = st.button("🚀 Generate", type="primary", use_container_width=True)

# Quick Examples
with st.expander("💡 Example Queries - Try Any Module!"):
    st.markdown("""
    The agent works with **any module** - just describe what you need:
    
    **Examples:**
    - `Create training for Payment Processing`
    - `Generate training materials for Inventory Management`
    - `Build training on User Authentication`
    - `Training for Order Management System`
    - `Create comprehensive training for Search Functionality`
    - `Generate training on Notification Service`
    - `Build training package for Reporting Dashboard`
    - `Training for Customer Support Module`
    
    **Tips:**
    - Include the module/feature name clearly
    - Use natural language
    - The agent will search your indexed data automatically
    """)

# ============================================================================
# GENERATION LOGIC
# ============================================================================

if generate_button:
    if not user_query:
        st.error("⚠️ Please enter a query first!")
    else:
        # Extract module name dynamically
        module_name = extract_module_name(user_query)
        
        # Show detected module
        st.markdown(f'<div class="info-box">🎯 <strong>Detected Module:</strong> {module_name}</div>', unsafe_allow_html=True)
        
        # Show progress
        with st.spinner(f"🔄 Generating training module for **{module_name}**..."):
            
            # Progress steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create initial state
                status_text.text("📋 Initializing agent...")
                progress_bar.progress(10)
                
                initial_state = create_initial_state(user_query, module_name)
                
                # Run agent
                status_text.text("🔍 Searching knowledge base...")
                progress_bar.progress(30)
                
                status_text.text("📚 Gathering documentation and test cases...")
                progress_bar.progress(60)
                
                final_state = training_agent.invoke(initial_state)
                
                status_text.text("📝 Generating training content...")
                progress_bar.progress(90)
                
                status_text.text("✅ Generation complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # ============================================================
                # DISPLAY RESULTS
                # ============================================================
                
                if final_state['markdown_output']:
                    st.markdown('<div class="success-box">✅ <strong>Training module generated successfully!</strong></div>', unsafe_allow_html=True)
                    
                    # Statistics
                    st.subheader("📊 Generation Statistics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("📚 Stories", len(final_state['stories']))
                    
                    with col2:
                        st.metric("📖 Documentation", len(final_state['documentation']))
                    
                    with col3:
                        st.metric("🧪 Test Cases", len(final_state['test_cases']))
                    
                    with col4:
                        st.metric("📦 Total Artifacts", final_state['total_artifacts_found'])
                    
                    with col5:
                        st.metric("🔄 Iterations", f"{final_state['iteration']}/{final_state['max_iterations']}")
                    
                    st.divider()
                    
                    # Download button
                    st.subheader("💾 Download Training Module")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"training_{module_name.lower().replace(' ', '_')}_{timestamp}.md"
                    
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    with col1:
                        st.download_button(
                            label="📥 Download Markdown",
                            data=final_state['markdown_output'],
                            file_name=filename,
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Copy to clipboard button (using streamlit)
                        if st.button("📋 Copy to Clipboard", use_container_width=True):
                            st.code(final_state['markdown_output'], language="markdown")
                            st.info("👆 Select all and copy the content above")
                    
                    with col3:
                        # Save locally
                        output_path = Path(filename)
                        output_path.write_text(final_state['markdown_output'], encoding='utf-8')
                        st.success(f"💾 Saved: {filename}")
                    
                    st.divider()
                    
                    # Display markdown
                    st.subheader("📄 Generated Training Module")
                    
                    # Show in tabs
                    tab1, tab2 = st.tabs(["📖 Rendered View", "📝 Raw Markdown"])
                    
                    with tab1:
                        st.markdown(final_state['markdown_output'])
                    
                    with tab2:
                        st.code(final_state['markdown_output'], language="markdown")
                    
                    # Show collected artifacts details
                    with st.expander("🔍 View Detailed Artifact Information"):
                        
                        # Stories
                        if final_state['stories']:
                            st.markdown("### 📚 User Stories")
                            for idx, story in enumerate(final_state['stories'], 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{idx}. {story['id']}**: {story['metadata'].get('title', 'N/A')}  
                                    - **Relevance Score:** {story.get('score', 0):.3f} (lower = more relevant)  
                                    - **Priority:** {story['metadata'].get('priority', 'N/A')}  
                                    - **Status:** {story['metadata'].get('status', 'N/A')}  
                                    - **Story Points:** {story['metadata'].get('story_points', 'N/A')}  
                                    - **Epic:** {story['metadata'].get('epic', 'N/A')}
                                    """)
                            st.divider()
                        
                        # Documentation
                        if final_state['documentation']:
                            st.markdown("### 📖 Documentation")
                            for idx, doc in enumerate(final_state['documentation'], 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{idx}. {doc['id']}**: {doc['metadata'].get('title', 'N/A')}  
                                    - **Relevance Score:** {doc.get('score', 0):.3f}  
                                    - **Type:** {doc['metadata'].get('doc_type', 'N/A')}  
                                    - **Source:** Confluence
                                    """)
                            st.divider()
                        
                        # Test Cases
                        if final_state['test_cases']:
                            st.markdown("### 🧪 Test Cases")
                            for idx, test in enumerate(final_state['test_cases'], 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{idx}. {test['id']}**: {test['metadata'].get('title', 'N/A')}  
                                    - **Objective:** {test['metadata'].get('objective', 'N/A')[:80]}...  
                                    - **Priority:** {test['metadata'].get('priority', 'N/A')}  
                                    - **Test Type:** {test['metadata'].get('test_type', 'N/A')}  
                                    - **Automation:** {test['metadata'].get('automation_status', 'N/A')}
                                    """)
                        
                        # Show relationships
                        if final_state['story_test_map']:
                            st.divider()
                            st.markdown("### 🔗 Story-Test Relationships")
                            for story_id, test_ids in final_state['story_test_map'].items():
                                st.markdown(f"- **{story_id}** → {len(test_ids)} test cases: {', '.join(test_ids[:5])}{'...' if len(test_ids) > 5 else ''}")
                
                else:
                    st.markdown('<div class="warning-box">⚠️ <strong>No training module was generated.</strong></div>', unsafe_allow_html=True)
                    
                    # Show what was collected
                    st.markdown(f"""
                    **Module:** {module_name}  
                    **Stories Found:** {len(final_state['stories'])}  
                    **Documentation Found:** {len(final_state['documentation'])}  
                    **Test Cases Found:** {len(final_state['test_cases'])}  
                    **Iterations Used:** {final_state['iteration']}/{final_state['max_iterations']}
                    """)
                    
                    if final_state.get('error_message'):
                        st.error(f"**Error:** {final_state['error_message']}")
                    
                    st.info("""
                    **Possible reasons:**
                    - No data indexed for this module in the knowledge base
                    - Module name not matching indexed data
                    - Try a different module name or check your indexed data
                    
                    **Available indexed modules:** Check your `test_data/` folder
                    """)
            
            except Exception as e:
                st.error(f"❌ **Error generating training module:**")
                st.exception(e)
                
                with st.expander("🐛 Full Debug Information"):
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# SIDEBAR INFO
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About")
    
    st.markdown("""
    This AI agent automatically generates comprehensive training modules by:
    
    1. 🔍 **Searching** JIRA user stories (semantic search)
    2. 📚 **Gathering** Confluence documentation
    3. 🧪 **Collecting** Zephyr test cases
    4. 🔗 **Mapping** relationships between artifacts
    5. 📝 **Generating** structured training content
    
    ---
    
    ### 🎯 How It Works
    
    The agent uses **RAG (Retrieval Augmented Generation)** with:
    
    - **Qdrant** vector database for semantic search
    - **Azure OpenAI GPT-4** for intelligent decisions
    - **LangGraph** for workflow orchestration
    - **Embeddings** for relevance scoring
    
    ---
    
    ### 💡 Tips for Best Results
    
    - Be specific about the module name
    - Use natural language in your query
    - The agent searches your indexed data automatically
    - Works with **any module** in your knowledge base
    
    ---
    
    ### 📊 Data Sources
    
    The agent searches across:
    - **JIRA** - User stories and requirements
    - **Confluence** - Technical documentation
    - **Zephyr** - Test cases and procedures
    
    ---
    
    ### 🔧 Configuration
    
    - **Max Iterations:** 8
    - **Search Top-K:** 10
    - **Embedding Model:** all-MiniLM-L6-v2
    - **LLM:** Azure OpenAI GPT-4.1-mini
    
    """)
    
    st.divider()
    
    st.caption("© 2024 TASConnect Training Generator  ")
    st.caption("Powered by Azure OpenAI + LangGraph + Qdrant")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>💡 <strong>Tip:</strong> The agent dynamically searches your knowledge base - it works with any module!</p>
    <p>🔍 Relevance scores: Lower = More similar (0.0 = perfect match)</p>
</div>
""", unsafe_allow_html=True)