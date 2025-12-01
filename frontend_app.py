"""
Simple Streamlit frontend for rare disease report generation.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_wrapper import run_pipeline
import time


st.set_page_config(
    page_title="Rare Disease Report Generator",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Rare Disease Report Generator")
st.markdown("Generate comprehensive Wikipedia-style reports for rare diseases using AI-powered research.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info("Make sure your environment variables are set:\n- GCP_PROJECT_ID\n- GCP_API_KEY (optional)\n- SERPER_API_KEY\n- VERTEX_MODEL (optional)")
    
    output_dir = st.text_input(
        "Output Directory",
        value="src/output",
        help="Directory where generated reports will be saved"
    )

# Main input
disease_name = st.text_input(
    "Enter Disease Name",
    placeholder="e.g., Duchenne Muscular Dystrophy",
    help="Enter the name of the rare disease you want to generate a report for"
)

# Status container
status_container = st.empty()

# Results container
results_container = st.empty()

# Run button
if st.button("Generate Report", type="primary", use_container_width=True):
    if not disease_name:
        st.error("Please enter a disease name")
        st.stop()
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_messages = []
    
    def progress_callback(message):
        status_messages.append(message)
        status_container.text_area(
            "Status",
            value="\n".join(status_messages),
            height=200,
            disabled=True
        )
        # Update progress (rough estimate)
        if "Wikipedia" in message:
            progress_bar.progress(20)
        elif "RAG" in message:
            progress_bar.progress(50)
        elif "plan" in message.lower():
            progress_bar.progress(70)
        elif "report" in message.lower():
            progress_bar.progress(90)
        elif "✅" in message:
            progress_bar.progress(100)
    
    try:
        with st.spinner("Running pipeline... This may take several minutes."):
            results = run_pipeline(
                disease_name=disease_name,
                output_dir=Path(output_dir),
                progress_callback=progress_callback
            )
        
        progress_bar.progress(100)
        st.success("✅ Report generated successfully!")
        
        # Display results
        with results_container.container():
            st.header("Generated Report")
            
            # Show report
            st.markdown("### Full Report")
            st.markdown(results["report"])
            
            # Show metadata
            with st.expander("View Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Guideline Sections:**", len(results["guideline"]["sections"]))
                    st.write("**Articles Found:**", results.get("metadata", {}).get("articles_found", "N/A"))
                with col2:
                    st.write("**RAG Sources:**", len(results["rag_response"].get("cited_documents", [])))
            
            # Download button
            st.download_button(
                label="Download Report (Markdown)",
                data=results["report"],
                file_name=f"report_{disease_name.replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
            # Show output paths
            st.info(f"Files saved to:\n- {results['output_paths']['report']}\n- {results['output_paths']['guideline_md']}")
    
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.exception(e)
        progress_bar.empty()

# Instructions
with st.expander("How to Use"):
    st.markdown("""
    1. **Enter Disease Name**: Type the name of the rare disease you want to research
    2. **Click Generate Report**: The pipeline will:
       - Search Wikipedia for relevant articles
       - Generate a research guideline
       - Gather evidence using RAG (Retrieval-Augmented Generation)
       - Create a comprehensive research report
    3. **View Results**: The generated report will appear below
    4. **Download**: Click the download button to save the report as Markdown
    
    **Note**: This process can take 5-10 minutes depending on the disease complexity.
    """)

