# Gemini Research Assistant with Langraph - Optimized for fewer API calls
import os
import time
import sys
from langchain_core.messages import HumanMessage

# Import components
from ui_components import setup_streamlit_ui, display_results, display_error, setup_tracing
from research_engine import create_research_graph, initialize_research_state

# Configure API key for Google Generative AI
GOOGLE_API_KEY = "AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA"  # Replace with your key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure LLM model - using smaller model to reduce quota usage
LLM_MODEL = "gemini-1.0-pro"

# Check if langtrace is available
try:
    import langtrace
    from langtrace import trace
    from langtrace.integrations import langgraph as langtrace_langgraph
    LANGTRACE_AVAILABLE = True
except ImportError:
    LANGTRACE_AVAILABLE = False

def process_research_query(query, enable_tracing=False):
    """Main function to process a research query through the graph"""
    start_time = time.time()
    
    try:
        # Step 1: Create the research graph
        graph = create_research_graph()
        
        # Step 2: Initialize the research state with the query
        initial_state = initialize_research_state(query)
        
        # Step 3: Execute the graph with or without tracing
        if enable_tracing and LANGTRACE_AVAILABLE:
            result, trace_url = run_with_tracing(graph, initial_state)
            return result, time.time() - start_time, None, trace_url
        else:
            result = graph.invoke(initial_state)
            return result, time.time() - start_time, None, None
    
    except Exception as e:
        return None, None, e, None

def run_with_tracing(graph, initial_state):
    """Execute the graph with tracing enabled"""
    with trace("gemini_research") as current_trace:
        monitored_graph = langtrace_langgraph.trace_compiled_graph(
            graph, 
            current_trace, 
            name="gemini_research"
        )
        
        result = monitored_graph.invoke(initial_state)
        return result, current_trace.url

def run_web_interface():
    """Run the Streamlit web interface"""
    import streamlit as st
    
    # Step 1: Setup the UI
    st, query = setup_streamlit_ui(LLM_MODEL)
    
    # Step 2: Setup tracing if available
    enable_tracing = setup_tracing(st) if LANGTRACE_AVAILABLE else False
    
    # Step 3: Process the query when requested
    if st.button("Research") and query:
        with st.spinner("Researching your question..."):
            result, processing_time, error, trace_url = process_research_query(query, enable_tracing)
            
            if error:
                display_error(st, error)
            else:
                if trace_url and LANGTRACE_AVAILABLE:
                    st.sidebar.success(f"Trace URL: {trace_url}")
                display_results(st, result, processing_time)

def run_cli():
    """Run the command line interface"""
    print("\n===== Gemini Research Assistant =====")
    query = input("\nEnter your question: ")
    
    print("\nResearching your question...")
    
    result, processing_time, error, _ = process_research_query(query)
    
    if error:
        print(f"An error occurred: {error}")
    else:
        print("\n----- Research Questions -----")
        for q in result["questions"]:
            print(f"- {q}")
            
        print("\n----- Final Answer -----")
        print(result["final_answer"])
        
        print(f"\nCompleted in {processing_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_web_interface()