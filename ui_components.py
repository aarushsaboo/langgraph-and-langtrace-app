# UI components for the research assistant
import traceback

def setup_streamlit_ui(model_name):
    """Set up the Streamlit UI components."""
    import streamlit as st
    
    st.set_page_config(page_title="Research Assistant with Gemini", layout="wide")
    
    st.title("Research Assistant with Langraph & Gemini")
    st.write("Ask any question, and I'll research it for you using Google's Gemini model!")
    
    # Model info
    st.sidebar.header("Model Configuration")
    st.sidebar.info(f"Using model: {model_name}")
    
    # Get user input
    query = st.text_input("Your question:")
    
    return st, query

def setup_tracing(st):
    """Set up tracing configuration in the Streamlit UI."""
    enable_tracing = st.sidebar.checkbox("Enable Langtrace visualization", value=False)
    
    if enable_tracing:
        langtrace_url = st.sidebar.text_input("Langtrace server URL", "http://localhost:8000")
        langtrace_key = st.sidebar.text_input("Langtrace API key", "your-api-key", type="password")
        
        if langtrace_url and langtrace_key:
            try:
                import langtrace
                langtrace.init(api_key=langtrace_key, host=langtrace_url)
                return True
            except Exception as e:
                st.sidebar.error(f"Failed to initialize Langtrace: {e}")
    
    return False

def display_results(st, result, processing_time):
    """Display the research results in the Streamlit UI."""
    # Create columns for results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display final answer
        st.subheader("Answer")
        st.write(result["final_answer"])
        st.caption(f"Processing time: {processing_time:.2f} seconds")
    
    with col2:
        # Display the research process
        st.subheader("Research Questions")
        for q in result["questions"]:
            st.write(f"- {q}")
    
    # Display detailed research results
    st.subheader("Detailed Research")
    tabs = st.tabs([f"Q{i+1}" for i in range(len(result["questions"]))])
    
    for i, (question, tab) in enumerate(zip(result["questions"], tabs)):
        with tab:
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Research:** {result['research_results'][question]}")

def display_error(st, error):
    """Display error information in the Streamlit UI."""
    st.error(f"An error occurred: {str(error)}")
    st.error(f"Error details: {type(error).__name__}")
    st.code(traceback.format_exc())