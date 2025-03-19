# Gemini Research Assistant with Langraph
# pip install langchain-core langchain langgraph langchain-google-genai streamlit

import os
import time
from typing import List, Dict, Any, TypedDict, Annotated
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import langgraph as lg
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import FieldSpec  # Import FieldSpec for field definition

# Configure API key for Google Generative AI
GOOGLE_API_KEY = "AIzaSyCiNMgc2NAV9scAmqwvaE170ZlwdXA-1Xk"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure LLM model
LLM_MODEL = "gemini-1.5-flash"  # Gemini model name
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# Try importing langtrace for visualization (optional)
try:
    import langtrace 
    from langtrace import trace
    from langtrace.integrations import langgraph as langtrace_langgraph
    LANGTRACE_AVAILABLE = True
except ImportError:
    LANGTRACE_AVAILABLE = False

# Define the state structure
class AgentState(TypedDict):
    messages: List[Any]  # Simplified without Annotated
    questions: List[str]
    research_results: Dict[str, str]
    final_answer: str

# Create prompts for each step
question_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Break down the user's query into 2-3 specific research questions that would help provide a comprehensive answer. Return only a Python list of strings.")
])

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Research the following question and provide a concise answer based on your knowledge."),
    ("human", "{question}")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Based on the following research results, provide a comprehensive answer to the user's original query."),
    ("system", "Research results: {research_results}")
])

# Function to break down the query into research questions
def break_down_query(state: AgentState) -> AgentState:
    """Break down the user query into specific research questions."""
    response = question_prompt.invoke({"messages": state["messages"]})
    result = llm.invoke(response)
    
    # Parse the list of questions from the response
    try:
        # This assumes the LLM returns a valid Python list representation
        import ast
        # Clean up the response content to make it more parseable
        content = result.content.strip()
        # If the content is wrapped in backticks, remove them
        if content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        if content.startswith("python"):
            content = content[6:].strip()
        questions = ast.literal_eval(content)
    except Exception as e:
        # Fallback if parsing fails - extract questions using a simple approach
        content = result.content
        # Look for list-like patterns in the response
        import re
        questions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)', content)
        if not questions:
            # Just split by newlines if no other pattern works
            questions = [line.strip() for line in content.split('\n') if line.strip()]
        
    return {"questions": questions}

# Function to research each question
def conduct_research(state: AgentState) -> AgentState:
    """Research each question and gather results."""
    research_results = {}
    
    for question in state["questions"]:
        prompt = research_prompt.invoke({"question": question})
        result = llm.invoke(prompt)
        research_results[question] = result.content
        
    return {"research_results": research_results}

# Function to synthesize the final answer
def synthesize_answer(state: AgentState) -> AgentState:
    """Synthesize research results into a comprehensive answer."""
    response = synthesis_prompt.invoke({
        "messages": state["messages"],
        "research_results": state["research_results"]
    })
    result = llm.invoke(response)
    
    final_message = AIMessage(content=result.content)
    
    return {
        "messages": state["messages"] + [final_message],
        "final_answer": result.content
    }

# Build the graph
def build_graph():
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("break_down_query", break_down_query)
    workflow.add_node("conduct_research", conduct_research)
    workflow.add_node("synthesize_answer", synthesize_answer)
    
    # Define the edges
    workflow.add_edge("break_down_query", "conduct_research")
    workflow.add_edge("conduct_research", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)
    
    # Set the entry point
    workflow.set_entry_point("break_down_query")
    
    # Compile the graph
    return workflow.compile()

# Streamlit web interface
def main():
    st.set_page_config(page_title="Research Assistant with Gemini", layout="wide")
    
    st.title("Research Assistant with Langraph & Gemini")
    st.write("Ask any question, and I'll research it for you using Google's Gemini model!")
    
    # Model info
    st.sidebar.header("Model Configuration")
    st.sidebar.info(f"Using model: {LLM_MODEL}")
    
    # Get user input
    query = st.text_input("Your question:")

    # Add trace toggle if langtrace is available
    enable_tracing = False
    if LANGTRACE_AVAILABLE:
        enable_tracing = st.sidebar.checkbox("Enable Langtrace visualization", value=False)
        
        if enable_tracing:
            langtrace_url = st.sidebar.text_input("Langtrace server URL", "http://localhost:8000")
            langtrace_key = st.sidebar.text_input("Langtrace API key", "your-api-key", type="password")
            
            if langtrace_url and langtrace_key:
                try:
                    langtrace.init(api_key=langtrace_key, host=langtrace_url)
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize Langtrace: {e}")
                    enable_tracing = False

    if st.button("Research"):
        if query:
            start_time = time.time()
            with st.spinner("Researching your question..."):
                try:
                    # Create graph
                    graph = build_graph()
                    
                    # Add trace monitoring if enabled
                    if enable_tracing and LANGTRACE_AVAILABLE:
                        with trace("gemini_research") as current_trace:
                            monitored_graph = langtrace_langgraph.trace_compiled_graph(
                                graph, 
                                current_trace, 
                                name="gemini_research"
                            )
                            
                            # Initialize state
                            initial_state = {
                                "messages": [HumanMessage(content=query)],
                                "questions": [],
                                "research_results": {},
                                "final_answer": ""
                            }
                            
                            # Run the graph
                            result = monitored_graph.invoke(initial_state)
                            st.sidebar.success(f"Trace URL: {current_trace.url}")
                    else:
                        # Initialize state without tracing
                        initial_state = {
                            "messages": [HumanMessage(content=query)],
                            "questions": [],
                            "research_results": {},
                            "final_answer": ""
                        }
                        
                        # Run the graph
                        result = graph.invoke(initial_state)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
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
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a question.")

# Command line interface for testing
def cli():
    print("\n===== Gemini Research Assistant =====")
    query = input("\nEnter your question: ")
    
    print("\nResearching your question...")
    
    # Create the graph
    graph = build_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "questions": [],
        "research_results": {},
        "final_answer": ""
    }
    
    # Run the graph
    try:
        result = graph.invoke(initial_state)
        
        print("\n----- Research Questions -----")
        for q in result["questions"]:
            print(f"- {q}")
            
        print("\n----- Final Answer -----")
        print(result["final_answer"])
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli()
    else:
        main()

# To run the web app: streamlit run app.py
# To run the CLI: python app.py --cli