# Research engine components - Optimized for fewer API calls
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import re

# Define the state structure for the research process
class AgentState(TypedDict):
    messages: List[Any]
    questions: List[str]
    research_results: Dict[str, str]
    final_answer: str

# Initialize the LLM with quota-friendly settings
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using smaller model
        temperature=0.2,         # Lower temperature for more focused responses
        max_output_tokens=1024,  # Limit token usage
    )

# Efficient single-call approach for research
def all_in_one_research(state: AgentState) -> AgentState:
    """Perform the entire research process in a single API call to save quota"""
    llm = get_llm()
    
    # Get the user's original query
    user_query = state["messages"][0].content
    
    # Create a comprehensive prompt that handles all steps
    system_message = f"""
    You are a research assistant analyzing this question: "{user_query}"
    
    Follow these steps:
    1. Break down the user's query into 2-3 specific research questions.
    2. For each question, provide a concise answer based on your knowledge.
    3. Finally, synthesize a comprehensive answer to the original query.
    
    Format your response exactly as follows:
    RESEARCH QUESTIONS:
    - Question 1
    - Question 2
    
    RESEARCH FINDINGS:
    Question 1: Your answer to question 1.
    Question 2: Your answer to question 2.
    
    FINAL ANSWER:
    Your synthesized answer to the original question.
    """
    
    # Fixed: Creating proper messages for Gemini
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_query)
    ]
    
    # Invoke the LLM with the properly formatted messages
    result = llm.invoke(messages)
    
    # Parse the response into our expected format
    content = result.content
    
    # Extract questions
    questions = []
    research_results = {}
    final_answer = ""
    
    # Parse the questions
    if "RESEARCH QUESTIONS:" in content:
        questions_section = content.split("RESEARCH QUESTIONS:")[1].split("RESEARCH FINDINGS:")[0].strip()
        questions = [q.strip()[2:] for q in questions_section.split("\n") if q.strip().startswith("- ")]
    
    # Parse research findings
    if "RESEARCH FINDINGS:" in content:
        findings_section = content.split("RESEARCH FINDINGS:")[1].split("FINAL ANSWER:")[0].strip()
        # Match each question and its answer
        for question in questions:
            pattern = re.escape(question) + r":(.*?)(?=" + "|".join([re.escape(q) + ":" for q in questions if q != question]) + "|$)"
            matches = re.search(pattern, findings_section, re.DOTALL)
            if matches:
                research_results[question] = matches.group(1).strip()
            else:
                # Fallback if regex fails
                research_results[question] = "No specific findings available."
    
    # Parse final answer
    if "FINAL ANSWER:" in content:
        final_answer = content.split("FINAL ANSWER:")[1].strip()
    
    # Create final message
    final_message = AIMessage(content=final_answer)
    
    return {
        "messages": state["messages"] + [final_message],
        "questions": questions,
        "research_results": research_results,
        "final_answer": final_answer
    }

# Build the research graph with a single node to reduce API calls
def create_research_graph():
    """Create and compile a simplified research workflow graph using one node"""
    workflow = StateGraph(AgentState)
    
    # Add a single node that does all research steps in one call
    workflow.add_node("research", all_in_one_research)
    
    # Direct connection from research to end
    workflow.add_edge("research", END)
    
    # Set the entry point
    workflow.set_entry_point("research")
    
    # Compile the graph
    return workflow.compile()

# Initialize the research state
def initialize_research_state(query):
    """Create the initial state for the research process"""
    return {
        "messages": [HumanMessage(content=query)],
        "questions": [],
        "research_results": {},
        "final_answer": ""
    }