# Research engine components
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Define the state structure for the research process
class AgentState(TypedDict):
    messages: List[Any]
    questions: List[str]
    research_results: Dict[str, str]
    final_answer: str

# Initialize the LLM
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create the prompts
def create_prompts():
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

    return question_prompt, research_prompt, synthesis_prompt

# Graph node functions
def break_down_query(state: AgentState) -> AgentState:
    """Break down the user query into specific research questions."""
    llm = get_llm()
    question_prompt, _, _ = create_prompts()
    
    response = question_prompt.invoke({"messages": state["messages"]})
    result = llm.invoke(response)
    
    # Parse the list of questions from the response
    try:
        # Try to parse as a Python list
        import ast
        content = result.content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        if content.startswith("python"):
            content = content[6:].strip()
        questions = ast.literal_eval(content)
    except Exception:
        # Fallback if parsing fails
        import re
        questions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)', result.content)
        if not questions:
            questions = [line.strip() for line in result.content.split('\n') if line.strip()]
        
    return {"questions": questions}

def conduct_research(state: AgentState) -> AgentState:
    """Research each question and gather results."""
    llm = get_llm()
    _, research_prompt, _ = create_prompts()
    
    research_results = {}
    
    for question in state["questions"]:
        prompt = research_prompt.invoke({"question": question})
        result = llm.invoke(prompt)
        research_results[question] = result.content
        
    return {"research_results": research_results}

def synthesize_answer(state: AgentState) -> AgentState:
    """Synthesize research results into a comprehensive answer."""
    llm = get_llm()
    _, _, synthesis_prompt = create_prompts()
    
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

# Build the research graph
def create_research_graph():
    """Create and compile the research workflow graph."""
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

# Initialize the research state
def initialize_research_state(query):
    """Create the initial state for the research process."""
    return {
        "messages": [HumanMessage(content=query)],
        "questions": [],
        "research_results": {},
        "final_answer": ""
    }