import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict, List, Union, Literal
from datetime import datetime
date = datetime.now().strftime(("%B %d, %Y"))
# Set page configuration
st.set_page_config(page_title="Multi-Agent System", layout="wide")
st.title("🧠 Multi-Agent System with DeepSeek and Gemini")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for API key configuration
with st.sidebar:
    st.header("Configuration")
    deepseek_api_key = "DeepseekAPI"
    gemini_api_key = "GeminiAPI"
    
    st.divider()
    st.markdown("""
    ### About this app
    
    This multi-agent system uses:
    - LangGraph for orchestrating multiple agents
    - DuckDuckGo search tool for retrieving information
    - DeepSeek and Gemini models as language models
    - Streamlit for the user interface
    """)

# Initialize the search tool
@st.cache_resource
def get_search_tool():
    return DuckDuckGoSearchRun()

# Set up the language models
def get_llms():
    llms = {}
    errors = []
    
    if deepseek_api_key:
        try:
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
            llms["deepseek"] = ChatDeepSeek(
                model="deepseek-chat",
                temperature=0.7
            )
        except Exception as e:
            errors.append(f"Error initializing DeepSeek: {str(e)}")
    else:
        errors.append("Please provide a valid DeepSeek API key.")
    
    if gemini_api_key:
        try:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            llms["gemini"] = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7
            )
        except Exception as e:
            errors.append(f"Error initializing Gemini: {str(e)}")
    else:
        errors.append("Please provide a valid Gemini API key.")
    
    return llms, errors

# Define the state schema
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    research_findings: str
    analysis: str
    next: Literal["researcher", "analyst", "writer", "end"]

# Create agent nodes
def create_researcher_node(llm, search_tool):
    def researcher(state: AgentState):
        messages = state["messages"]
        human_message = messages[-1].content
        
        # Add current date and emphasize recency
        current_date = date
        researcher_prompt = f"""You are a Research Specialist. Your goal is to find accurate and UP-TO-DATE information as of {current_date}.
        
        The user query is: {human_message}
        
        Use the search tool to gather CURRENT information. Focus on the most recent developments, news, and data.
        Explicitly search for content from the past year. Add terms like '2024', '2025', 'recent', 'latest' to your searches.
        
        Be thorough in your research and provide all important details discovered, prioritizing the newest information."""
        
        # Modify the search query to emphasize recency
        search_query = f"{human_message} latest 2025 recent developments"
        search_results = search_tool.run(search_query)
        
        # Generate response
        response = llm.invoke(f"{researcher_prompt}\n\nSearch results: {search_results}\n\nProvide your comprehensive research findings with focus on the most current information:")
        
        return {
            "messages": messages,
            "research_findings": response.content,
            "analysis": state.get("analysis", ""),
            "next": "analyst"
        }
    
    return researcher

def create_analyst_node(llm):
    def analyst(state: AgentState):
        """Analyst agent that analyzes the information gathered by the researcher."""
        messages = state["messages"]
        research_findings = state["research_findings"]
        original_query = messages[-1].content
        
        # Create analyst prompt
        analyst_prompt = f"""You are a Data Analyst. Your goal is to analyze information and extract meaningful insights.
        
        The original user query was: {original_query}
        
        The researcher has provided the following information:
        {research_findings}
        
        Analyze this information to:
        1. Identify key patterns and insights
        2. Evaluate the reliability of the information
        3. Highlight the most important facts
        4. Note any contradictions or gaps
        
        Present your analysis in a clear and structured format."""
        
        # Generate analysis
        response = llm.invoke(analyst_prompt)
        
        # Update state
        return {
            "messages": messages,
            "research_findings": research_findings,
            "analysis": response.content,
            "next": "writer"
        }
    
    return analyst

def create_writer_node(llm):
    def writer(state: AgentState):
        """Writer agent that creates the final response based on research and analysis."""
        messages = state["messages"]
        analysis = state["analysis"]
        research_findings = state["research_findings"]
        original_query = messages[-1].content
        
        # Create writer prompt
        writer_prompt = f"""You are a Content Writer. Your goal is to create clear, concise, and engaging content.
        
        The original user query was: {original_query}
        
        You have the following information:
        1. Research findings: {research_findings}
        2. Analysis: {analysis}
        
        Create a comprehensive, well-structured response that:
        1. Directly answers the user's query
        2. Presents information in a logical flow
        3. Uses clear and engaging language
        4. Includes all relevant facts and insights
        5. Is accurate and balanced
        6. Provide the links for mentioned resources
        Your response should be both informative and easy to understand."""
        
        # Generate final response
        response = llm.invoke(writer_prompt)
        
        # Update state with the final response and set next to end
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "research_findings": research_findings,
            "analysis": analysis,
            "next": "end"
        }
    
    return writer

# Build and run the LangGraph
def build_graph(llms):
    # Get the search tool
    search_tool = get_search_tool()
    
    # Create the agent nodes
    researcher = create_researcher_node(llms["deepseek"], search_tool)
    analyst = create_analyst_node(llms["gemini"])
    writer = create_writer_node(llms["deepseek"])
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)
    
    # Define the conditional edges
    workflow.add_conditional_edges(
        "researcher",
        lambda state: state["next"],
        {"analyst": "analyst", "writer": "writer", "end": END}
    )
    
    workflow.add_conditional_edges(
        "analyst",
        lambda state: state["next"],
        {"writer": "writer", "end": END}
    )
    
    workflow.add_conditional_edges(
        "writer",
        lambda state: state["next"],
        {"end": END}
    )
    
    # Set the entry point
    workflow.set_entry_point("researcher")
    
    # Compile the graph
    return workflow.compile()

def run_multi_agent_system(query):
    # Get the language models
    llms, errors = get_llms()
    
    if errors or len(llms) < 2:
        return f"Error: {'; '.join(errors)}"
    
    # Build the graph
    graph = build_graph(llms)
    
    # Run the graph
    with st.spinner("The AI agents are working on your request..."):
        try:
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "research_findings": "",
                "analysis": "",
                "next": "researcher"
            }
            
            result = graph.invoke(initial_state)
            
            # Return the final message from the writer
            return result["messages"][-1].content
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            return f"Error processing your request: {str(e)}"

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask something...")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get and display assistant response
    if deepseek_api_key and gemini_api_key:
        response = run_multi_agent_system(user_input)
        if response:
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        with st.chat_message("assistant"):
            st.markdown("Please provide both DeepSeek and Gemini API keys in the sidebar to continue.")
            st.session_state.messages.append({"role": "assistant", "content": "Please provide both DeepSeek and Gemini API keys in the sidebar to continue."})

# Add a reset button
if st.button("Reset Conversation"):
    st.session_state.messages = []
    st.rerun()