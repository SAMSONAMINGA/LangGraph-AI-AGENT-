# Load environment variables (for OpenAI and Tavily keys)
from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ====================
# 1. Define Tools
# ====================

# Web search tool using Tavily
search = TavilySearchResults(max_results=5)

@tool
def search_tool(query: str):
    """Search the web for current information using Tavily."""
    return search.invoke(query)

# Clothing recommendation tool
@tool
def recommend_clothing(weather: str) -> str:
    """
    Recommend clothing based on a weather description.
    
    Args:
        weather: Weather description (e.g., "Sunny, 75°F" or "Rainy and cold")
    """
    weather = weather.lower()
    if any(word in weather for word in ["snow", "freezing", "blizzard"]):
        return "Wear a heavy coat, scarf, gloves, hat, and winter boots."
    elif any(word in weather for word in ["rain", "shower", "drizzle", "wet"]):
        return "Bring a raincoat, umbrella, and waterproof shoes."
    elif any(word in weather for word in ["hot", "90", "95", "100"]) or "heat" in weather:
        return "Wear light clothing: T-shirt, shorts, hat, and sunscreen."
    elif any(word in weather for word in ["cold", "below 50", "chilly"]) or "40" in weather or "30" in weather:
        return "Wear a warm jacket, sweater, and layers."
    else:
        return "A light jacket or sweater should be sufficient."

# Local helper for fallback (the @tool-decorated function returns a tool object,
# so use this local helper when we need to call the logic directly).
def recommend_clothing_local(weather: str) -> str:
    weather = weather.lower()
    if any(word in weather for word in ["snow", "freezing", "blizzard"]):
        return "Wear a heavy coat, scarf, gloves, hat, and winter boots."
    elif any(word in weather for word in ["rain", "shower", "drizzle", "wet"]):
        return "Bring a raincoat, umbrella, and waterproof shoes."
    elif any(word in weather for word in ["hot", "90", "95", "100"]) or "heat" in weather:
        return "Wear light clothing: T-shirt, shorts, hat, and sunscreen."
    elif any(word in weather for word in ["cold", "below 50", "chilly"]) or "40" in weather or "30" in weather:
        return "Wear a warm jacket, sweater, and layers."
    else:
        return "A light jacket or sweater should be sufficient."
# Register tools
tools = [search_tool, recommend_clothing]
tools_by_name = {tool.name: tool for tool in tools}

# ====================
# 2. Set up the Model
# ====================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt that encourages step-by-step reasoning and tool use
system_prompt = """
You are a helpful AI assistant that thinks step-by-step and uses tools when necessary.

Guidelines:
- First, reason about what information you need to answer the question.
- If you need up-to-date or external information, use the available tools.
- After receiving tool results, incorporate them into your reasoning.
- Finally, provide a clear and complete answer to the user.
- Always show your thinking process.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="scratch_pad"),
])

# Bind tools to the model so it can call them
model_with_tools = chat_prompt | model.bind_tools(tools)

# ====================
# 3. Define Agent State
# ====================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ====================
# 4. Define Graph Nodes
# ====================

def call_model(state: AgentState):
    """Call the LLM with the current conversation history."""
    try:
        response = model_with_tools.invoke({"scratch_pad": state["messages"]})
        return {"messages": [response]}
    except Exception as e:
        # Fallback for offline testing or quota errors: return a mock assistant message.
        warn = f"(MOCK) LLM unavailable: {e}. Returning a mock response."

        # Simple heuristic mock: if last user message asks about weather, fabricate a small answer
        last_user = state["messages"][-1].content if state["messages"] else ""
        if "weather" in last_user.lower() and "zurich" in last_user.lower():
            weather_desc = "Cloudy, 48°F"
            clothing = recommend_clothing_local(weather_desc)
            content = f"{warn}\nMocked weather for Zurich: {weather_desc}. Recommendation: {clothing}"
        else:
            content = warn + "\n(Mock) I can't connect to the LLM right now. Please check API keys."
        # Create an AIMessage so LangGraph/langchain can coerce it properly.
        msg = AIMessage(content=content)
        # Ensure `.tool_calls` exists so downstream logic can read it.
        setattr(msg, "tool_calls", [])
        return {"messages": [msg]}

def tool_node(state: AgentState):
    """Execute all tool calls from the last message."""
    outputs = []
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ))
    return {"messages": outputs}

# Decide whether to continue or end
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# ====================
# 5. Build the Graph
# ====================

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# After using tools, always go back to the agent
workflow.add_edge("tools", "agent")

# Conditional routing from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Start at the agent node
workflow.set_entry_point("agent")

# Compile the graph
graph = workflow.compile()

# ====================
# 6. Helper to Run & Stream Output
# ====================

def run_agent(query: str):
    print(f"🤖 User: {query}")
    print("=" * 60)
    inputs = {"messages": [HumanMessage(content=query)]}
    
    for chunk in graph.stream(inputs, stream_mode="values"):
        message = chunk["messages"][-1]
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"\n{message}\n")

# ====================
# 7. Test the Agent
# ====================

if __name__ == "__main__":
    print("🚀 Starting ReAct Agent with LangGraph!")
    print("=" * 60)
    
    # Example query from the notebook
    run_agent("What's the weather like in Zurich right now, and what should I wear?")
    
    print("\n" + "=" * 60)
    print("✅ Agent test completed! Try your own queries by modifying run_agent()")