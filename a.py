import os
import getpass
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, MessagesState


# Set the Groq API Key
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Load model
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.environ["GROQ_API_KEY"]
)

# Ask something basic to test (optional)
response = llm.invoke("What's the future of AI?")
print(response.content)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b


# Bind tools to LLM
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# Node 1: LLM decides what to do
def llm_call(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ] + state["messages"]
            )
        ]
    }

# Node 2: Perform the tool call
def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Edge condition: decide whether to continue or stop
def should_continue(state: MessagesState) -> Literal["Action", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END


# Build the agent graph
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

checkpointer = MemorySaver()

# Compile the agent
agent = agent_builder.compile(checkpointer=checkpointer)
# Specify a thread
config = {"configurable": {"thread_id": "1"}}
# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = agent.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Multiply that by 2.")]
messages = agent.invoke({"messages": messages}, config)

# Print the result
for m in messages["messages"]:
    m.pretty_print()
