from pydantic import ValidationError
from rich import print
from rich.markdown import Markdown
from typing import Annotated, Sequence, TypedDict
from langchain.tools import BaseTool, tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result.
    
    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The result of multiplying a and b.
    """
    return a * b

llm = init_chat_model("ollama:qwen2")
tools: list[BaseTool] = [multiply]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

system_messages = SystemMessage(
        "You are a helpful assistant. You have access to tools." \
        "When the user asks for something, first look if you have" \
        "a tool that solves that problem"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: AgentState) -> AgentState:
    print(f"call llm {state['messages'][-1]})")
    llm_result = llm_with_tools.invoke(state["messages"])
    return {"messages": [llm_result]}

def call_tools(state: AgentState) -> AgentState:
    print(f"call tools {state["messages"][-1]}")
    messages = state["messages"]
    llm_response = messages[-1]
    if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls"):
        call = llm_response.tool_calls[-1]
        name, args, id_ = call["name"], call["args"], call["id"]        

        try:
            content = tools_by_name[name].invoke(args)
            status = "success"
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as error:
            content = f"Please, fix your mistakes: {error}"
            status = "error"

        return {"messages": ToolMessage(content=content, tool_call_id=id_, status=status)}

def check_tool(state: AgentState) -> AgentState:
    messages = state["messages"]
    print(f"check tool {messages[-1]}")
    llm_response = messages[-1]
    if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls"):
        return 'call_tools'
    return END

builder = StateGraph(
    AgentState,
    context_schema=None,
    input_schema=None,
    output_schema=AgentState
)

builder.add_node("call_llm", call_llm)
builder.add_node("call_tools", call_tools)

builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", check_tool, {'call_tools', END})
builder.add_edge('call_tools', 'call_llm')

checkpointer = InMemorySaver()
graph = builder.compile() ## sem checkpointer, pois não queremos salvar o estado nesse exemplo
config = RunnableConfig(configurable={"thread_id": 1})

print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        print(f"---")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        human_message = HumanMessage(user_input)

        result = graph.invoke({"messages": [system_messages, human_message]}, config=config)

        print(Markdown(result["messages"][-1].content))
        print(Markdown(f"---"))