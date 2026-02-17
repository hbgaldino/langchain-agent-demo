from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig
from rich import print
from rich.markdown import Markdown
from langchain.chat_models import init_chat_model

llm = init_chat_model("ollama:qwen2")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages ]

def call_llm(state: AgentState) -> AgentState:
    llm_result = llm.invoke(state["messages"])
    return {"messages": [llm_result]}


builder = StateGraph(
    AgentState,
    context_schema=None,
    input_schema=None,
    output_schema=AgentState
)

builder.add_node("call_llm", call_llm)

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)


checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config= RunnableConfig(configurable={"thread_id": 1})

if __name__ == "__main__":    
    while True:
        user_input = input("You: ")
        print(Markdown(f"---"))

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        human_message = HumanMessage(user_input)

        result = graph.invoke({"messages": human_message}, config=config)

        print(Markdown(result["messages"][-1].content))
        print(Markdown(f"---"))