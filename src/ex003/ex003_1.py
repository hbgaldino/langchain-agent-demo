from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
import operator

# 1 - Definir estado
class State(TypedDict):
    nodes_path: Annotated[list[str], operator.add]


# 2 - Definir os nodes

def node_a(state: State) -> State:
    
    output_state: State = {'nodes_path': ["A"]}
    print("> node_a", f"{state=}", f"{output_state=}")
    return output_state

def node_b(state: State) -> State:
    
    output_state: State = {'nodes_path': ["B"]}
    print("> node_b", f"{state=}", f"{output_state=}")
    return output_state

# Definir o builder do Grapho

builder = StateGraph(State)

builder.add_node("A", node_a)
builder.add_node("B", node_b)

builder.add_edge("__start__", "A")
builder.add_edge("A", "B")
builder.add_edge("B", "__end__")

## compilar

graph = builder.compile()

## pegar resultado

response = graph.invoke({"nodes_path": []})

print(response.state)