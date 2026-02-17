from langchain.tools import tool, BaseTool
from langchain.chat_models import init_chat_model
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage

from pydantic_core import ValidationError
from rich import print

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


if __name__ == "__main__":
    llm = init_chat_model('ollama:qwen2')  

    system_messages = SystemMessage(
        "You are a helpful assistant. You have access to tools." \
        "When the user asks for something, first look if you have" \
        "a tool that solves that problem"
    )

    humam_message = HumanMessage(
        "Oi sou o Higor. Pode me falar quanto é 2 vezes 4?"
    )

    messages: list[BaseMessage] = [system_messages, humam_message]    

    # result = llm.invoke(messages)
    # print(result)

    tools: list[BaseTool] = [multiply]
    tools_by_name = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(tools)
    llm_response = llm_with_tools.invoke(messages)
    
    messages.append(llm_response)

    if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls"):
        call = llm_response.tool_calls[-1]
        name, args, id_ = call["name"], call["args"], call["id"]
        
        try:
            content = tools_by_name[name].invoke(args)
            status = "success"
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as error:
            content = f"Please, fix your mistakes: {error}"
            status = "error"

        tool_message = ToolMessage(content=content, tool_call_id=id_, status=status)
        messages.append(tool_message)
    
    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)
    print(messages)

        