from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from rich import print

llm = init_chat_model('ollama:qwen2')

system_message = SystemMessage("Você é um assistente útil e amigável.")

human_message = HumanMessage("Olá meu nome é Higor")

messages = [system_message, human_message]

response = llm.invoke(messages)

print(f"{'AI':-^80}")
print(response.content)

messages.append(response)

while True:
    user_input = input("Você: ")
    human_message = HumanMessage(user_input)

    if user_input.lower() in ['sair', 'exit', 'quit']:        
        break

    messages.append(human_message)

    response = llm.invoke(messages)

    print(f"{'AI':-^80}")
    print(response.content)
    print()

    messages.append(response)