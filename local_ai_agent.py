import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# load_dotenv()

llm = init_chat_model(
    "qwen3:4b",
    model_provider="ollama",
    temperature=0.5,
)

def list_directory(path: str=".") -> str:
    """List all files and folders in the given directory path. Default is current directory."""
    result = ""
    items = os.listdir()
    files = []
    folders = []

    for item in items:
        if os.path.isdir(item):
            folders.append(item)
        else:
            files.append(item)
    if folders:
        result = "Folders:\n " + result + "\n".join(folders) + "\n\n"
    if files:
        result = "Files:\n" + result + "\n".join(files)
    return result
#
# a=list_directory('/home/satvir/langchain_practice')
# print(a)
#
agent = create_agent(
    model=llm,
    tools=[list_directory]
)
#
user_input = "Give me the list of files in the dir"
messages = [{"role":"user", "content":user_input}]

chunks = list(agent.stream({"messages": messages}, stream_mode='updates'))
print('')
for chunk in chunks:
    for step, data in chunk.items():
        if step == 'model':
            msg = data['messages'][-1]
            print("MSG toOL CALL", msg.tool_calls)
            if msg.tool_calls:
                tool = msg.tool_calls[0]['name']
                message = f"🔧 Using tool: {tool}"
                print(message)
            else:
                ai_message = msg.content
        elif step == 'tools':
            result = data['messages'][-1].content
            result = result if len(result) < 50 else result[:50] + "..."
            print(f"🗒️ Tool result:\n {result}")

#
#
print(30*"-", "\n")
print(ai_message)