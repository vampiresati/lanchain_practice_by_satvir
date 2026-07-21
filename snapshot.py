from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage

agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[],
    system_prompt="You are a helpful assistant.",
)

stream = agent.stream_events(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain the latest AI trends."
            }
        ]
    },
    version="v3",
)

for snapshot in stream.values:
    latest_message = snapshot["messages"][-1]

    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")

    elif latest_message.tool_calls:
        print(
            f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}"
        )