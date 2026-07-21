from langchain.agents import create_agent
def get_time() -> str:
    """Return the current time."""
    return "10:30 AM"
agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[get_time],
)
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the time?",
            }
        ]
    }
)
print(response["messages"][-1].content)