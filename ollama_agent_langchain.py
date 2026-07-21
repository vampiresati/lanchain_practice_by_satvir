from langchain.agents import create_agent
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."
agent = create_agent(model="ollama:qwen3:4b",tools=[get_weather],)
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in sirhind?",
            }
        ]
    }
)
print(response["messages"][-1].content)