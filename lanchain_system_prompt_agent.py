from langchain.agents import create_agent
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

tools = [add, multiply, get_weather]
agent = create_agent(
    model="ollama:qwen3:4b",
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Be concise and accurate. "
        "Use tools whenever they help answer the user's question."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 125 * 87?"
            }
        ]
    }
)

print(response["messages"][-1].content)