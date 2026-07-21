from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

# Create the agent
agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant.",
)

# Create a conversation ID
config = {
    "configurable": {
        "thread_id": str(uuid4())
    }
}

# First message
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Satvir."
            }
        ]
    },
    config=config,
)

print(result["messages"][-1].content)

# Second message (same thread_id)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    },
    config=config,
)

print(result["messages"][-1].content)