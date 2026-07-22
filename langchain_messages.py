from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent(
    model="ollama:qwen3:4b",
    system_prompt="You are a spiritual master."
)

msg = HumanMessage(content="What is the meaning of life?")
result = agent.invoke({"messages": [msg]})

print(result["messages"][-1].content)