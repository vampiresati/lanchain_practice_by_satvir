from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[search],
    middleware=[
        FilesystemMiddleware(
            backend=StateBackend()
        )
    ],
    system_prompt=(
        "You are a research assistant. "
        "Search the web for current questions and summarize accurately."
    ),
)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Search for recent AI news, summarize it, "
                    "and save the summary to /ai-news.md."
                ),
            }
        ]
    }
)

print(result["messages"][-1].content)
print("\nVirtual files:")
print(result.get("files", {}))