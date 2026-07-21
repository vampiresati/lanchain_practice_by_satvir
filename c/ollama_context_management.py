from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

from deepagents.backends import StateBackend
from deepagents.middleware import (
    FilesystemMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
    SummarizationMiddleware,
)

# Shared backend
backend = StateBackend()

# Local Ollama model
model = "ollama:qwen3:4b"

# Search tool
search = DuckDuckGoSearchRun()

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        # Virtual filesystem
        FilesystemMiddleware(
            backend=backend,
        ),

        # Summarizes long conversations
        SummarizationMiddleware(
            model=model,
            backend=backend,
        ),

        # Loads persistent instructions
        MemoryMiddleware(
            backend=backend,
            sources=["./AGENTS.md"],
        ),

        # Loads reusable skills
        SkillsMiddleware(
            backend=backend,
            sources=["./skills"],
        ),
    ],
    system_prompt="""
You are a helpful AI assistant.

- Use the search tool when current information is needed.
- Use available skills when appropriate.
- Store and retrieve files using the filesystem.
- Be concise and accurate.
""",
)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Search for the latest LangChain news and save the summary to ai-news.md"
            }
        ]
    }
)
print(result["messages"][-1].content)