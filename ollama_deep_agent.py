from uuid import uuid4
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import InMemorySaver


# Free web search tool.
# This tool uses the internet, but the Qwen model itself runs locally.
search = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web for current information. "
        "Use this for recent news, events, technologies, and facts "
        "that may have changed."
    ),
)

# Store conversation history while this Python program is running.
checkpointer = InMemorySaver()

# StateBackend creates a virtual filesystem inside the agent's state.
filesystem = FilesystemMiddleware(
    backend=StateBackend()
)

agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[search],
    middleware=[filesystem],
    checkpointer=checkpointer,
    system_prompt="""
You are a helpful research assistant running with a local Ollama model.

Instructions:
- Use web_search when the user asks about recent or current information.
- Do not claim that you searched unless you used the search tool.
- Summarize search findings clearly.
- Include source names or URLs when they appear in search results.
- Use filesystem tools when asked to create, read, or update notes.
- Store research notes in files when the user requests it.
- Be concise and accurate.
""",
)


def print_final_answer(result: dict) -> None:
    """Print the final AI response from an agent result."""
    messages = result.get("messages", [])

    if not messages:
        print("No messages were returned.")
        return

    final_message = messages[-1]
    print("\nAgent:")
    print(final_message.content)


def main() -> None:
    # Reusing this thread ID maintains the conversation and virtual files.
    config = {
        "configurable": {
            "thread_id": str(uuid4())
        }
    }

    print("Local Ollama Deep Agent")
    print("Model: qwen3:4b")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
            )

            print_final_answer(result)

        except Exception as error:
            print(f"\nError: {error}")
            print(
                "Check that Ollama is running and that "
                "'qwen3:4b' is installed."
            )


if __name__ == "__main__":
    main()