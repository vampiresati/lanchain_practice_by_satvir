from langchain_ollama import ChatOllama


def main() -> None:
    model = ChatOllama(
        model="qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.2,
    )

    response = model.invoke(
        "Explain LangChain in simple words with one practical Python example."
    )

    print(response.content)


if __name__ == "__main__":
    main()