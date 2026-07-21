from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langsmith import traceable


class SupportState(TypedDict):
    question: str
    context: list[str]
    answer: str


docs = [
    "Starter plans are limited to 5 users.",
    "Enterprise plans support unlimited users.",
    "Starter API limits are 1,000 requests per hour.",
]


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


@traceable(run_type="retriever", name="Retrieve Documents")
def retrieve_documents(query: str) -> list[str]:
    query_lower = query.lower()

    results = [
        document
        for document in docs
        if any(
            word in document.lower()
            for word in query_lower.split()
            if len(word) > 3
        )
    ]

    return results[:2]


def retrieve_node(state: SupportState) -> dict:
    return {
        "context": retrieve_documents(state["question"]),
    }


def answer_node(state: SupportState) -> dict:
    if not state["context"]:
        return {
            "answer": "I could not find that information."
        }

    context = "\n".join(state["context"])

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer only from the following context:\n\n"
                    f"{context}"
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )

    return {
        "answer": str(response.content),
    }


builder = StateGraph(SupportState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile()


result = graph.invoke(
    {
        "question": "How many Starter users are allowed?",
        "context": [],
        "answer": "",
    }
)

print(result["answer"])