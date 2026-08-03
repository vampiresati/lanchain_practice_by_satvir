from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFDirectoryLoader,
    TextLoader,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


# Load variables from .env
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to your .env file."
    )


# Folder containing PDF and TXT documents
directory = Path("My Documents")

if not directory.exists():
    directory.mkdir(parents=True)
    raise FileNotFoundError(
        f"Folder '{directory}' was created. "
        "Add PDF or TXT files and run the program again."
    )


# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
)


# --------------------------------------------------
# Load PDF documents
# --------------------------------------------------

pdf_loader = PyPDFDirectoryLoader(
    str(directory)
)

pdf_docs = pdf_loader.load()


# --------------------------------------------------
# Load TXT documents
# --------------------------------------------------

text_loader = DirectoryLoader(
    path=str(directory),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8",
        "autodetect_encoding": True,
    },
    silent_errors=True,
)

text_docs = text_loader.load()


# Combine PDF and TXT documents
docs = pdf_docs + text_docs

if not docs:
    raise ValueError(
        "No PDF or TXT documents were found inside "
        f"'{directory}'."
    )

print(f"Loaded {len(docs)} document pages/files.")


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(docs)

print("Documents added to the vector store.")


# --------------------------------------------------
# Ask questions continuously
# --------------------------------------------------

print("\nRAG assistant is ready.")
print("Type 'exit' to stop.\n")

while True:
    user_query = input("Enter your query: ").strip()

    if not user_query:
        continue

    if user_query.lower() in {"exit", "quit", "q"}:
        print("Assistant stopped.")
        break

    # Retrieve the two most relevant documents
    retrieved_docs = vector_store.similarity_search(
        query=user_query,
        k=2,
    )

    # Combine retrieved document text
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    system_prompt = f"""
You are a helpful assistant that answers questions about the supplied
documents.

Use only the following document context to answer the user's question.

If the answer cannot be found in the context, respond:
"I could not find that information in the provided documents."

Do not invent information.

Document context:

{context}
""".strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    try:
        response = llm.invoke(messages)

        print(f"\nAssistant: {response.content}\n")

        print("Retrieved sources:")

        for index, document in enumerate(
            retrieved_docs,
            start=1,
        ):
            source = document.metadata.get(
                "source",
                "Unknown source",
            )

            page = document.metadata.get("page")

            if page is not None:
                print(
                    f"{index}. {source}, page {page + 1}"
                )
            else:
                print(f"{index}. {source}")

        print()

    except Exception as error:
        print(f"\nError while generating answer: {error}\n")