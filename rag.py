import os
import sys

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # pip install langchain-chroma
# pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

DOCS_PATH = "./docs"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print("Created docs/ folder. Add your PDFs and text files there.")
        return []

    documents = []
    for file in sorted(os.listdir(DOCS_PATH)):
        path = os.path.join(DOCS_PATH, file)
        try:
            if file.endswith(".pdf"):
                documents.extend(PyPDFLoader(path).load())
            elif file.endswith((".txt", ".md")):
                documents.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Reuse existing DB to avoid re-embedding and duplicating chunks on every run
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("Loading existing vector store from ./chroma_db")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print("Loading documents...")
    documents = load_documents()
    if not documents:
        return None
    print(f"Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH,
    )
    print("Vector store created")
    return vectorstore


def build_chain(vectorstore):
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.3,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        verbose=False,
    )


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("No documents found. Add files to the docs/ folder and try again.")
        return

    chain = build_chain(vectorstore)
    print("\nRAG Assistant ready. Type your questions (type 'quit' to exit):\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        try:
            response = chain.invoke({"question": question})
            print(f"\nAssistant: {response['answer']}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
