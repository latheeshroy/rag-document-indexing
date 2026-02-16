# rag-document-indexing
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()
print("Loaded Key:", os.getenv("GROQ_API_KEY"))


# ----------------------------
# Clear old chroma_db folder
# ----------------------------
def clear_chroma_db():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("Old chroma_db deleted.")

    os.mkdir("chroma_db")
    print("Fresh chroma_db created.")


# ----------------------------
# Load Documents
# ----------------------------
def load_documents():
    loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    documents = loader.load()
    print("Total Documents Loaded:", len(documents))
    return documents


# ----------------------------
# Split Documents
# ----------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)
    print("Total Chunks Created:", len(chunks))
    return chunks


# ----------------------------
# Create Vector Store
# ----------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="rag_collection_v1"
    )

    vector_store.persist()
    print("Vector store created successfully.")


# ----------------------------
# Test Search
# ----------------------------
def test_search():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="rag_collection_v1"
    )

    query = "What is RAG in AI?"
    results = vector_store.similarity_search(query, k=2)

    print("\nSearch Results:\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print("-" * 50)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    clear_chroma_db()
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
    test_search()
