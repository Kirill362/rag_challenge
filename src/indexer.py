from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def build_vectorstore(documents, persist_path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists(persist_path):
        return FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(persist_path)
    return db
