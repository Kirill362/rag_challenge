import glob
import os
import hashlib
import pickle
from langchain_community.document_loaders import PyPDFLoader


CACHE_PATH = "cache/documents.pkl"


def load_documents():
    if os.path.exists(CACHE_PATH):
        print("Loading documents from cache...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("Parsing PDFs...")
    documents = []
    pdf_paths = glob.glob(os.path.join("data/pdfs", "*.pdf"))

    for path in pdf_paths:
        sha1 = compute_sha1(path)
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["pdf_sha1"] = sha1
            d.metadata["page_index"] = d.metadata.get("page", 0)

        documents.extend(docs)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"Cached {len(documents)} pages")
    return documents


def compute_sha1(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()
