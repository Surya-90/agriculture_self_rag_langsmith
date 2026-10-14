import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def load_documents(data_dir: str) -> List[Document]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Document] = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        lower = filename.lower()
        if lower.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif lower.endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8", autodetect_encoding=True).load())

    if not documents:
        raise ValueError("No .pdf or .txt documents found in the data folder.")
    return documents


def get_embeddings() -> HuggingFaceEmbeddings:
    # Embeddings are computed locally on CPU; no document text is sent to Hugging Face APIs.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(data_dir: str):
    documents = load_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Documents were loaded, but no text chunks were created.")
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore, len(documents), len(chunks)
