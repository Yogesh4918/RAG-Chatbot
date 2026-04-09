"""
Vector Store Module
Manages ChromaDB operations for persistent vector storage and retrieval.
"""

import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from backend.embeddings import get_embedding_function


def get_vector_store():
    """
    Get or create a persistent ChromaDB vector store.
    
    Returns:
        Chroma: LangChain Chroma vector store instance.
    """
    embedding_function = get_embedding_function()
    
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vector_store


def add_documents_to_store(documents: List[Document]) -> int:
    """
    Add document chunks to the ChromaDB vector store.
    
    Each chunk is embedded and stored with its metadata (source, page number).
    
    Args:
        documents: List of chunked Document objects.
        
    Returns:
        int: Number of documents added.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return len(documents)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Perform similarity search in the vector store.
    
    Converts the query into an embedding and finds the k most similar
    document chunks using cosine similarity.
    
    Args:
        query: User's question or search query.
        k: Number of similar documents to retrieve.
        
    Returns:
        List[Document]: Top-k most relevant document chunks.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results


def get_retriever(k: int = 4):
    """
    Get a retriever interface for the vector store.
    
    The retriever wraps similarity search into a LangChain-compatible
    interface for use in chains.
    
    Args:
        k: Number of documents to retrieve.
        
    Returns:
        VectorStoreRetriever: LangChain retriever object.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def get_document_count() -> int:
    """
    Get the total number of documents stored in the vector store.
    
    Returns:
        int: Number of document chunks in the store.
    """
    vector_store = get_vector_store()
    collection = vector_store._collection
    return collection.count()


def clear_vector_store():
    """
    Clear all documents from the vector store by deleting and recreating the collection.
    Uses ChromaDB's API instead of file deletion to avoid lock issues.
    """
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except ValueError:
        pass  # Collection doesn't exist, nothing to clear
