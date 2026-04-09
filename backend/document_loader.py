"""
Document Loader Module
Handles PDF ingestion, website URL loading, text extraction, and chunking using LangChain.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to the uploads directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        str: Full path to the saved file.
    """
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects (one per page).
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List[Document]: Extracted pages as LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_website(url: str) -> List[Document]:
    """
    Load content from a website URL and return Document objects.
    
    Uses WebBaseLoader to fetch and parse HTML content from the given URL.
    
    Args:
        url: The website URL to load content from.
        
    Returns:
        List[Document]: Extracted web content as LangChain Document objects.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.
    
    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs, sentences, words) to preserve context.
    
    Args:
        documents: List of LangChain Document objects.
        
    Returns:
        List[Document]: Chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def process_pdf(uploaded_file) -> List[Document]:
    """
    End-to-end PDF processing: save → load → chunk.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        List[Document]: Processed and chunked documents ready for embedding.
    """
    # Step 1: Save the uploaded file to disk
    file_path = save_uploaded_file(uploaded_file)
    
    # Step 2: Load the PDF and extract text
    documents = load_pdf(file_path)
    
    # Step 3: Split into chunks
    chunks = split_documents(documents)
    
    return chunks


def process_url(url: str) -> List[Document]:
    """
    End-to-end website URL processing: fetch → parse → chunk.
    
    Args:
        url: The website URL to process.
        
    Returns:
        List[Document]: Processed and chunked documents ready for embedding.
    """
    # Step 1: Load web content
    documents = load_website(url)
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    return chunks

