"""
Embeddings Module
Initializes the HuggingFace embedding model for converting text into vectors.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import EMBEDDING_MODEL_NAME


def get_embedding_function():
    """
    Create and return a HuggingFace embedding function.
    
    Uses the 'all-MiniLM-L6-v2' model which produces 384-dimensional
    embeddings. This model runs locally — no API key required.
    
    Returns:
        HuggingFaceEmbeddings: Embedding function for use with vector stores.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},       # Use CPU for compatibility
        encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine similarity
    )
    return embeddings
