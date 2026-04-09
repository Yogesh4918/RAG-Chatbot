"""
Configuration module for the RAG Chatbot.
Loads environment variables and defines constants used across the pipeline.
"""

import os
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────
load_dotenv()

# ── Groq API Configuration ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# ── LLM Settings ───────────────────────────────────────────────────────────
LLM_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq-hosted LLaMA 3.3 70B
LLM_TEMPERATURE = 0.3                    # Lower = more deterministic
LLM_MAX_TOKENS = 1024                    # Maximum tokens in LLM response

# ── Embedding Settings ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # HuggingFace sentence-transformer

# ── Text Splitting Settings ────────────────────────────────────────────────
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks

# ── ChromaDB Settings ──────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "rag_documents"

# ── File Upload Settings ───────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

# ── Retrieval Settings ─────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 4        # Number of chunks to retrieve per query

# ── Create necessary directories ───────────────────────────────────────────
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
