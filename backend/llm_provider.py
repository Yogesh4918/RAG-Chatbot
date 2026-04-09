"""
LLM Provider Module
Configures and provides access to the Groq-hosted LLM for inference.
"""

from langchain_groq import ChatGroq
from backend.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS


def get_llm():
    """
    Initialize and return a Groq LLM instance.
    
    Groq provides ultra-fast inference for open-source models like
    LLaMA 3 and Mixtral using their custom LPU hardware.
    
    Returns:
        ChatGroq: Configured LLM instance.
        
    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file.\n"
            "Get a free API key at: https://console.groq.com"
        )
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return llm
