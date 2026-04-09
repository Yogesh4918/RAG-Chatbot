"""
RAG Chain Module
Builds the complete Retrieval-Augmented Generation pipeline using LangChain LCEL.
Combines the retriever, prompt template, and LLM into a unified chain.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from backend.llm_provider import get_llm
from backend.vector_store import get_retriever
from backend.config import RETRIEVAL_TOP_K


# ── Custom Prompt Template ──────────────────────────────────────────────────
# This prompt instructs the LLM to answer ONLY from the provided context,
# preventing hallucination and ensuring factual accuracy.

RAG_PROMPT_TEMPLATE = """You are a helpful and knowledgeable AI assistant. 
Use the following pieces of retrieved context to answer the user's question.

IMPORTANT RULES:
1. Answer ONLY based on the provided context.
2. If the context does not contain enough information to answer, say: 
   "I don't have enough information in the uploaded documents to answer this question."
3. Be detailed and thorough in your responses.
4. If relevant, mention which part of the document your answer comes from.
5. Use clear formatting with bullet points or numbered lists when appropriate.

Context:
{context}

Question: {question}

Helpful Answer:"""


def format_docs(docs):
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of Document objects from the retriever.
        
    Returns:
        str: Concatenated document contents separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """
    Build and return the complete RAG chain using LCEL (LangChain Expression Language).
    
    The chain follows this flow:
    1. User query → Retriever finds top-k relevant chunks
    2. Chunks are formatted and injected into the prompt as context
    3. Augmented prompt → LLM generates a grounded response
    
    Returns:
        Tuple of (chain, retriever) for querying with source tracking.
    """
    llm = get_llm()
    retriever = get_retriever(k=RETRIEVAL_TOP_K)
    
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Build LCEL chain: retriever → format → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


def ask_question(question: str) -> dict:
    """
    Process a user question through the RAG pipeline.
    
    Args:
        question: User's natural language question.
        
    Returns:
        dict: Contains 'result' (answer string) and 
              'source_documents' (list of retrieved Document chunks).
    """
    chain, retriever = get_rag_chain()
    
    # Get source documents separately for display
    source_documents = retriever.invoke(question)
    
    # Get the answer from the chain
    answer = chain.invoke(question)
    
    return {
        "result": answer,
        "source_documents": source_documents
    }
