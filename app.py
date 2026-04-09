"""
Streamlit Chatbot UI for the RAG-based Document Q&A System.
Features a premium dark theme with glassmorphism, animated elements,
chat history, source document display, and PDF management.
"""

import sys
import os
import streamlit as st

# ── Add project root to path so backend imports work ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_loader import process_pdf, process_url
from backend.vector_store import add_documents_to_store, get_document_count, clear_vector_store
from backend.rag_chain import ask_question


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Chatbot | Intelligent Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS – Premium Dark Theme with Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ── Global Styles ──────────────────────────────────────────────── */
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }
    
    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
    }
    
    /* ── Main Header ────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(59,130,246,0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
    }
    
    /* ── Chat Messages ──────────────────────────────────────────────── */
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        animation: fadeIn 0.4s ease-out;
        line-height: 1.7;
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(79,70,229,0.15) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-left: 4px solid #8b5cf6;
        color: #e2e8f0;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(30,58,95,0.5) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        color: #e2e8f0;
    }
    
    /* ── Status Cards ───────────────────────────────────────────────── */
    .status-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ── Source Documents ────────────────────────────────────────────── */
    .source-doc {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    .source-doc-header {
        color: #60a5fa;
        font-weight: 600;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
    }
    
    /* ── Buttons ─────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* ── File Uploader ──────────────────────────────────────────────── */
    .stFileUploader {
        border: 2px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 14px !important;
        background: rgba(30, 41, 59, 0.3) !important;
    }
    
    /* ── Text Input ─────────────────────────────────────────────────── */
    .stChatInput > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 14px !important;
    }
    
    /* ── Expander ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
        color: #c4b5fd !important;
    }
    
    /* ── Animations ──────────────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .pulse { animation: pulse 2s infinite; }
    
    /* ── Scrollbar ───────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15, 12, 41, 0.5); }
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.5); }
    
    /* ── Divider ─────────────────────────────────────────────────────── */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
        margin: 1rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR – Document Management
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📚 Document Manager")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ── PDF Upload ──────────────────────────────────────────────────────
    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload PDF documents to build the knowledge base"
    )
    
    # ── Process PDFs Button ─────────────────────────────────────────────
    if uploaded_files:
        if st.button("🚀 Process PDFs", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                total_chunks = 0
                
                for uploaded_file in uploaded_files:
                    st.info(f"📄 Processing: {uploaded_file.name}")
                    chunks = process_pdf(uploaded_file)
                    num_added = add_documents_to_store(chunks)
                    total_chunks += num_added
                    st.session_state.uploaded_files.append(uploaded_file.name)
                
                st.session_state.documents_loaded = True
                st.session_state.total_chunks = get_document_count()
                st.success(f"✅ Processed {len(uploaded_files)} PDF(s) → {total_chunks} chunks indexed!")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ── Website URL Input ───────────────────────────────────────────────
    st.markdown("### 🌐 Add Website URL")
    url_input = st.text_input(
        "Enter a website URL",
        placeholder="https://example.com/article",
        key="url_input",
        help="Paste a website URL to extract and index its content"
    )
    
    if url_input:
        if st.button("🌐 Process URL", use_container_width=True):
            with st.spinner(f"Fetching content from {url_input}..."):
                try:
                    chunks = process_url(url_input)
                    num_added = add_documents_to_store(chunks)
                    st.session_state.uploaded_files.append(f"🌐 {url_input}")
                    st.session_state.documents_loaded = True
                    st.session_state.total_chunks = get_document_count()
                    st.success(f"✅ Website processed → {num_added} chunks indexed!")
                except Exception as e:
                    st.error(f"❌ Failed to load URL: {str(e)}")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ── Knowledge Base Stats ────────────────────────────────────────────
    st.markdown("### 📊 Knowledge Base")
    
    try:
        doc_count = get_document_count()
    except Exception:
        doc_count = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="status-card" style="text-align:center;">
            <div class="stat-number">{doc_count}</div>
            <div class="stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="status-card" style="text-align:center;">
            <div class="stat-number">{len(st.session_state.uploaded_files)}</div>
            <div class="stat-label">Sources</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Uploaded Files List ─────────────────────────────────────────────
    if st.session_state.uploaded_files:
        st.markdown("### 📁 Loaded Sources")
        for fname in st.session_state.uploaded_files:
            if fname.startswith("🌐"):
                st.markdown(f"- {fname}")
            else:
                st.markdown(f"- 📄 `{fname}`")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # ── Clear Actions ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Actions")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🧹 Clear KB", use_container_width=True):
            clear_vector_store()
            st.session_state.uploaded_files = []
            st.session_state.documents_loaded = False
            st.session_state.total_chunks = 0
            st.success("Knowledge base cleared!")
            st.rerun()
    
    # ── About Section ───────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div class="source-doc" style="font-size: 0.8rem;">
        <strong>RAG Chatbot v1.0</strong><br>
        Built with LangChain, Groq, ChromaDB<br>
        & HuggingFace Embeddings<br><br>
        <em>Powered by LLaMA 3 via Groq</em>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT – Chat Interface
# ═══════════════════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 RAG Document Chatbot</h1>
    <p>Upload PDFs or add website URLs and ask intelligent questions — powered by LLaMA 3 & LangChain</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome Message (if no documents loaded) ───────────────────────────────
if not st.session_state.uploaded_files:
    st.markdown("""
    <div class="chat-message bot-message">
        <strong>👋 Welcome!</strong><br><br>
        I'm your intelligent document assistant. Here's how to get started:<br><br>
        <strong>1.</strong> 📄 Upload PDF documents <strong>or</strong> 🌐 paste a website URL in the sidebar<br>
        <strong>2.</strong> 🚀 Click "Process" to index the content<br>
        <strong>3.</strong> 💬 Ask me anything about your documents or web pages!<br><br>
        <em>I use Retrieval-Augmented Generation (RAG) to provide accurate,
        context-aware answers grounded in your sources.</em>
    </div>
    """, unsafe_allow_html=True)

# ── Display Chat History ───────────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Show source documents if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Source Documents", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-doc">
                        <div class="source-doc-header">
                            📄 Source {i} | Page {source.get('page', 'N/A')} | {source.get('source', 'Unknown')}
                        </div>
                        {source.get('content', '')}
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat Input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Check if documents are loaded
    try:
        doc_count = get_document_count()
    except Exception:
        doc_count = 0
    
    if doc_count == 0:
        st.warning("⚠️ Please upload and process documents first using the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You</strong><br>{prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Get response from RAG chain
        with st.spinner("🔍 Searching documents and generating response..."):
            try:
                response = ask_question(prompt)
                answer = response["result"]
                source_docs = response.get("source_documents", [])
                
                # Format source documents
                sources = []
                for doc in source_docs:
                    sources.append({
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "page": doc.metadata.get("page", "N/A"),
                        "source": os.path.basename(doc.metadata.get("source", "Unknown"))
                    })
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Display assistant response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant</strong><br>{answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if sources:
                    with st.expander("📚 View Source Documents", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-doc">
                                <div class="source-doc-header">
                                    📄 Source {i} | Page {source['page']} | {source['source']}
                                </div>
                                {source['content']}
                            </div>
                            """, unsafe_allow_html=True)
                
            except ValueError as e:
                st.error(f"❌ Configuration Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.rerun()
