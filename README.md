# 🧠 RAG-based Intelligent Document Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask natural language questions about their content. Built with **LangChain**, **Groq** (LLaMA 3), **ChromaDB**, and **Streamlit**.

---

## 📋 Problem Domain Description

Traditional LLMs (Large Language Models) are powerful but limited to their training data — they cannot answer questions about private, domain-specific, or recently created documents. This creates a significant gap for organizations and individuals who need AI-powered Q&A over their own knowledge base.

**Our RAG Chatbot solves this** by combining:
- **Retrieval**: Semantic search over user-uploaded documents to find relevant context
- **Augmented Generation**: Injecting retrieved context into the LLM prompt to generate accurate, grounded responses

### Use Cases
- 📖 Research paper Q&A — Upload academic papers and ask questions
- 📋 Policy/compliance review — Query internal policy documents
- 📚 Study assistant — Upload textbooks and get explained answers
- 📄 Contract analysis — Ask questions about legal documents

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Dark Theme)                        │
│  ┌──────────────┐  ┌────────────────────────────────────────┐   │
│  │  PDF Upload   │  │          Chat Interface                │   │
│  │  Sidebar      │  │  User Query → AI Response + Sources   │   │
│  └──────┬───────┘  └───────────────┬────────────────────────┘   │
└─────────┼──────────────────────────┼────────────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────┐    ┌──────────────────────────────────────────┐
│ DOCUMENT INGEST │    │           RAG PIPELINE                   │
│                 │    │                                          │
│ PyPDFLoader     │    │  Query ──→ Embedding ──→ Vector Search  │
│      ↓          │    │                              ↓          │
│ Text Splitter   │    │  Top-K Chunks ──→ Prompt Template       │
│      ↓          │    │                       ↓                 │
│ Chunks          │    │              Groq LLM (LLaMA 3)         │
│      ↓          │    │                       ↓                 │
│ HuggingFace     │    │              Grounded Response           │
│ Embeddings      │    │                                          │
│      ↓          │    └──────────────────────────────────────────┘
│ ChromaDB        │                      ↑
│ (Persistent)    │──────────────────────┘
└─────────────────┘
```

### Data Flow

1. **Document Ingestion**: User uploads PDFs → extracted text → split into overlapping chunks (1000 chars, 200 overlap) → embedded using HuggingFace `all-MiniLM-L6-v2` → stored in ChromaDB
2. **Query Processing**: User question → embedded → cosine similarity search in ChromaDB → top-4 relevant chunks retrieved
3. **Response Generation**: Retrieved chunks + user question → injected into prompt template → sent to Groq LLM (LLaMA 3-8B) → grounded answer returned with source references

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangChain | Chain management, prompt templates, retrieval |
| **LLM** | Groq (LLaMA 3-8B) | Ultra-fast inference for response generation |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local text-to-vector conversion (384 dims) |
| **Vector Store** | ChromaDB | Persistent vector storage & similarity search |
| **Document Loading** | PyPDF | PDF text extraction |
| **UI** | Streamlit | Interactive chatbot interface |
| **Environment** | python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
RC/
├── backend/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration & environment variables
│   ├── document_loader.py   # PDF ingestion & text chunking
│   ├── embeddings.py        # HuggingFace embedding model setup
│   ├── vector_store.py      # ChromaDB operations (CRUD)
│   ├── llm_provider.py      # Groq LLM configuration
│   └── rag_chain.py         # LangChain RAG pipeline assembly
├── frontend/
│   └── app.py               # Streamlit chatbot UI
├── data/
│   └── uploads/             # Uploaded PDF storage (auto-created)
├── chroma_db/               # Persistent vector store (auto-created)
├── .env                     # API keys (not tracked in git)
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Free Groq API key ([Get one here](https://console.groq.com))

### Step 1: Clone or Navigate to Project
```bash
cd d:/Y1/RC
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key
Edit the `.env` file and add your Groq API key:
```
GROQ_API_KEY=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
```

### Step 5: Run the Application
```bash
streamlit run frontend/app.py
```

The app will launch at `http://localhost:8501`

---

## 💡 How to Use

1. **Upload Documents**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click "🚀 Process Documents" to ingest and index them
3. **Ask Questions**: Type your question in the chat input at the bottom
4. **View Sources**: Expand "📚 View Source Documents" to see which parts of your documents were used
5. **Manage**: Clear chat history or reset the knowledge base using sidebar buttons

---

## ⚙️ Design Choices

### Why Groq?
- **Speed**: Groq's LPU (Language Processing Unit) provides 10-100x faster inference than GPU-based solutions
- **Free tier**: Generous free API usage for development and testing
- **Quality models**: Access to LLaMA 3, Mixtral, and other top open-source models

### Why ChromaDB?
- **Persistent storage**: Data survives application restarts
- **No external server**: Runs embedded in the Python process
- **LangChain native**: First-class integration with the LangChain ecosystem

### Why HuggingFace Embeddings?
- **Runs locally**: No API key or internet required for embedding
- **Quality**: `all-MiniLM-L6-v2` offers excellent quality-speed tradeoff
- **384 dimensions**: Compact embeddings that are fast to search

### Chunking Strategy
- **1000 character chunks** with **200 character overlap** ensures:
  - Chunks are small enough for precise retrieval
  - Overlap prevents losing context at chunk boundaries
  - `RecursiveCharacterTextSplitter` preserves natural text boundaries

---

## 🔧 Configuration

All settings are centralized in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL_NAME` | `llama3-8b-8192` | Groq model to use |
| `LLM_TEMPERATURE` | `0.3` | Creativity level (0=deterministic, 1=creative) |
| `LLM_MAX_TOKENS` | `1024` | Maximum response length |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks to retrieve |

---

## 📝 License

This project is developed as an academic project for educational purposes.
