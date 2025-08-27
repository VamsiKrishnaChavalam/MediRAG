# application.py
import os
import re
import json
import time
import tempfile
from datetime import datetime

import streamlit as st
import nltk
import pytesseract
from pdf2image import convert_from_path

# Pinecone & hybrid search (new pinecone package)
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

# OpenAI-compatible (Groq) client
import openai
import pandas as pd

# -----------------------------
# One-time NLTK setup
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

# -----------------------------
# Streamlit page config & Optimized CSS
# -----------------------------
st.set_page_config(
    page_title="MediRAG", 
    layout="wide", 
    initial_sidebar_state="expanded",
    
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables */
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f2e;
        --bg-tertiary: #252b3a;
        --bg-accent: #2a3441;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --color-accent: #4285f4;
        --color-success: #34a853;
        --color-warning: #fbbc04;
        --color-error: #ea4335;
        --border-color: #3c4043;
        --border-radius: 12px;
        --border-radius-sm: 8px;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.4);
        --transition: all 0.2s ease;
        --space-xs: 0.5rem;
        --space-sm: 1rem;
        --space-md: 1.5rem;
        --space-lg: 2rem;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding: var(--space-md) var(--space-lg);
        max-width: 100%;
    }
    
    /* App Header */
    .app-header {
        background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-accent) 100%);
        padding: var(--space-lg);
        border-radius: var(--border-radius);
        margin-bottom: var(--space-md);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-lg);
        text-align: center;
    }
    
    .app-header h1 {
        color: var(--text-primary);
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 var(--space-xs) 0;
        background: linear-gradient(135deg, var(--text-primary), var(--color-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .app-header .subtitle {
        color: var(--text-secondary);
        margin: 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Chat Container */
    .chat-container {
        background: var(--bg-secondary);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: var(--space-md);
        box-shadow: var(--shadow);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    
    /* Welcome Message */
    .welcome-message {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 500px;
        text-align: center;
        padding: var(--space-lg);
        color: var(--text-primary);
    }
    
    .welcome-content h2 {
        font-size: 2.2rem;
        margin-bottom: var(--space-md);
        color: var(--text-primary);
        font-weight: 600;
        background: linear-gradient(135deg, var(--text-primary), var(--color-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .welcome-content p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: var(--space-lg);
        font-size: 1.1rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .feature-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--space-md);
        max-width: 700px;
        margin: 0 auto;
    }
    
    .feature-card {
        background: var(--bg-tertiary);
        padding: var(--space-md);
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--border-color);
        text-align: center;
        transition: var(--transition);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--color-accent);
    }
    
    .feature-card h4 {
        color: var(--text-primary);
        margin-bottom: var(--space-xs);
        font-size: 1rem;
        font-weight: 600;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Message Styling */
    .message-row {
        border-bottom: 1px solid var(--border-color);
        padding: 0;
        margin: 0;
        transition: var(--transition);
    }
    
    .user-message-row {
        background: var(--bg-secondary);
    }
    
    .user-message-row:hover {
        background: var(--bg-tertiary);
    }
    
    .bot-message-row {
        background: var(--bg-tertiary);
    }
    
    .bot-message-row:hover {
        background: var(--bg-accent);
    }
    
    .message-content {
        max-width: 800px;
        margin: 0 auto;
        padding: var(--space-md);
        display: flex;
        gap: var(--space-sm);
        align-items: flex-start;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        flex-shrink: 0;
        box-shadow: var(--shadow);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, var(--color-accent), #5a9fd4);
        color: white;
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, var(--color-success), #4caf50);
        color: white;
    }
    
    .message-text {
        flex: 1;
        line-height: 1.6;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .message-text p {
        margin-bottom: var(--space-sm);
        color: var(--text-primary);
    }
    
    .message-text p:last-child {
        margin-bottom: 0;
    }
    
    /* Sidebar */
    .sidebar-content {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: var(--space-sm);
    }
    
    .upload-section {
        background: var(--bg-tertiary);
        padding: var(--space-md);
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--border-color);
        margin-bottom: var(--space-md);
        transition: var(--transition);
        position: relative;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-accent), var(--color-success));
    }
    
    .upload-section:hover {
        border-color: var(--color-accent);
        box-shadow: var(--shadow);
    }
    
    .upload-section h4 {
        color: var(--text-primary);
        font-size: 1rem;
        margin-bottom: var(--space-xs);
        font-weight: 600;
    }
    
    .upload-section p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: var(--space-sm);
        line-height: 1.5;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--bg-accent) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius-sm) !important;
        color: var(--text-primary) !important;
        padding: var(--space-sm) !important;
        font-size: 0.95rem !important;
        transition: var(--transition) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.15) !important;
        outline: none !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--color-accent), #5a9fd4) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius-sm) !important;
        padding: var(--space-sm) var(--space-md) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow) !important;
        min-height: 42px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3367d6, var(--color-accent)) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Feedback buttons */
    .feedback-section {
        margin-top: var(--space-sm);
        padding-top: var(--space-sm);
        border-top: 1px solid var(--border-color);
    }
    
    .feedback-btn {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        padding: var(--space-xs) var(--space-sm) !important;
        margin: var(--space-xs) !important;
        min-height: 32px !important;
        border-radius: var(--border-radius-sm) !important;
    }
    
    .feedback-btn:hover {
        background: var(--bg-accent) !important;
        border-color: var(--color-accent) !important;
        color: var(--text-primary) !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div > div {
        background-color: var(--bg-accent) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--border-radius-sm) !important;
        transition: var(--transition) !important;
        padding: var(--space-md) !important;
        min-height: 100px !important;
    }
    
    .stFileUploader > div > div > div:hover {
        border-color: var(--color-accent) !important;
        background-color: var(--bg-tertiary) !important;
    }
    
    /* Status Messages */
    .status-message {
        padding: var(--space-sm) var(--space-md);
        border-radius: var(--border-radius-sm);
        margin: var(--space-sm) 0;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: var(--space-xs);
        line-height: 1.4;
    }
    
    .success-message {
        background: rgba(52, 168, 83, 0.1);
        color: var(--color-success);
        border: 1px solid rgba(52, 168, 83, 0.3);
    }
    
    .success-message::before {
        content: '✓';
        font-weight: bold;
    }
    
    .error-message {
        background: rgba(234, 67, 53, 0.1);
        color: var(--color-error);
        border: 1px solid rgba(234, 67, 53, 0.3);
    }
    
    .error-message::before {
        content: '⚠';
        font-weight: bold;
    }
    
    .info-message {
        background: rgba(66, 133, 244, 0.1);
        color: var(--color-accent);
        border: 1px solid rgba(66, 133, 244, 0.3);
    }
    
    .info-message::before {
        content: 'ℹ';
        font-weight: bold;
    }
    
    /* Health Indicator */
    .health-indicator {
        display: flex;
        align-items: center;
        gap: var(--space-xs);
        padding: var(--space-sm);
        background: var(--bg-tertiary);
        border-radius: var(--border-radius-sm);
        margin: var(--space-sm) 0;
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }
    
    .health-indicator:hover {
        background: var(--bg-accent);
    }
    
    .health-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .health-healthy {
        background: var(--color-success);
        color: var(--color-success);
        box-shadow: 0 0 6px currentColor;
    }
    
    .health-error {
        background: var(--color-error);
        color: var(--color-error);
        box-shadow: 0 0 6px currentColor;
    }
    
    .health-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--color-accent) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric {
        background: var(--bg-tertiary) !important;
        padding: var(--space-sm) !important;
        border-radius: var(--border-radius-sm) !important;
        border: 1px solid var(--border-color) !important;
        margin: var(--space-xs) 0 !important;
    }
    
    /* Dashboard */
    .dashboard-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-sm);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    
    .dashboard-card:hover {
        border-color: var(--color-accent);
        box-shadow: var(--shadow-lg);
    }
    
    .dashboard-card h3 {
        color: var(--text-primary);
        margin-bottom: var(--space-sm);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: var(--space-lg) 0 var(--space-md) 0;
        padding-bottom: var(--space-xs);
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-tertiary) !important;
        border-radius: var(--border-radius-sm) !important;
        padding: var(--space-xs) !important;
        border: 1px solid var(--border-color) !important;
        margin-bottom: var(--space-md) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border-radius: var(--border-radius-sm) !important;
        font-weight: 500 !important;
        transition: var(--transition) !important;
        padding: var(--space-sm) !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        background: linear-gradient(135deg, var(--color-accent), #5a9fd4) !important;
        box-shadow: var(--shadow) !important;
    }
    
    /* Form */
    .stForm {
        background: var(--bg-tertiary);
        padding: var(--space-md);
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--border-color);
        margin: var(--space-sm) 0;
        box-shadow: var(--shadow);
    }
    
    /* DataFrame */
    .stDataFrame {
        margin: var(--space-sm) 0 !important;
        border-radius: var(--border-radius-sm) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow) !important;
    }
    
    .stDataFrame, .stDataFrame td, .stDataFrame th {
        color: var(--text-primary) !important;
        font-size: 0.85rem !important;
        background-color: var(--bg-tertiary) !important;
        padding: var(--space-xs) !important;
    }
    
    .stDataFrame th {
        background-color: var(--bg-accent) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        padding: var(--space-sm) var(--space-xs) !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 2rem;
        }
        
        .welcome-content h2 {
            font-size: 1.8rem;
        }
        
        .feature-cards {
            grid-template-columns: 1fr;
            gap: var(--space-sm);
        }
        
        .message-content {
            padding: var(--space-sm);
        }
        
        .main .block-container {
            padding: var(--space-sm);
        }
    }
    
    /* Utility classes */
    .mb-sm { margin-bottom: var(--space-sm); }
    .mb-md { margin-bottom: var(--space-md); }
    .mt-sm { margin-top: var(--space-sm); }
    .mt-md { margin-top: var(--space-md); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Config & secrets
# -----------------------------
def get_secret(name, default=None):
    """
    Prefer Streamlit secrets, then env var, then default.
    This avoids requiring a separate file if you provide env vars.
    """
    try:
        # st.secrets throws if no secrets.toml present; use get safely
        val = None
        try:
            val = st.secrets.get(name)
        except Exception:
            val = None
        return val if val is not None else os.environ.get(name, default)
    except Exception:
        return os.environ.get(name, default)

# API Keys - Replace with your actual keys
LLM_API_KEY = "your llm api key"
PINECONE_API_KEY = "your pinecone api key"
PINECONE_INDEX_NAME = "your pinecone index"
HF_TOKEN = "hugging face token"

GEN_MODEL = "llama3-70b-8192"
JUDGE_MODEL = "llama3-70b-8192"

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# -----------------------------
# Initialize services
# -----------------------------
@st.cache_resource
def initialize_services():
    try:
        if not (LLM_API_KEY and PINECONE_API_KEY):
            raise RuntimeError("Missing LLM_API_KEY or PINECONE_API_KEY")

        llm_client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=LLM_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        bm25_path = "rag-veda.json"
        if os.path.exists(bm25_path):
            bm25_encoder = BM25Encoder().load(bm25_path)
        else:
            bm25_encoder = BM25Encoder()

        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index
        )

        return {"llm": llm_client, "retriever": retriever, "embeddings": embeddings,
                "bm25_encoder": bm25_encoder, "index": index, "bm25_path": bm25_path,
                "status": "healthy"}
    except Exception as e:
        return {"llm": None, "retriever": None, "embeddings": None, "bm25_encoder": None,
                "index": None, "bm25_path": None, "status": "error", "error": str(e)}

services = initialize_services()

# -----------------------------
# Utilities: OCR & chunking with metadata
# -----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_text_pages(pdf_path: str, pdf_name: str) -> list:
    """
    OCR the PDF and return a list of dicts:
    [{"page": 1, "text": "....", "pdf_name": "file.pdf"}, ...]
    """
    pages = []
    try:
        images = convert_from_path(pdf_path)
        for i, img in enumerate(images, start=1):
            txt = clean_text(pytesseract.image_to_string(img))
            if txt:
                pages.append({"page": i, "text": txt, "pdf_name": pdf_name})
        return pages
    except Exception as e:
        st.error(f"OCR extraction error: {e}")
        return []

def chunk_page_text(page_dict, chunk_size=800):
    """
    Create chunks from a page while preserving metadata for citation.
    Returns texts[], metadatas[] aligned lists.
    """
    txt = page_dict["text"]
    chunks = [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size) if txt[i:i+chunk_size].strip()]
    texts = []
    metas = []
    for j, ch in enumerate(chunks):
        texts.append(ch)
        metas.append({
            "pdf_name": page_dict.get("pdf_name"),
            "page": page_dict.get("page"),
            "chunk_id": j
        })
    return texts, metas

# -----------------------------
# Retriever helpers
# -----------------------------
def get_contexts(query: str, retriever, top_n=3):
    """
    Return top_n Document-like objects from the retriever (with page_content and metadata).
    """
    try:
        docs = retriever.invoke(query)
        return docs[:top_n]
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

# -----------------------------
# Prompt building: strict PDF-only instructions
# -----------------------------
def build_rag_prompt(query, message_history, docs):
    history_text = ""
    if message_history:
        for msg in message_history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Format context blocks with explicit citation markers
    if docs:
        ctx_blocks = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            pdf = meta.get("pdf_name")
            page = meta.get("page", "?")
            ctx_blocks.append(f"[C{i}] {pdf} (p.{page})\n{d.page_content}")
        context_blocks = "\n\n".join(ctx_blocks)
    else:
        context_blocks = "NO_RELEVANT_DOCUMENTS_FOUND"

    prompt = f"""
You are a knowledgeable medical assistant tasked with answering user queries clearly and accurately. You provide preventions and diet plans when requested.

### Chat History:
{history_text}

### Knowledge Base Context:
{context_blocks if context_blocks else "No relevant documents were found."}

Now, based on the above context and chat history, answer the following question:

Question: {query}

Guidelines:
- If the user greets you, greet them back politely. do not add any citations for greeting. dont add any sources to the response too.
- Always base your answers strictly on the provided context. Do not use any external knowledge or make assumptions.
- Do not greet for every question asked. Greet only if the user greets you first.
- Never fabricate or guess answers. If the information is not in the context, say you don't know.
- Never start a response with "As an AI language model".
- Never provide medical advice beyond general information. Always recommend consulting a healthcare professional for specific concerns.
- If the user asks for a summary of the documents, provide a brief overview without citations.
- If the context contains "NO_RELEVANT_DOCUMENTS_FOUND", respond with: "I couldn't find this information in the uploaded documents."
- If the answer is not contained within the provided context, respond with: "I'm sorry, I don't have that information in the uploaded documents."
- EXAMPLE FORMAT FOR GREETING:
User: Hello
Assistant: Hello! How can I assist you today?

You are a polite and professional medical chatbot.  

- Your role is to answer medical or drug-related queries and respond to simple greetings.  
- If the user asks questions that are not related to medicine, health, or drugs (e.g., math problems, random trivia, coding), do not attempt to answer.  
- Instead, reply politely with something like:  
  "I'm a medical chatbot, here to help with medical or drug-related queries. Please ask me something in that domain."  

Always keep your tone helpful, clear, and user-friendly.

1. Standard Structure
- Every drug/condition response should follow the same clear sections:
- Drug Summary / Overview – short description (name, class, purpose).
- Usage / Indications – what it's used for.
- Dosage & Administration
- Adult dose
- Pediatric dose
- Route (PO/IV/other)
- Frequency
- Maximum dose (if applicable)
- Precautions / Safety Notes – contraindications, warnings, interactions (keep brief).
- What to Do Next – simple actionable advice (e.g., consult doctor, when to seek care).
- Disclaimer – always include a safety disclaimer.
- Citations – structured reference to source(s).

2. Tone & Language
- Clear, concise, non-technical (layman-friendly).
- Avoid jargon unless needed; explain medical terms briefly.
- Neutral and professional — never give direct prescriptions, only general information.

3. Safety First

- Never give exact personalized medical advice (like "you should take X now").
- Always include disclaimer: "This information is for educational purposes only and not a substitute for professional medical advice."
- Encourage users to consult a healthcare professional.

4. Consistency Rules

- Always use SI units (mg, g, mL).
- Show frequency as q4–6h, q12h etc.
- Specify adult vs pediatric doses separately.
- State routes (PO, IV, IM, etc.) clearly.

5. Citation Guidelines

- Always include at least one citation (real or placeholder).
- Citation should the pdf name

6. Response Length

- Keep Summary short (2–3 lines).
- Use bullet points for clarity.
- Avoid long paragraphs unless needed for explanation.

EXAMPLE RESPONSE FORMAT:
Summary
Paracetamol (Acetaminophen) is an analgesic and antipyretic used to reduce fever and mild to moderate pain.

Usage / Indications
- Fever
- Mild to moderate pain

Dosage & Administration
- Adult: 500–1000 mg PO/IV q4–6h (max 4 g/day)
- Pediatric: 10–15 mg/kg PO/IV q4–6h (max per guideline)

Precautions / Safety
- Avoid in severe liver disease
- Use with caution with alcohol or hepatotoxic drugs

What To Do Next
Paracetamol may be considered as directed. Always consult a doctor before use.

Disclaimer
This information is for educational purposes only and not medical advice. Always consult a healthcare professional.

Citations
- Source: WHO Guidelines, p. 24


### Chat history:
{history_text}

### Knowledge Base Context:
{context_blocks}

### Question:
{query}

Answer only using the context above.
"""
    return prompt.strip()

# -----------------------------
# LLM call helper
# -----------------------------
def llm_chat(client, model, prompt, temperature=0.0, max_tokens=1200):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# -----------------------------
# Deterministic sources output (from retrieved docs metadata)
# -----------------------------
def format_sources_from_docs(docs):
    seen = set()
    items = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        pdf = meta.get("pdf_name")
        page = meta.get("page", "?")
        key = (pdf, page)
        if key not in seen:
            seen.add(key)
            items.append(f"{pdf}, p. {page}")
    return items

# -----------------------------
# LLM-as-Judge (optional, kept from your pipeline)
# -----------------------------
def judge_score(client, model, question, answer, contexts, dimension: str):
    ctx_joined = "\n\n".join([c.page_content for c in contexts]) if contexts else "NO_CONTEXT"
    rubric = {
        "faithfulness": "Score 1.0 if the answer is fully supported by the context, 0.0 if it contradicts or invents facts; scale smoothly otherwise.",
        "answer_relevancy": "Score 1.0 if the answer fully addresses the user's question, 0.0 if irrelevant; scale if partial.",
        "context_relevancy": "Score 1.0 if the retrieved context is relevant and useful to answer the question, 0.0 if irrelevant; scale if partially relevant."
    }[dimension]

    prompt = f"""
You are an impartial evaluator for a Retrieval-Augmented Generation system.

Dimension to score: {dimension}
Rubric: {rubric}

Question:
{question}

Answer:
{answer}

Retrieved Context:
{ctx_joined}

Return ONLY a JSON object like:
{{"score": 0.0, "reason": "brief reason"}}
"""
    try:
        out = llm_chat(client, model, prompt, temperature=0.0, max_tokens=200)
        m = re.search(r"\{.*\}", out, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {"score": None, "reason": "parse_error"}
        score = data.get("score", None)
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
        else:
            score = None
        return score, data.get("reason", "")
    except Exception as e:
        return None, f"judge_error: {e}"

def evaluate_realtime(client, model, question, answer, contexts):
    start = time.time()
    f_score, f_reason = judge_score(client, model, question, answer, contexts, "faithfulness")
    a_score, a_reason = judge_score(client, model, question, answer, contexts, "answer_relevancy")
    c_score, c_reason = judge_score(client, model, question, answer, contexts, "context_relevancy")
    latency = time.time() - start
    return {"faithfulness": f_score, "answer_relevancy": a_score, "context_relevancy": c_score,
            "latency_sec": latency, "reasons": {"faithfulness": f_reason, "answer_relevancy": a_reason, "context_relevancy": c_reason}}

# -----------------------------
# Session state initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""
if "last_answer_idx" not in st.session_state:
    st.session_state.last_answer_idx = None
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# -----------------------------
# App header with enhanced spacing
# -----------------------------
st.markdown("""
<div class="app-header">
    <h1>MediRAG - AI Medical Assistant</h1>
    <div class="subtitle">Your intelligent medical information companion powered by advanced AI</div>
</div>
""", unsafe_allow_html=True)

# Add some breathing room
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Tabs: Chat + Metrics with enhanced styling
# -----------------------------
tab_chat, tab_metrics = st.tabs(["Chat", "Metrics Dashboard"])

# -----------------------------
# ENHANCED CHAT TAB
# -----------------------------
with tab_chat:
    # Add spacing before columns
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_main, col_sidebar = st.columns([3, 1], gap="large")

    with col_main:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div class="welcome-message">
                    <div class="welcome-content">
                        <h2>🩺 Welcome to MediRAG</h2>
                        <p>I'm your AI-powered medical assistant. Upload medical PDFs to get personalized, evidence-based responses with proper citations.</p>
                        <div class="feature-cards">
                            <div class="feature-card">
                                <h4>💊 Drug Information</h4>
                                <p>Comprehensive medication details, dosages, and interactions from uploaded documents.</p>
                            </div>
                            <div class="feature-card">
                                <h4>🔬 Health Guidance</h4>
                                <p>Evidence-based health advice and medical information.</p>
                            </div>
                            <div class="feature-card">
                                <h4>📄 PDF Analysis</h4>
                                <p>Personalized insights extracted from your uploaded medical PDFs.</p>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, m in enumerate(st.session_state.messages):
                    if m["role"] == "user":
                        st.markdown(f"""
                        <div class="message-row user-message-row">
                            <div class="message-content">
                                <div class="message-avatar user-avatar">👤</div>
                                <div class="message-text"><p>{m["content"]}</p></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        formatted = m["content"].replace('\n\n', '</p><p>').replace('\n', '<br>')
                        if not formatted.startswith('<p>'):
                            formatted = f'<p>{formatted}</p>'
                        st.markdown(f"""
                        <div class="message-row bot-message-row">
                            <div class="message-content">
                                <div class="message-avatar bot-avatar">AI</div>
                                <div class="message-text">{formatted}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced feedback section with better spacing
                        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([0.15, 0.15, 0.7])
                        with col1:
                            if st.button("👍", key=f"up_{i}", help="Mark as helpful response", use_container_width=True):
                                st.session_state.feedback.append({
                                    "index": i,
                                    "response": m["content"],
                                    "feedback": "The response is relevant and helpful"
                                })
                                if st.session_state.get("logs"):
                                    st.session_state.logs[-1]["feedback"] = "Helpful response provided"
                                    pd.DataFrame(st.session_state.logs).to_csv("logs/metrics.csv", index=False)
                                st.success("Thank you for your feedback!")

                        with col2:
                            if st.button("👎", key=f"down_{i}", help="Mark as not helpful", use_container_width=True):
                                st.session_state.feedback.append({
                                    "index": i,
                                    "response": m["content"],
                                    "feedback": "The response needs improvement"
                                })
                                if st.session_state.get("logs"):
                                    st.session_state.logs[-1]["feedback"] = "Response needs improvement"
                                    pd.DataFrame(st.session_state.logs).to_csv("logs/metrics.csv", index=False)
                                st.warning("Thanks! We'll work on improving our responses.")
                        st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced chat input form with proper spacing
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message", 
                placeholder="Ask about medications, conditions, or treatments from uploaded PDFs...", 
                label_visibility="collapsed", 
                key="message_input"
            )
            
            # Add spacing between input and buttons
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col2:
                submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)
            with col3:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_answer_idx = None
                    st.rerun()

            if submit_button and user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                
                with st.spinner("Analyzing your question..."):
                    try:
                        if services["status"] != "healthy":
                            raise RuntimeError(services.get("error", "Service unavailable"))

                        # Retrieve contexts (documents with metadata)
                        contexts = get_contexts(user_input.strip(), services["retriever"], top_n=4)

                        # Build prompt that forces PDF-only answers
                        rag_prompt = build_rag_prompt(user_input.strip(), st.session_state.messages, contexts)

                        # Call generator model
                        start_gen = time.time()
                        answer = llm_chat(services["llm"], GEN_MODEL, rag_prompt, temperature=0.0, max_tokens=1200)
                        gen_latency = time.time() - start_gen

                        # If the LLM replied with the abstain phrase or contexts empty, enforce abstain
                        if not contexts:
                            final_answer = "I couldn't find this information in the uploaded documents. Please upload relevant medical PDFs to get accurate answers."
                            used_contexts = []
                        else:
                            # Build deterministic sources list from retrieved contexts
                            sources = format_sources_from_docs(contexts)
                            if "I couldn't find" in answer or "couldn't find" in answer.lower():
                                final_answer = "I couldn't find this information in the uploaded documents. Please upload relevant medical PDFs."
                                used_contexts = []
                            else:
                                final_answer = answer
                                used_contexts = contexts

                        # Append assistant message
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        st.session_state.last_answer_idx = len(st.session_state.messages) - 1

                        # Evaluate using judge model (optional)
                        metrics = evaluate_realtime(services["llm"], JUDGE_MODEL, user_input.strip(), final_answer, used_contexts) if services["llm"] else {"faithfulness": None, "answer_relevancy": None, "context_relevancy": None, "latency_sec": None, "reasons": {}}

                        # Persist log entry
                        log_entry = {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "question": user_input.strip(),
                            "answer": final_answer,
                            "faithfulness": metrics.get("faithfulness"),
                            "answer_relevancy": metrics.get("answer_relevancy"),
                            "context_relevancy": metrics.get("context_relevancy"),
                            "gen_latency_sec": round(gen_latency, 3),
                            "eval_latency_sec": round(metrics.get("latency_sec", 0), 3) if metrics.get("latency_sec") else None,
                            "judge_reasons": json.dumps(metrics.get("reasons", {})),
                            "feedback": None,
                        }
                        st.session_state.logs.append(log_entry)

                        # Save logs persistently
                        os.makedirs("logs", exist_ok=True)
                        pd.DataFrame(st.session_state.logs).to_csv("logs/metrics.csv", index=False)

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ I encountered an error while processing your request: {str(e)}"})
                
                st.rerun()

    # Enhanced Sidebar with proper spacing
    with col_sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-section">
                <div class="upload-section">
                    <h4>Upload Medical PDF</h4>
                    <p>Upload medical PDFs, research papers, or drug information documents. The assistant will provide answers based exclusively on uploaded content.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # File uploader with enhanced feedback and spacing
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=['pdf'], 
            help="Upload medical PDFs for accurate, evidence-based responses", 
            label_visibility="collapsed"
        )

        # Add spacing after file uploader
        st.markdown("<br>", unsafe_allow_html=True)

        # Process file only if it's new and not already processed
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            if not st.session_state.file_processed or st.session_state.last_uploaded_file != file_id:
                with st.spinner("Processing your PDF..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            pdf_path = tmp_file.name

                        pdf_name = uploaded_file.name
                        pages = extract_text_pages(pdf_path, pdf_name)

                        if not pages:
                            st.markdown(
                                '<div class="status-message error-message">No text could be extracted from the PDF. Please ensure the PDF contains readable text.</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            all_texts, all_metas = [], []
                            for p in pages:
                                texts, metas = chunk_page_text(p, chunk_size=800)
                                all_texts.extend(texts)
                                all_metas.extend(metas)

                            if services['status'] == 'healthy':
                                # fit BM25 and re-create retriever
                                services['bm25_encoder'].fit(all_texts)
                                services['bm25_encoder'].dump(services['bm25_path'])

                                services['retriever'] = PineconeHybridSearchRetriever(
                                    embeddings=services['embeddings'],
                                    sparse_encoder=services['bm25_encoder'],
                                    index=services['index']
                                )

                                # add texts + metadata
                                services['retriever'].add_texts(texts=all_texts, metadatas=all_metas)

                                st.markdown(
                                    f'<div class="status-message success-message">Successfully processed "{uploaded_file.name}" with {len(pages)} pages and {len(all_texts)} chunks!</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    '<div class="status-message error-message">Service temporarily unavailable for PDF processing. Please try again later.</div>',
                                    unsafe_allow_html=True
                                )

                        os.unlink(pdf_path)

                        # Update session state
                        st.session_state.file_processed = True
                        st.session_state.last_uploaded_file = file_id

                    except Exception as e:
                        st.markdown(
                            f'<div class="status-message error-message">Error processing PDF: {str(e)}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    f'<div class="status-message info-message">"{uploaded_file.name}" is already processed and ready to use!</div>',
                    unsafe_allow_html=True
                )

        # Add divider with spacing
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced system status section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### System Status")
        if services.get('status') == 'healthy':
            st.markdown("""
            <div class="health-indicator">
                <div class="health-dot health-healthy"></div>
                <div class="health-text">All services online and ready</div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="health-indicator">
                <div class="health-dot health-error"></div>
                <div class="health-text">Service issues detected</div>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Service Error: {services.get('error', 'Unknown error')}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Quick stats section with spacing
        if st.session_state.messages:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Session Stats")
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            
            # Create metrics with proper spacing
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Messages", total_messages)
            with col2:
                st.metric("User Questions", user_messages)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ENHANCED METRICS DASHBOARD TAB
# -----------------------------
with tab_metrics:
    # Add spacing at top of metrics tab
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<h1 class="section-header">📊 RAG Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Real-time performance metrics and evaluation data for the MediRAG system.")
    
    # Add spacing after description
    st.markdown("<br>", unsafe_allow_html=True)

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/metrics.csv"
    
    # Load and merge data
    if os.path.exists(csv_path):
        try:
            df_file = pd.read_csv(csv_path)
            if st.session_state.logs:
                df_mem = pd.DataFrame(st.session_state.logs)
                df = pd.concat([df_file, df_mem]).drop_duplicates(subset=["time", "question"], keep="last")
            else:
                df = df_file
        except Exception:
            df = pd.DataFrame(st.session_state.logs)
    else:
        df = pd.DataFrame(st.session_state.logs)

    if df.empty:
        # Enhanced empty state with better spacing
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.info("No evaluation data available yet. Start chatting to generate metrics and insights!")
        st.markdown("""
        ### What you'll see here:
        - **Faithfulness Score**: How well responses stick to the uploaded documents
        - **Answer Relevancy**: How well responses address user questions  
        - **Context Relevancy**: How relevant retrieved context is to queries
        - **Response Times**: System performance metrics
        - **User Feedback**: Real-time satisfaction tracking
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Convert numeric columns
        for col in ["faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "eval_latency_sec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Summary metrics with enhanced spacing
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Performance Overview")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="large")
        
        with col1:
            faith_avg = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0
            st.metric("Faithfulness", f"{faith_avg:.3f}", help="How well responses stick to document content")
        
        with col2:
            ans_avg = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0
            st.metric("Answer Relevancy", f"{ans_avg:.3f}", help="How well responses address user questions")
        
        with col3:
            ctx_avg = df['context_relevancy'].mean() if 'context_relevancy' in df.columns else 0
            st.metric("Context Relevancy", f"{ctx_avg:.3f}", help="How relevant retrieved content is")
        
        with col4:
            latency_avg = df['gen_latency_sec'].mean() if 'gen_latency_sec' in df.columns else 0
            st.metric("Avg Response Time", f"{latency_avg:.2f}s", help="Average time to generate responses")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing between sections
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Interactive data table with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Recent Interactions")
        st.markdown("<br>", unsafe_allow_html=True)
        
        show_cols = ["time", "question", "faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "feedback"]
        available_cols = [col for col in show_cols if col in df.columns]
        
        if available_cols:
            display_df = df[available_cols].tail(10).round(3)
            st.dataframe(display_df, use_container_width=True, height=350)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Performance trends with enhanced styling
        if {"faithfulness", "answer_relevancy", "context_relevancy"}.issubset(df.columns):
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### Performance Trends")
            st.markdown("<br>", unsafe_allow_html=True)
            chart_data = df[["faithfulness", "answer_relevancy", "context_relevancy"]].tail(20)
            st.line_chart(chart_data, height=400)
            st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Detailed analysis with better layout
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            if "gen_latency_sec" in df.columns:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown("### Response Time Distribution")
                st.markdown("<br>", unsafe_allow_html=True)
                st.bar_chart(df["gen_latency_sec"].tail(10), height=300)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if "feedback" in df.columns:
                feedback_counts = df["feedback"].value_counts()
                if not feedback_counts.empty:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.markdown("### User Feedback Summary")
                    st.markdown("<br>", unsafe_allow_html=True)
                    for feedback, count in feedback_counts.items():
                        if feedback and feedback != "None":
                            st.write(f"**{feedback}**: {count} responses")
                    st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Judge reasoning analysis with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### AI Judge Analysis (Last 5 Interactions)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if "judge_reasons" in df.columns:
            for idx, (_, row) in enumerate(df.tail(5).iterrows()):
                with st.expander(f"Query {idx + 1}: {row.get('question', '')[:80]}..."):
                    try:
                        reasons = json.loads(row.get("judge_reasons", "{}"))
                        for dimension, reason in reasons.items():
                            if reason:
                                st.write(f"**{dimension.title()}**: {reason}")
                    except Exception:
                        st.write("Raw judge data:", row.get("judge_reasons", "No data"))
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Export functionality with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Data Export")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("Download CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Complete Data",
                    data=csv,
                    file_name=f"medirag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("Clear All Data", use_container_width=True):
                # Add a confirmation dialog
                if st.button("⚠️ Confirm Clear All", use_container_width=True):
                    st.session_state.logs = []
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    st.success("All data cleared!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add final spacing at bottom
st.markdown("<br><br><br>", unsafe_allow_html=True)
