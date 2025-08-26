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
# Streamlit page config & CSS
# -----------------------------
st.set_page_config(page_title="MediRAG - AI Medical Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""

<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #343541;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #40414f 0%, #2d2e3f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #565869;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .app-header h1 {
        color: #ececf1;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        text-align: center;
    }
    
    .app-header .subtitle {
        color: #8e8ea0;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Chat container */
    .chat-container {
        background: #343541;
        border-radius: 12px;
        border: 1px solid #4a4a4a;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        padding: 0;
        margin-bottom: 1rem;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #40414f;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 4px;
    }
    
    /* Welcome message */
    .welcome-message {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 500px;
        text-align: center;
        padding: 2rem;
        color: #ececf1;
    }
    
    .welcome-content h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #ececf1;
        font-weight: 600;
    }
    
    .welcome-content p {
        color: #8e8ea0;
        line-height: 1.6;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .feature-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .feature-card {
        background: #2d2e3f;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #565869;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card h4 {
        color: #ececf1;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .feature-card p {
        color: #8e8ea0;
        font-size: 0.75rem;
        margin: 0;
    }
    
    /* Message styling */
    .message-row {
        border-bottom: 1px solid #4a4a4a;
        padding: 0;
        margin: 0;
    }
    
    .user-message-row {
        background: #343541;
    }
    
    .bot-message-row {
        background: #444654;
    }
    
    .message-content {
        max-width: 768px;
        margin: 0 auto;
        padding: 2rem;
        display: flex;
        gap: 1.5rem;
        align-items: flex-start;
    }
    
    .message-avatar {
        width: 40px;
        height: 30px;
        border-radius: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: #10a37f;
        color: white;
    }
    
    .bot-avatar {
        background: #19c37d;
        color: white;
    }
    
    .message-text {
        flex: 1;
        line-height: 1.7;
        color: #ececf1;
    }
    
    .message-text p {
        margin-bottom: 1rem;
        color: #ececf1;
    }
    
    .message-text p:last-child {
        margin-bottom: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202123;
    }
    
    .sidebar-content {
        background: #202123;
        color: #ececf1;
        padding: 1rem;
    }
    
    .upload-section {
        background: #2d2e3f;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #565869;
        margin-bottom: 1rem;
    }
    
    .upload-section h4 {
        color: #ececf1;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .upload-section p {
        color: #8e8ea0;
        font-size: 0.75rem;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 6px;
        color: #ececf1;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
    }
    
    .stTextArea > div > div > textarea {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 6px;
        color: #ececf1;
        padding: 0.75rem 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #19c37d;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #10a37f;
        transform: translateY(-1px);
    }
    
    /* File uploader */
    .stFileUploader > div > div > div {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 6px;
    }
    
    /* Status messages */
    .status-message {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.875rem;
    }
    
    .success-message {
        background: rgba(16, 163, 127, 0.1);
        color: #10a37f;
        border: 1px solid rgba(16, 163, 127, 0.3);
    }
    
    .error-message {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Labels and text */
    .stMarkdown, .stText {
        color: #ececf1;
    }
    
    label {
        color: #ececf1 !important;
    }
    
    /* Spinner */
    .stSpinner {
        color: #10a37f;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #40414f;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8e8ea0;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ececf1;
        background-color: #565869;
    }
    
    /* Chat input area - Fixed at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #343541;
        padding: 1rem 2rem 2rem;
        border-top: 1px solid #4a4a4a;
        z-index: 1000;
        box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .input-wrapper {
        background: #40414f;
        border: 1px solid #565869;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: flex-end;
        gap: 0.75rem;
        max-width: 768px;
        margin: 0 auto;
        transition: all 0.2s;
    }
    
    .input-wrapper:focus-within {
        border-color: #10a37f;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
    }
    
    /* Add padding to main content to avoid overlap with fixed input */
    .main .block-container {
        padding-bottom: 120px;
    }
    
    /* Health status indicator */
    .health-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: #2d2e3f;
        border-radius: 6px;
        margin-top: 1rem;
    }
    
    .health-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .health-healthy {
        background: #10a37f;
    }
    
    .health-error {
        background: #ef4444;
    }
    
    .health-text {
        font-size: 0.75rem;
        color: #8e8ea0;
    }
      /* Fix metric numbers */
    [data-testid="stMetricValue"] {
        color: #00FFAA !important;   /* bright teal */
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* Fix metric labels */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;   /* white */
        font-weight: 500;
    }
    <style>
    [data-testid="stMetricValue"] {
        color: #00FFAA !important;   /* bright teal */
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }


    /* Fix dataframe / judge reasons table */
    .stDataFrame, .stDataFrame td, .stDataFrame th {
        color: #ECECF1 !important;   /* light grey text */
        font-size: 0.9rem;
    }

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

# You can set these values using environment variables or .streamlit/secrets.toml.
# Example env var names: LLM_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, HF_TOKEN
LLM_API_KEY = "your api key "
PINECONE_API_KEY = "your pinecone db key"

PINECONE_INDEX_NAME = "pine cone index"
HF_TOKEN = "hugging face taken"

GEN_MODEL =  "llama3-70b-8192"
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
            # Allow continuing in dev even if keys not present, but mark unhealthy
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
- If the user greatss you, greet them back politely. do not add any citations for greeting . dont add any sources to the responce too.
- Always base your answers strictly on the provided context. Do not use any external knowledge or make assumptions.
- Do not greet for every question asked. Greet only if the user greets you first.
- Never fabricate or guess answers. If the information is not in the context, say you don't know.
- Never start a resonse with "As an AI language model".
- Never provide medical advice beyond general information. Always recommend consulting a healthcare professional for specific concerns.
- If the user asks for a summary of the documents, provide a brief overview without citations.
- If the context contains "NO_RELEVANT_DOCUMENTS_FOUND", respond with: "I couldn't find this information in the uploaded documents."
- If the answer is not contained within the provided context, respond with: "I'm sorry, I don't have that information in the uploaded documents."
- EXAMPLE FORMAT FOR GREETING:
User: Hello
Assistant: Hello! How can I assist you today?

1. Standard Structure
- Every drug/condition response should follow the same clear sections:
- Drug Summary / Overview – short description (name, class, purpose).
- Usage / Indications – what it’s used for.
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

- Never give exact personalized medical advice (like “you should take X now”).
- Always include disclaimer: “This information is for educational purposes only and not a substitute for professional medical advice.”
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

def render_chat_message(role, content, index, query=None):
    with st.chat_message(role):
        st.markdown(content)

        # Only show feedback for assistant messages
        if role == "assistant":
            col1, col2 = st.columns([0.15, 0.15])
            with col1:
                if st.button("👍", key=f"up_{index}"):
                    st.session_state["feedback"].append({
                        "query": query,
                        "response": content,
                        "feedback": "up",
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"down_{index}"):
                    st.session_state["feedback"].append({
                        "query": query,
                        "response": content,
                        "feedback": "down",
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.warning("Feedback noted!")

# -----------------------------
# App header
# -----------------------------
st.markdown("""
<div class="app-header">
    <h1>MediRAG - AI Medical Assistant</h1>
    <div class="subtitle">Your intelligent medical information companion</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs: Chat + Metrics
# -----------------------------
tab_chat, tab_metrics = st.tabs(["Chat", "Metrics Dashboard"])

# -----------------------------
# CHAT TAB
# -----------------------------
with tab_chat:
    col_main, col_sidebar = st.columns([3, 1])

    with col_main:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div class="welcome-message">
                    <div class="welcome-content">
                        <h2> Welcome to MediRAG</h2>
                        <p>I'm your AI-powered medical assistant. Upload PDFs for personalized responses.</p>
                        <div class="feature-cards">
                            <div class="feature-card">
                                <h4> Drug Information</h4>
                                <p>Medication details, dosages, interactions.</p>
                            </div>
                            <div class="feature-card">
                                <h4> Health Guidance</h4>
                                <p>Evidence-based health advice.</p>
                            </div>
                            <div class="feature-card">
                                <h4> PDF Analysis</h4>
                                <p>Personalized insights from uploaded medical PDFs.</p>
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
                                <div class="message-avatar user-avatar">User</div>
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
                        col1, col2 = st.columns([0.1, 0.1])
                        with col1:
                            if st.button("Is the responce relavent 👍", key=f"up_{i}"):
                                st.session_state.feedback.append({
                                        "index": i,
                                        "response": m["content"],
                                        "feedback": "The response is relevant"
                                    })
                                    # 🔥 Update logs + save to CSV
                                if st.session_state.get("logs"):
                                        st.session_state.logs[-1]["feedback"] = "Relevant response was provided"
                                        pd.DataFrame(st.session_state.logs).to_csv("logs/metrics.csv", index=False)

                        with col2:
                            if st.button("Bad response👎", key=f"down_{i}"):
                                st.session_state.feedback.append({
                                        "index": i,
                                        "response": m["content"],
                                        "feedback": "The response is not relevant"
                                    })
                                    # 🔥 Update logs + save to CSV
                                if st.session_state.get("logs"):
                                    st.session_state.logs[-1]["feedback"] = "Irrelevant response was provided"
                                    pd.DataFrame(st.session_state.logs).to_csv("logs/metrics.csv", index=False)


        # Fixed chat input
        st.markdown("""
        <div class="chat-input-container">
            <div class="input-wrapper"><div style="flex:1;"><div id="chat-input-placeholder"></div></div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", placeholder="Ask about the uploaded PDFs...", label_visibility="collapsed", key="message_input")
            submit_button = st.form_submit_button("Send", type="primary")

            st.markdown("""
            <script>
            // small JS - clone input into fixed area (kept from your original)
            document.addEventListener('DOMContentLoaded', function() {
                const input = document.querySelector('[data-testid="stTextInput"] input');
                const placeholder = document.getElementById('chat-input-placeholder');
                const form = document.querySelector('[data-testid="stForm"]');
                const submitButton = document.querySelector('[data-testid="stForm"] button[kind="primaryFormSubmit"]');
                if (input && placeholder) {
                    const inputClone = input.cloneNode(true);
                    inputClone.style.background='transparent'; inputClone.style.border='none'; inputClone.style.color='#ececf1';
                    inputClone.style.fontSize='1rem'; inputClone.style.outline='none'; inputClone.style.width='100%'; inputClone.style.padding='.5rem 0';
                    placeholder.appendChild(inputClone);
                    const sendBtn = document.createElement('button'); sendBtn.innerHTML='➤';
                    sendBtn.style.background='#19c37d'; sendBtn.style.border='none'; sendBtn.style.borderRadius='6px';
                    sendBtn.style.width='32px'; sendBtn.style.height='32px'; sendBtn.style.color='white'; sendBtn.style.cursor='pointer';
                    placeholder.parentElement.appendChild(sendBtn);
                    inputClone.addEventListener('input', function(){ input.value=this.value; input.dispatchEvent(new Event('input',{bubbles:true})); });
                    input.addEventListener('input', function(){ inputClone.value=this.value; });
                    inputClone.addEventListener('keydown', function(e){ if (e.key==='Enter'){ e.preventDefault(); if (this.value.trim()){ input.value=this.value; input.dispatchEvent(new Event('input',{bubbles:true})); submitButton.click(); } }});
                    sendBtn.addEventListener('click', function(e){ e.preventDefault(); if (inputClone.value.trim()){ input.value=inputClone.value; input.dispatchEvent(new Event('input',{bubbles:true})); submitButton.click(); }});
                    if (form) { form.style.display='none'; }
                    inputClone.focus();
                }
            });
            </script>
            """, unsafe_allow_html=True)

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
                            final_answer = "I couldn't find this information in the uploaded documents."
                            used_contexts = []
                        else:
                            # To be safe: if LLM used outside knowledge, judge or simple heuristic could detect.
                            # But we will respect LLM output except when it clearly says it couldn't find.
                            # Build deterministic sources list from retrieved contexts
                            sources = format_sources_from_docs(contexts)
                            if "I couldn't find" in answer or "couldn't find" in answer.lower():
                                final_answer = "I couldn't find this information in the uploaded documents."
                                used_contexts = []
                            else:
                                # append Sources block deterministically (not trusting LLM for citation)
                                source_text = "\n".join(f"- {s}" for s in sources) if sources else ""
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
                        st.session_state.messages.append({"role": "assistant", "content": "I encountered an error while processing your request."})
                st.rerun()

    # Sidebar: PDF upload + services health
    with col_sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="upload-section">
                <h4> Upload Medical PDF</h4>
                <p>Upload PDF documents. The assistant will answer using only uploaded PDFs and show sources (file + page).</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], help="Upload medical PDFs", label_visibility="collapsed")

        if uploaded_file is not None:
            with st.spinner("Processing your PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        pdf_path = tmp_file.name

                    pdf_name = uploaded_file.name
                    pages = extract_text_pages(pdf_path, pdf_name)

                    if not pages:
                        st.markdown('<div class="status-message error-message">No text could be extracted from PDF</div>', unsafe_allow_html=True)
                    else:
                        all_texts, all_metas = [], []
                        for p in pages:
                            texts, metas = chunk_page_text(p, chunk_size=800)
                            all_texts.extend(texts)
                            all_metas.extend(metas)

                        if services['status'] == 'healthy':
                            # fit BM25 and re-create retriever (keeps index same but uses new sparse encoder)
                            services['bm25_encoder'].fit(all_texts)
                            services['bm25_encoder'].dump(services['bm25_path'])

                            services['retriever'] = PineconeHybridSearchRetriever(
                                embeddings=services['embeddings'],
                                sparse_encoder=services['bm25_encoder'],
                                index=services['index']
                            )

                            # Add texts to the index with metadata so we can cite file+page
                            # retriever.add_texts should accept (texts, metadatas)
                            services['retriever'].add_texts(texts=all_texts, metadatas=all_metas)

                            st.markdown('<div class="status-message success-message">PDF processed and indexed successfully!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-message error-message">Service unavailable for PDF processing</div>', unsafe_allow_html=True)

                    os.unlink(pdf_path)
                except Exception as e:
                    st.markdown(f'<div class="status-message error-message">Error processing PDF: {e}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_answer_idx = None
            st.rerun()

        st.markdown("---")
        if services.get('status') == 'healthy':
            st.markdown("""<div class="health-indicator"><div class="health-dot health-healthy"></div><div class="health-text">All services online</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="health-indicator"><div class="health-dot health-error"></div><div class="health-text">Service issues: {services.get('error')}</div></div>""", unsafe_allow_html=True)

# -----------------------------
# METRICS DASHBOARD TAB
# -----------------------------
with tab_metrics:
    st.title("📊 RAG Evaluation Dashboard")

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/metrics.csv"
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
        st.info("No evaluation data yet. Start chatting to generate metrics.")
    else:
        for col in ["faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "eval_latency_sec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.write("### Recent Interactions")
        show_cols = ["time", "question", "faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "eval_latency_sec", "feedback"]
        st.dataframe(df[show_cols].tail(15), use_container_width=True)

        st.write("### Trends")
        if {"faithfulness","answer_relevancy","context_relevancy"}.issubset(df.columns):
            st.line_chart(df[["faithfulness", "answer_relevancy", "context_relevancy"]])

        st.write("### Averages")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faithfulness (avg)", f"{df['faithfulness'].mean():.2f}" if "faithfulness" in df else "—")
        col2.metric("Answer Relevancy (avg)", f"{df['answer_relevancy'].mean():.2f}" if "answer_relevancy" in df else "—")
        col3.metric("Context Relevancy (avg)", f"{df['context_relevancy'].mean():.2f}" if "context_relevancy" in df else "—")
        if "gen_latency_sec" in df:
            col4.metric("Gen Latency (avg s)", f"{df['gen_latency_sec'].mean():.2f}")
        else:
            col4.metric("Gen Latency (avg s)", "—")

        st.write("### Judge Reasons (last 5)")
        if "judge_reasons" in df:
            for _, row in df.tail(5).iterrows():
                with st.expander(f"{row.get('time','')} — {row.get('question','')[:60]}"):
                    try:
                        reasons = json.loads(row.get("judge_reasons","{}"))
                    except Exception:
                        reasons = {"raw": row.get("judge_reasons")}
                    st.json(reasons)

