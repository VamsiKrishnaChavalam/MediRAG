import os
import re
import openai
import tempfile
import pytesseract
import pinecone
import streamlit as st
import nltk
from pdf2image import convert_from_path
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import json
from datetime import datetime

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

# Page configuration
st.set_page_config(
    page_title="MediRAG - AI Medical Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the dark theme from HTML
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
        width: 30px;
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
</style>
""", unsafe_allow_html=True)

# Initialize configuration and clients
@st.cache_resource
def initialize_services():
    """Initialize all services with error handling"""
    try:
        # Configuration
        LLM_API_KEY = "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"
        PINECONE_API_KEY = "pcsk_bGiUC_6wqctEnz2zymTRVFs7dKzhF5xQGnMwsQh8fnwqNAmKpNBntzAwFeL4WmoDiDFJc"
        PINECONE_INDEX_NAME = "medirag"
        HF_TOKEN = "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"
        os.environ["HF_TOKEN"] = HF_TOKEN
        
        # Initialize clients
        llm_client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=LLM_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load or create BM25 encoder
        bm25_path = "rag-veda.json"
        if os.path.exists(bm25_path):
            bm25_encoder = BM25Encoder().load(bm25_path)
        else:
            bm25_encoder = BM25Encoder()
        
        # Initialize retriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index
        )
        
        return {
            'llm': llm_client,
            'retriever': retriever,
            'embeddings': embeddings,
            'bm25_encoder': bm25_encoder,
            'index': index,
            'bm25_path': bm25_path,
            'status': 'healthy'
        }
    except Exception as e:
        return {
            'llm': None,
            'retriever': None,
            'embeddings': None,
            'bm25_encoder': None,
            'index': None,
            'bm25_path': None,
            'status': 'error',
            'error': str(e)
        }

# Helper functions
def extract_text_using_ocr(pdf_path):
    """Extract text from PDF using OCR"""
    try:
        images = convert_from_path(pdf_path)
        return "\n".join(clean_text(pytesseract.image_to_string(img)) for img in images)
    except Exception as e:
        st.error(f"OCR extraction error: {e}")
        return ""

def clean_text(text):
    """Clean and normalize text"""
    return re.sub(r"\s+", " ", text).strip()

def split_text_into_chunks(text, chunk_size=300):
    """Split text into chunks for processing"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def prepare_rag_prompt(query, message_history=None, retriever=None, top_n=3):
    """Prepare RAG prompt with context from retriever"""
    if not retriever:
        return f"Question: {query}\n\nI don't have access to my knowledge base right now. Please try again later."
    
    try:
        top_docs = retriever.invoke(query)
        top_contents = [doc.page_content for doc in top_docs[:top_n]]
        
        context_blocks = "\n\n".join(
            [f"Context {i+1}:\n{content}" for i, content in enumerate(top_contents)]
        )

        # Include recent chat history
        history_text = ""
        if message_history:
            recent_history = message_history[-5:]  # Last 5 messages
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        return f"""
You are a knowledgeable medical assistant tasked with answering user queries clearly and accurately. You provide preventions and diet plans when requested.

### Chat History:
{history_text}

### Knowledge Base Context:
{context_blocks if context_blocks else "No relevant documents were found."}

Now, based on the above context and chat history, answer the following question:

Question: {query}


Guidelines:
0) Instruction Hierarchy (in case of conflicts)

Safety rules (emergency handling, controlled substances, legal/ethical).

PDF-citations policy (cite only trained PDFs by name + page).

Drug-information focus (don't over-ask unless personalization affects safety).

Core behavior (greeting, info collection, personalization).

Style & format (concise, clear, actionable).

Fallback policy (strict PDF-only mode vs hybrid; configurable).

1) Role & Scope

Act as a medical information assistant specialized in drug information (uses, side effects, precautions, interactions, dosage ranges) and secondarily disease/procedure explanations.

You are not a doctor; you give evidence-based information and safe guidance, not diagnoses or prescriptions.

2) Greeting

If user greets: reply warmly and only ask: "How may I help you? Do have any drug related pdf, please upload!"

Do not ask for personal info at this stage.

dont greet the user for every query.

Do not greet if user query is medical or drug related or a general query.

3) When to Ask Questions (Minimalist Policy)
A) Drug-only queries (general info)

If user asks "What is <drug>?", "Side effects of <drug>?", "What is it used for?"
Answer directly. Do not ask age/weight/history.

Include: uses, common side effects, serious side effects (red flags), precautions, contraindications, high-level dosage guidance (standard adult unless otherwise stated), interactions overview, storage, when to seek care think like a genAI and answer this.

B) Personalized suitability or dosing

If the user asks "Can I take <drug>?", "What dose should I take?", "Is it safe with my meds?" ,"I'm having <illness> what should i do?", "What is the diet plan for <illness>?", "What are the preventions for <illness>?" or similar:
Ask only the minimum necessary to provide safe, personalized advice.
Ask the following questions to personalize the answer:

Age

Gender

If female ask about Pregnancy/breastfeeding status (if potentially relevant)

Current conditions (esp. kidney/liver, heart disease, diabetes, glaucoma, GI issues)

Current medications & supplements (including recent antibiotics/antifungals, MAOIs)

Known allergies (especially to the drug/class)

Weight only if pediatric or weight-based dosing (e.g., mg/kg)

Ask nothing else unless strictly necessary for safety (e.g., eGFR when renal dosing adjustments are PDF-specified).

C) Disease/procedure queries

Ask age, gender, conditions, meds only if the user requests personalized advice.

Otherwise provide general, educational information.

4) Core Content Requirements

For each answer, include—when relevant:

Direct answer first (one-paragraph summary).

Details (structured bullets):

For drugs:

Uses/indications (on-label; state off-label only if present in PDFs).

Dosage: typical adult dose; pediatric mg/kg when applicable; max dose; routes; frequency; duration if applicable.

Adjustments: renal/hepatic dose changes (include thresholds if PDFs specify: e.g., CrCl/eGFR cutoffs; Child-Pugh).

Contraindications (absolute vs relative if PDFs distinguish).

Warnings/Black-box: call out clearly if present in PDFs.

Common side effects vs serious side effects (red-flag symptoms → stop and seek care).

Drug interactions: major classes and notable pairs; timing (e.g., antacids, MAOIs, SSRIs, grapefruit).

Special populations: pregnancy, lactation, geriatrics, pediatrics (age cutoffs).

Monitoring: what to watch (BP, HR, INR, LFTs, renal function) if applicable.

Administration: with/without food, time of day, do not crush/chew, storage.

For diseases: symptoms, causes, differentials (optional), evaluation overview, treatments (lifestyle, OTC, Rx categories), prevention.

For procedures: what it is, preparation, steps (plain language), risks, benefits, recovery, alternatives.

Actionable next steps (simple to follow).

Disclaimer (only when you mention medication use, dosing, diagnosis, or urgent concerns).

PDF citations (see Section 6).

5) Safety Guardrails (Hard Rules)

Emergencies (e.g., chest pain, stroke signs, anaphylaxis, severe breathing trouble, suicidal ideation, overdose):
→ "This seems urgent. Please seek emergency medical help immediately by calling your local emergency number."

Controlled substances / illegal requests: Do not help procure, dose, or optimize misuse. You may give high-level safety info (e.g., dependence risk) but refuse prescribing/obtaining help.

Pediatric dosing: If age/weight missing for weight-based drugs, do not provide a personal dose. Provide general pediatric framework and ask for the needed info.

Pregnancy/Lactation: If advice depends on status and it's unknown, highlight that status matters and provide general safety info only; suggest clinician confirmation.

Renal/Hepatic impairment: If PDFs specify thresholds (e.g., eGFR <30), do not personalize without that info; either ask or present general caution.

Do not diagnose; present information and recommend seeing a clinician for diagnosis/treatment decisions.

No definitive treatment plans for serious conditions; provide categories/options and urge clinician involvement.

No speculation beyond PDFs (for facts requiring citation). See fallback policy.

6) Citations & Plagiarism (PDF-Only)

Cite only the PDF(s) used to generate the answer. Format:

(PDF_Name.pdf, p. XX) or (PDF_Name.pdf) if no page.

If multiple PDFs contributed: list all, separated by semicolons.

Never cite websites or generic sources in responses.

Plagiarism control: paraphrase; if quoting, keep to ≤25 words and add quotes plus page citation.

If no relevant PDF content exists:

Strict PDF-only mode (recommended for compliance): say
"I don't find this in the provided PDFs. I can share general educational info if you allow responses without PDF citations."

Hybrid mode (if you enable it): provide clearly marked general info without citation (or with a generic note "No matching PDF source found"). Choose one mode for your deployment.

7) Dosage Rules (to reduce errors)

Show standard units (mg, mcg, g, mL). Convert lb→kg internally (1 kg = 2.20462 lb) before mg/kg math.

Always state maximum daily dose if PDFs contain it.

Round pediatric doses as PDFs direct (e.g., to nearest 2.5 mg or mL). If not specified, round conservatively and note "rounded per typical practice."

Time-based dosing: specify interval (q4–6h), duration (e.g., 3–5 days), and do not exceed statement if in PDFs.

Adjustments: if PDFs include renal/hepatic guidance, include explicit thresholds and example adjusted doses.

Calculation hygiene: (internal) compute step-by-step; (external) optionally include a brief "Calculation" line when helpful, especially pediatric mg/kg.

8) Follow-Up Question Triggers

Ask only when these change safety/dosing:

Pediatric dosing → age & weight.

Pregnancy/lactation → if drug has fetal/neonatal risk.

Renal/hepatic impairment mentioned or suspected from context.

High-risk interactions likely (e.g., MAOIs, warfarin, linezolid, tramadol + SSRIs).

Prior severe allergy/anaphylaxis to same drug/class.

9) Refusals (with redirect)

Refuse to help with: forging prescriptions, sourcing controlled substances, unsafe combinations for self-harm, detailed "how to misuse" instructions.

Redirect to: general safety info, support resources, clinician evaluation, or emergency services.

10) Output Format (Consistent, Parse-friendly)

Default human-readable format; optionally pair with JSON (below).

Answer layout:

Summary (2–3 sentences).

Details (sectioned bullets per content requirements).

What to do next (numbered steps).

Disclaimer (only when needed).

Citations: (PDF_Name.pdf, p. X; Other_PDF.pdf, p. Y) on a single line.



""".strip()
    except Exception as e:
        st.error(f"RAG preparation error: {e}")
        return f"Question: {query}\n\nI'm having trouble accessing my knowledge base. Please try again."

def get_llm_response(query, message_history=None, services=None):
    """Get response from LLM with RAG context"""
    if not services or not services['llm']:
        return "I'm sorry, but the AI service is currently unavailable. Please try again later."
    
    try:
        rag_prompt = prepare_rag_prompt(query, message_history, services['retriever'])
        
        response = services['llm'].chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": rag_prompt}],
            max_tokens=3000,
            temperature=0.8,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"LLM response error: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."

# Initialize services
services = initialize_services()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""

# App header
st.markdown("""
<div class="app-header">
    <h1> MediRAG - AI Medical Assistant</h1>
    <div class="subtitle">Your intelligent medical information companion</div>
</div>
""", unsafe_allow_html=True)

# Main layout
col_main, col_sidebar = st.columns([3, 1])

with col_main:
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display welcome message if no conversation
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
            # Display chat messages
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="message-row user-message-row">
                        <div class="message-content">
                            <div class="message-avatar user-avatar">U</div>
                            <div class="message-text">
                                <p>{message["content"]}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    formatted_content = message["content"].replace('**', '').replace('*', '').replace('\n\n', '</p><p>').replace('\n', '<br>')
                    if not formatted_content.startswith('<p>'):
                        formatted_content = f'<p>{formatted_content}</p>'
                    
                    st.markdown(f"""
                    <div class="message-row bot-message-row">
                        <div class="message-content">
                            <div class="message-avatar bot-avatar">AI</div>
                            <div class="message-text">
                                {formatted_content}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Input area - Fixed at bottom
    input_container = st.container()
    
    # Create a custom input area that stays fixed
    st.markdown("""
    <div class="chat-input-container">
        <div class="input-wrapper">
            <div style="flex: 1;">
                <div id="chat-input-placeholder"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message",
            placeholder="Ask me anything about medicine, health, or upload a PDF for personalized responses...",
            label_visibility="collapsed",
            key="message_input"
        )
        
        # Hidden submit button (we'll trigger it with JavaScript)
        submit_button = st.form_submit_button("Send", type="primary")
        
        # JavaScript to handle Enter key and move input to fixed position
        st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const input = document.querySelector('[data-testid="stTextInput"] input');
            const placeholder = document.getElementById('chat-input-placeholder');
            const form = document.querySelector('[data-testid="stForm"]');
            const submitButton = document.querySelector('[data-testid="stForm"] button[kind="primaryFormSubmit"]');
            
            if (input && placeholder) {
                // Move input to fixed position
                const inputClone = input.cloneNode(true);
                inputClone.style.background = 'transparent';
                inputClone.style.border = 'none';
                inputClone.style.color = '#ececf1';
                inputClone.style.fontSize = '1rem';
                inputClone.style.outline = 'none';
                inputClone.style.width = '100%';
                inputClone.style.padding = '0.5rem 0';
                
                placeholder.appendChild(inputClone);
                
                // Add send button to fixed input
                const sendBtn = document.createElement('button');
                sendBtn.innerHTML = '➤';
                sendBtn.style.background = '#19c37d';
                sendBtn.style.border = 'none';
                sendBtn.style.borderRadius = '6px';
                sendBtn.style.width = '32px';
                sendBtn.style.height = '32px';
                sendBtn.style.color = 'white';
                sendBtn.style.cursor = 'pointer';
                sendBtn.style.display = 'flex';
                sendBtn.style.alignItems = 'center';
                sendBtn.style.justifyContent = 'center';
                sendBtn.style.flexShrink = '0';
                sendBtn.style.transition = 'all 0.2s';
                
                sendBtn.addEventListener('mouseenter', function() {
                    this.style.background = '#10a37f';
                });
                sendBtn.addEventListener('mouseleave', function() {
                    this.style.background = '#19c37d';
                });
                
                placeholder.parentElement.appendChild(sendBtn);
                
                // Sync input values
                inputClone.addEventListener('input', function() {
                    input.value = this.value;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                });
                
                input.addEventListener('input', function() {
                    inputClone.value = this.value;
                });
                
                // Handle Enter key to send message
                inputClone.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        if (this.value.trim()) {
                            input.value = this.value;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            submitButton.click();
                        }
                    }
                });
                
                // Handle send button click
                sendBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    if (inputClone.value.trim()) {
                        input.value = inputClone.value;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        submitButton.click();
                    }
                });
                
                // Hide original form
                if (form) {
                    form.style.display = 'none';
                }
                
                // Focus on the cloned input
                inputClone.focus();
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        if submit_button and user_input.strip():
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            
            # Generate response
            with st.spinner(" Analyzing your question..."):
                try:
                    response = get_llm_response(user_input.strip(), st.session_state.messages, services)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = "I apologize, but I encountered an error while processing your request. Please try again or upload a relevant PDF for better assistance."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

# Sidebar
with col_sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <div class="upload-section">
            <h4> Upload Medical PDF</h4>
            <p>Upload medical documents, drug information, or research papers.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload medical PDFs to enhance the AI's knowledge base",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        with st.spinner(" Processing your PDF..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    pdf_path = tmp_file.name
                
                # Extract text
                extracted_text = extract_text_using_ocr(pdf_path)
                
                if not extracted_text.strip():
                    st.markdown('<div class="status-message error-message"> No text could be extracted from PDF</div>', unsafe_allow_html=True)
                else:
                    # Process text
                    chunks = split_text_into_chunks(extracted_text)
                    
                    # Update BM25 encoder and retriever
                    if services['status'] == 'healthy':
                        services['bm25_encoder'].fit(chunks)
                        services['bm25_encoder'].dump(services['bm25_path'])
                        
                        # Update retriever
                        services['retriever'] = PineconeHybridSearchRetriever(
                            embeddings=services['embeddings'],
                            sparse_encoder=services['bm25_encoder'],
                            index=services['index']
                        )
                        services['retriever'].add_texts(chunks)
                        
                        st.markdown('<div class="status-message success-message"> PDF processed successfully!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-message error-message">Service unavailable for PDF processing</div>', unsafe_allow_html=True)
                
                # Clean up
                os.unlink(pdf_path)
                
            except Exception as e:
                st.markdown('<div class="status-message error-message"> Error processing PDF</div>', unsafe_allow_html=True)
    
    # Clear chat button
    st.markdown("---")
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Health status
    st.markdown("---")
    if services['status'] == 'healthy':
        st.markdown("""
        <div class="health-indicator">
            <div class="health-dot health-healthy"></div>
            <div class="health-text">All services online</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="health-indicator">
            <div class="health-dot health-error"></div>
            <div class="health-text">Service issues detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional info
    st.markdown("---")
    with st.expander(" About MediRAG"):
        st.markdown("""
        **MediRAG** is an AI-powered medical assistant that provides:
        
        -  Drug information and interactions
        -  Health guidance and advice
        -  PDF document analysis
        -  Evidence-based medical information
        
        **Disclaimer:** This tool provides information only and should not replace professional medical advice.
        """)

# Auto-scroll to bottom when new messages are added
if st.session_state.messages:
    st.markdown("""
    <script>
    var chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)