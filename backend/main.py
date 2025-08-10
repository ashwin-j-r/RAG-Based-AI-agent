import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
from dotenv import load_dotenv
load_dotenv() 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiting and app
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Initialize models
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel(model_name="gemini-2.0-flash")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

# In-memory storage for sessions
session_store: Dict[str, List[Dict[str, str]]] = {}

class Query(BaseModel):
    text: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history: List[Dict[str, str]]

# Knowledge base
FAISS_INDEX: Optional[faiss.Index] = None
TEXT_CHUNKS: List[str] = []

@app.on_event("startup")
async def load_knowledge_base():
    global FAISS_INDEX, TEXT_CHUNKS
    
    try:
        pdf_path = os.getenv("KNOWLEDGE_PDF", "Knowledge.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        text = extract_text_from_pdf(pdf_path)
        TEXT_CHUNKS = split_into_chunks(text)
        
        if os.path.exists("faiss_index.index"):
            FAISS_INDEX = faiss.read_index("faiss_index.index")
            logger.info("Loaded existing FAISS index")
        else:
            embeddings = create_embeddings(TEXT_CHUNKS)
            FAISS_INDEX = create_faiss_index(embeddings)
            faiss.write_index(FAISS_INDEX, "faiss_index.index")
            logger.info("Created new FAISS index")
            
        logger.info(f"Loaded knowledge base with {len(TEXT_CHUNKS)} chunks")
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {str(e)}")
        raise

@app.post("/query")
@limiter.limit("10/minute")
async def handle_query(request: Request, query: Query) -> JSONResponse:
    try:
        if FAISS_INDEX is None:
            raise HTTPException(status_code=503, detail="Knowledge base not loaded")
        
        session_id = query.session_id or str(uuid.uuid4())
        chat_history = get_chat_history(session_id)
        
        answer = await generate_rag_response(query.text, chat_history)
        
        new_entry = {
            "user": query.text,
            "bot": answer,
            "timestamp": datetime.now().isoformat()
        }
        save_to_history(session_id, new_entry)
        
        return JSONResponse({
            "answer": answer,
            "session_id": session_id,
            "history": get_chat_history(session_id)
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500
        )

# Helper functions with improved error handling
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(chunks: List[str]) -> np.ndarray:
    return EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index

def retrieve_relevant_chunks(query: str, k: int = 3) -> List[str]:
    query_embedding = EMBEDDING_MODEL.encode([query])
    _, indices = FAISS_INDEX.search(np.array(query_embedding).astype('float32'), k)
    return [TEXT_CHUNKS[i] for i in indices[0] if i < len(TEXT_CHUNKS)]

async def generate_rag_response(query: str, history: Optional[List[Dict]] = None) -> str:
    try:
        relevant_chunks = retrieve_relevant_chunks(query)
        context = "\n\n".join(relevant_chunks)
        
        history_context = ""
        if history:
            history_context = "\nPrevious conversation:\n" + \
                "\n".join(f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:])
        
        prompt = f"""You are a helpful assistant. Use the following context to answer questions.
        
        Context:
        {context}
        
        {history_context}
        
        Current Question: {query}
        
        Answer concisely and helpfully:"""
        
        response = await GEMINI_MODEL.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I couldn't generate a response. Please try again."

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    return session_store.get(session_id, [])

def save_to_history(session_id: str, entry: Dict[str, str]):
    if session_id not in session_store:
        session_store[session_id] = []
    session_store[session_id].append(entry)