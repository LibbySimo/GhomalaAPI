from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import time
import logging
import requests
from pathlib import Path
import uuid
import PyPDF2
from pydantic import BaseModel
from typing import Optional
import io
from data import dictionary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configurationgive me a short summary in ghomala
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf"}
XAI_API_KEY = os.getenv("XAI_API_KEY", "your_xai_api_key_here")  # Replace with your actual API key
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3"

# Global variable to store dictionary content
dictionary_content = ""


# Pydantic models
class QueryRequest(BaseModel):
    file_path: str
    question: str
    response_language: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    if not filename:
        return False
    return Path(filename.lower()).suffix in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def load_dictionary():
    """Load the dictionary content directly from base64 - just like the working example."""
    global dictionary_content
    try:
        # Decode the dictionary directly - same as working example
        dictionary_content = base64.b64decode(dictionary).decode('utf-8', errors='ignore')
        logger.info(f"Successfully loaded dictionary: {len(dictionary_content)} characters")
        return dictionary_content
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
        return "Dictionary unavailable."

def generate_with_retry(model, messages, api_key, max_retries=3, base_delay=1):
    """Generate content with retry logic - same as working example."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1} of {max_retries + 1}")
            payload = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "text"}
            }

            response = requests.post(XAI_API_URL, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (429, 500):
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Server error ({e.response.status_code}) encountered. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries reached. Server error ({e.response.status_code}) persists.")
                    raise HTTPException(status_code=e.response.status_code, detail="xAI API server error")
            else:
                logger.error(f"HTTP error with status code {e.response.status_code}: {e}")
                raise HTTPException(status_code=e.response.status_code, detail=f"API error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def detect_language(text: str) -> str:
    """Simple language detection."""
    if not text:
        return "English"

    text_lower = text.lower()

    # Ghomala' indicators
    ghomala_words = ["ndzí", "mbə", "nkə", "tə", "bə", "ghomala", "ghomàla", "púsi", "bap", "nòm", "yəŋ", "pú", "fà'"]
    if any(word in text_lower for word in ghomala_words):
        return "Ghomala'"

    # French indicators
    french_words = ["est-ce", "comment", "pourquoi", "qu'est-ce", "où", "quand", "qui", "quoi", "le", "la", "les", "de", "du", "des", "français"]
    if any(word in text_lower for word in french_words):
        return "French"

    return "English"

@app.on_event("startup")
async def startup_event():
    """Load dictionary on startup."""
    load_dictionary()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dictionary_loaded": len(dictionary_content) > 0,
        "supported_languages": ["English", "French", "Ghomala'"]
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(None)):
    """Handle file uploads."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
            f.write(content)

        extracted_text = extract_text_from_pdf(file_path)
        detected_language = detect_language(extracted_text)

        return JSONResponse(
            status_code=201,
            content={
                "document_id": filename,
                "status": "success",
                "file_path": file_path,
                "detected_language": detected_language,
                "text_length": len(extracted_text)
            }
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/query")
async def query(request: QueryRequest):
    """Handle queries about uploaded documents."""
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found")

    try:
        # Extract document content
        document_text = extract_text_from_pdf(request.file_path)

        # Determine response language
        response_language = request.response_language or detect_language(request.question)

        # Create messages for multilingual fine-tuned model
        messages = [
            {
                "role": "system",
                "content": f"""You are a multilingual AI assistant fluent in Ghomala', French, and English. You have been fine-tuned to understand and respond naturally in all three languages. Always respond in the requested language ({response_language}) unless specifically asked otherwise.

Reference vocabulary and translations:
{dictionary_content}

Use this knowledge naturally in your responses without mentioning it explicitly."""
            },
            {
                "role": "user",
                "content": f"""Here is a document:\n\n{document_text}\n\nPlease answer this question in {response_language}: {request.question}"""
            }
        ]

        # Generate response
        answer = generate_with_retry(MODEL, messages, XAI_API_KEY)

        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "response_language": response_language,
            "document_language": detect_language(document_text)
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/translate")
async def translate(request: TranslationRequest):
    """Direct translation endpoint."""
    try:
        # Create translation messages for multilingual model
        messages = [
            {
                "role": "system",
                "content": f"""You are a multilingual AI assistant fluent in Ghomala', French, and English. You have been fine-tuned on multilingual data and can translate naturally between these languages. Provide accurate translations while maintaining the original meaning and cultural context.

Reference vocabulary and translations:
{dictionary_content}

Use this knowledge naturally in your translations without mentioning it explicitly."""
            },
            {
                "role": "user",
                "content": f"""Translate the following text from {request.source_language} to {request.target_language}:\n\n{request.text}"""
            }
        ]

        # Generate translation
        translation = generate_with_retry(MODEL, messages, XAI_API_KEY)

        return {
            "status": "success",
            "original_text": request.text,
            "translation": translation,
            "source_language": request.source_language,
            "target_language": request.target_language
        }

    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/api/dictionary-lookup/{word}")
async def dictionary_lookup(word: str, target_language: str = "French"):
    """Look up a word in the dictionary."""
    try:
        messages = [
            {
                "role": "system",
                "content": f"""You are a multilingual AI assistant with expertise in Ghomala', French, and English. You have been fine-tuned to understand and translate between these languages naturally. Provide the translation for the requested word in {target_language}.

Reference vocabulary and translations:
{dictionary_content}

Use this knowledge naturally without mentioning it explicitly."""
            },
            {
                "role": "user",
                "content": f"""What is the translation of "{word}" in {target_language}? If you don't know this word, please say "Translation not available"."""
            }
        ]

        result = generate_with_retry(MODEL, messages, XAI_API_KEY)

        return {
            "status": "success",
            "word": word,
            "translation": result,
            "target_language": target_language
        }

    except Exception as e:
        logger.error(f"Error in dictionary lookup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dictionary lookup error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dictionary_loaded": len(dictionary_content) > 0,
        "supported_languages": ["English", "French", "Ghomala'"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
