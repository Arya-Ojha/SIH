"""
Simplified multilingual WhatsApp chatbot backend.
This version works with minimal dependencies for demonstration.
"""

import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual WhatsApp Chatbot",
    description="A simplified multilingual chatbot for demonstration",
    version="1.0.0"
)

# Simple language detection (basic implementation)
def detect_language_simple(text: str) -> str:
    """Simple language detection based on character patterns."""
    text_lower = text.lower()
    
    # Check for Hindi characters (Devanagari script)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return 'hi'
    
    # Check for Bengali characters
    if any('\u0980' <= char <= '\u09FF' for char in text):
        return 'bn'
    
    # Default to English
    return 'en'

# Simple translation (mock implementation)
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Mock translation function - returns original text with language info."""
    if source_lang == target_lang:
        return text
    
    # For demo purposes, just add language info
    return f"[{source_lang}->{target_lang}] {text}"

# Simple knowledge base
KNOWLEDGE_BASE = {
    "admission": "To be eligible for admission, students need a high school diploma, minimum GPA of 3.0, SAT score of 1200+, and completed application materials.",
    "tuition": "Undergraduate tuition is $15,000 per semester for full-time students. Graduate tuition is $18,000 per semester.",
    "requirements": "Admission requirements include: high school diploma, minimum GPA 3.0, SAT 1200+, ACT 26+, application form, transcripts, letters of recommendation, personal statement, and $50 application fee.",
    "deadline": "Fall admission deadline is January 15 for regular decision. Early decision deadline is November 1.",
    "contact": "Admissions office phone: (555) 123-4567, email: admissions@university.edu. Office hours: Monday-Friday, 9:00 AM - 5:00 PM."
}

def search_knowledge_base(query: str) -> str:
    """Simple keyword-based search in knowledge base."""
    query_lower = query.lower()
    
    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            return answer
    
    return "I don't know, please check with the admin."

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Multilingual WhatsApp Chatbot API",
        "status": "running",
        "version": "1.0.0-simplified"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "supported_languages": ["en", "hi", "bn"]
    }

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint that processes incoming messages.
    
    Expected payload format:
    {
        "message": "user message text",
        "from": "user_phone_number",
        "timestamp": "2024-01-01T00:00:00Z"
    }
    """
    try:
        # Parse request body
        body = await request.json()
        logger.info(f"Received webhook: {body}")
        
        # Extract message
        user_message = body.get("message", "").strip()
        user_id = body.get("from", "unknown")
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "No message provided"}
            )
        
        logger.info(f"Processing message from {user_id}: {user_message}")
        
        # Step 1: Detect language
        detected_lang = detect_language_simple(user_message)
        logger.info(f"Detected language: {detected_lang}")
        
        # Step 2: Translate to English for processing (mock)
        english_query = translate_text(user_message, detected_lang, 'en')
        logger.info(f"Processed query: {english_query}")
        
        # Step 3: Search knowledge base
        english_response = search_knowledge_base(english_query)
        logger.info(f"Found response: {english_response}")
        
        # Step 4: Translate response back to original language (mock)
        final_response = translate_text(english_response, 'en', detected_lang)
        logger.info(f"Final response: {final_response}")
        
        # Return response
        return JSONResponse(content={
            "response": final_response,
            "original_language": detected_lang,
            "language_name": {"en": "English", "hi": "Hindi", "bn": "Bengali"}.get(detected_lang, "English"),
            "context_found": english_response != "I don't know, please check with the admin.",
            "user_id": user_id
        })
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )

@app.get("/search")
async def search_endpoint(query: str):
    """Search endpoint for testing."""
    try:
        result = search_knowledge_base(query)
        
        return JSONResponse(content={
            "query": query,
            "result": result,
            "found": result != "I don't know, please check with the admin."
        })
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Search error", "message": str(e)}
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {
            "status": "running",
            "knowledge_base_size": len(KNOWLEDGE_BASE),
            "supported_languages": ["en", "hi", "bn"],
            "version": "1.0.0-simplified"
        }
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Stats error", "message": str(e)}
        )

if __name__ == "__main__":
    # Set up environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting simplified server on {host}:{port}")
    
    # Run the application
    uvicorn.run(
        "app_simple:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
