"""
LLM integration module for Google Gemini AI.
Handles query processing and response generation.
"""

import os
import logging
from typing import Optional
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles Google Gemini AI integration for question answering."""
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.warning("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Gemini AI client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Error initializing Gemini AI: {e}")
                self.client = None
    
    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate response using Google Gemini AI.
        
        Args:
            query: User question
            context: Retrieved context from PDFs
            
        Returns:
            Generated response
        """
        if not self.client:
            return "I don't know, please check with the admin. (Gemini AI not configured)"
        
        try:
            # Prepare the prompt
            if context:
                prompt = f"""You are a helpful assistant for an educational institution. Answer the user's question based on the provided context from institutional documents.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based on the provided context
2. If the answer is not available in the context, respond with: "I don't know, please check with the admin."
3. Be helpful, accurate, and concise
4. If the context contains relevant information, provide a clear and informative answer

Answer:"""
            else:
                prompt = f"""You are a helpful assistant for an educational institution. 

Question: {query}

Since no relevant context was found in the institutional documents, please respond with: "I don't know, please check with the admin."

Answer:"""
            
            # Make API call
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                )
            )
            
            answer = response.text.strip()
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I don't know, please check with the admin. (Error occurred)"
    
    def is_configured(self) -> bool:
        """
        Check if Gemini AI is properly configured.
        
        Returns:
            True if configured, False otherwise
        """
        return self.client is not None and self.api_key is not None

# Global LLM handler instance
llm_handler = LLMHandler()
