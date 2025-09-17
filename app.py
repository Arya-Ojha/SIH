"""
WhatsApp Bot with RAG, TTS, STT, and Multilingual Support
Refactored for better code organization and enhanced logging
"""

import os
import base64
import tempfile
import urllib.request
from typing import Optional, Tuple
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from langdetect import detect
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

# =====================================================
# CONFIGURATION & INITIALIZATION
# =====================================================

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        load_dotenv()
        print("🔧 Loading configuration...")
        
        # Twilio Configuration
        self.ACCOUNT_SID = os.getenv("ACCOUNT_SID")
        self.AUTH_TOKEN = os.getenv("AUTH_TOKEN")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
        
        # API Keys
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        
        # ElevenLabs Configuration
        self.VOICE_ID = "CZdRaSQ51p0onta4eec8"  # Adam voice
        self.TTS_MODEL = "eleven_multilingual_v2"
        self.STT_MODEL = "Scribe_v1"
        
        # RAG Configuration
        self.KNOWLEDGE_FILE = "knowledge_base.txt"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.LLM_MODEL = "gemini-2.0-flash-exp"
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # Language Configuration
        self.HINGLISH_WORDS = ['hai', 'hain', 'kya', 'aap', 'main', 'kar', 'kaise', 'kahan', 'kab', 'kyun']
        self.MISDETECTED_LANGS = ["sw", "id", "ms", "so", "af", "tl"]
        
        print("✅ Configuration loaded successfully")
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration"""
        print("🔍 Validating configuration...")
        
        required_vars = [
            ('ACCOUNT_SID', self.ACCOUNT_SID),
            ('AUTH_TOKEN', self.AUTH_TOKEN),
            ('TWILIO_WHATSAPP_NUMBER', self.TWILIO_WHATSAPP_NUMBER),
            ('GOOGLE_API_KEY', self.GOOGLE_API_KEY),
            ('ELEVENLABS_API_KEY', self.ELEVENLABS_API_KEY)
        ]
        
        missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
        else:
            print("✅ All required environment variables are present")

# Initialize configuration
config = Config()

# Initialize clients
print("🔗 Initializing external service clients...")
twilio_client = Client(config.ACCOUNT_SID, config.AUTH_TOKEN)
elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
print("✅ Clients initialized successfully")

# Global variables for RAG system
vector_store = None
qa_chain = None

# =====================================================
# RAG SYSTEM MANAGEMENT
# =====================================================

class RAGSystem:
    """Handles RAG system initialization and queries"""
    
    @staticmethod
    def initialize() -> bool:
        """Initialize the RAG system with knowledge base"""
        global vector_store, qa_chain
        
        print("🚀 Starting RAG system initialization...")
        
        try:
            # Initialize LLM
            print(f"🤖 Initializing {config.LLM_MODEL} language model...")
            llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL,
                temperature=0.7,
                max_tokens=1024
            )
            print("✅ Language model initialized")
            
            # Initialize embeddings
            print(f"📊 Initializing {config.EMBEDDING_MODEL} embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            print("✅ Embeddings initialized")
            
            # Load knowledge base
            if not os.path.exists(config.KNOWLEDGE_FILE):
                print(f"❌ Knowledge base file '{config.KNOWLEDGE_FILE}' not found")
                return False
            
            print(f"📚 Loading knowledge base from {config.KNOWLEDGE_FILE}...")
            loader = TextLoader(config.KNOWLEDGE_FILE, encoding='utf-8')
            documents = loader.load()
            print(f"📄 Loaded {len(documents)} documents")
            
            # Split documents into chunks
            print("✂️ Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            print(f"📝 Created {len(texts)} text chunks")
            
            # Create vector store
            print("🔍 Creating vector store...")
            vector_store = FAISS.from_documents(texts, embeddings)
            print("✅ Vector store created successfully")
            
            # Create prompt template
            print("📋 Setting up prompt template...")
            prompt_template = """You are a helpful assistant for college students answering queries related to their college. Use the following context to answer the question.
            
Context: {context}

Question: {question}

Answer in a clear, concise, and friendly manner:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            print("✅ Prompt template configured")
            
            # Create QA chain
            print("⛓️ Building QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            print("✅ QA chain created successfully")
            
            print("🎉 RAG system initialization completed successfully!")
            return True
            
        except Exception as e:
            print(f"💥 RAG system initialization failed: {e}")
            return False
    
    @staticmethod
    def process_query(english_text: str) -> str:
        """Process user query using RAG system"""
        global qa_chain
        
        print(f"🔄 Processing query: '{english_text[:50]}...'")
        
        try:
            if qa_chain is None:
                print("⚠️ RAG system not initialized, providing fallback response")
                fallback_response = f"You asked: {english_text}\n\nI understand your message. How can I assist you further?"
                print(f"📤 Fallback response: {fallback_response[:50]}...")
                return fallback_response
            
            print("🤖 Invoking RAG chain...")
            response = qa_chain.invoke({"query": english_text})
            
            answer = response.get('result', "Sorry, I couldn't find an answer to your question.")
            
            # Log source documents if available
            if 'source_documents' in response and response['source_documents']:
                print(f"📚 Used {len(response['source_documents'])} source documents")
                for i, doc in enumerate(response['source_documents']):
                    print(f"📖 Source {i+1}: {doc.page_content[:100]}...")
            
            print(f"✅ Generated response: {answer[:100]}...")
            return answer
            
        except Exception as e:
            print(f"💥 Error processing query: {e}")
            error_response = "Sorry, I couldn't process your request. Please try again."
            print(f"📤 Error response: {error_response}")
            return error_response

# =====================================================
# ELEVENLABS INTEGRATION
# =====================================================

class ElevenLabsService:
    """Handles ElevenLabs TTS and STT operations"""
    
    @staticmethod
    def text_to_speech(text: str, voice_id: str = None) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs TTS"""
        voice_id = voice_id or config.VOICE_ID
        
        print(f"🔊 Converting text to speech...")
        print(f"📝 Text: '{text[:100]}...'")
        print(f"🎤 Voice ID: {voice_id}")
        print(f"🤖 Model: {config.TTS_MODEL}")
        
        try:
            audio = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=config.TTS_MODEL,
                voice_settings=VoiceSettings(stability=0.7, similarity_boost=0.5)
            )
            print(f"✅ TTS conversion successful, audio size: {len(audio)} bytes")
            return audio
            
        except Exception as e:
            print(f"💥 TTS conversion failed: {e}")
            return None
    
    @staticmethod
    def speech_to_text(audio_url: str) -> Optional[str]:
        """Convert speech to text using ElevenLabs STT"""
        print(f"🎤 Converting speech to text...")
        print(f"🔗 Audio URL: {audio_url}")
        print(f"🤖 STT Model: {config.STT_MODEL}")
        
        try:
            # Download and convert audio
            print("⬇️ Downloading audio file...")
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
                urllib.request.urlretrieve(audio_url, temp_file.name)
                print(f"📁 Downloaded to: {temp_file.name}")
                
                # Convert OGG to MP3
                print("🔄 Converting OGG to MP3...")
                audio = AudioSegment.from_file(temp_file.name, format="ogg")
                mp3_path = temp_file.name.replace(".ogg", ".mp3")
                audio.export(mp3_path, format="mp3")
                print(f"📁 Converted to: {mp3_path}")
                
                # Perform STT
                print("🗣️ Performing speech-to-text conversion...")
                with open(mp3_path, "rb") as f:
                    response = elevenlabs_client.speech_to_text.convert(
                        file=f, 
                        model_id=config.STT_MODEL
                    )
                
                transcription = response.text
                print(f"✅ STT conversion successful: '{transcription}'")
                
                # Cleanup
                print("🧹 Cleaning up temporary files...")
                os.unlink(temp_file.name)
                os.unlink(mp3_path)
                
                return transcription
                
        except Exception as e:
            print(f"💥 STT conversion failed: {e}")
            return None

# =====================================================
# TRANSLATION & LANGUAGE PROCESSING
# =====================================================

class LanguageProcessor:
    """Handles language detection, translation, and Hinglish processing"""
    
    @staticmethod
    def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        """Translate text using Google Translator"""
        print(f"🌐 Translating text: {source_lang} → {target_lang}")
        print(f"📝 Original text: '{text[:50]}...'")
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            print(f"✅ Translation successful: '{result[:50]}...'")
            return result
        except Exception as e:
            print(f"💥 Translation failed: {e}")
            print("📤 Returning original text")
            return text
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Smart language detection with Hinglish support"""
        print(f"🔍 Detecting language for: '{text[:50]}...'")
        
        try:
            detected = detect(text)
            print(f"🎯 Initial detection: {detected}")
            
            # Check for Hinglish patterns
            if detected == 'hi' and text.isascii():
                print("⚡ Detected Hinglish (Hindi words in Latin script)")
                return 'hinglish'
            elif detected == 'en':
                text_lower = text.lower()
                hinglish_count = sum(1 for word in config.HINGLISH_WORDS if word in text_lower)
                if hinglish_count > 0:
                    print(f"⚡ Detected Hinglish (found {hinglish_count} Hinglish words)")
                    return 'hinglish'
            
            # Handle misdetected languages
            if detected in config.MISDETECTED_LANGS:
                print(f"🔧 Correcting misdetected language {detected} → hinglish")
                return 'hinglish'
            
            print(f"✅ Final language detection: {detected}")
            return detected
            
        except Exception as e:
            print(f"💥 Language detection failed: {e}")
            print("📤 Defaulting to English")
            return 'en'
    
    @staticmethod
    def handle_hinglish(text: str, to_english: bool = True) -> str:
        """Handle Hinglish text conversion"""
        direction = "Hinglish → English" if to_english else "English → Hinglish"
        print(f"🔄 Converting {direction}")
        print(f"📝 Input: '{text[:50]}...'")
        
        try:
            if to_english:
                # Hinglish → Devanagari → English
                print("📝 Converting to Devanagari script...")
                devanagari = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                print(f"📝 Devanagari: '{devanagari[:50]}...'")
                
                result = LanguageProcessor.translate_text(devanagari, 'hi', 'en')
                print(f"✅ Hinglish conversion result: '{result[:50]}...'")
                return result
            else:
                # English → Hindi → Hinglish
                print("🔄 Converting English to Hindi...")
                hindi = LanguageProcessor.translate_text(text, 'en', 'hi')
                print(f"📝 Hindi: '{hindi[:50]}...'")
                
                print("🔄 Converting Hindi to Hinglish...")
                result = transliterate(hindi, sanscript.DEVANAGARI, sanscript.ITRANS)
                print(f"✅ Hinglish conversion result: '{result[:50]}...'")
                return result
                
        except Exception as e:
            print(f"💥 Hinglish conversion failed: {e}")
            print("📤 Returning original text")
            return text
    
    @staticmethod
    def process_multilingual_input(input_text: str) -> Tuple[str, str]:
        """Process multilingual input and return English text and final answer"""
        print(f"🌍 Processing multilingual input: '{input_text[:50]}...'")
        
        # Detect language
        detected_lang = LanguageProcessor.detect_language(input_text)
        print(f"🏷️ Detected language: {detected_lang}")
        
        # Process based on detected language
        if detected_lang == 'hinglish':
            print("🔄 Processing Hinglish input...")
            english_text = LanguageProcessor.handle_hinglish(input_text, to_english=True)
            answer_en = RAGSystem.process_query(english_text)
            final_answer = LanguageProcessor.handle_hinglish(answer_en, to_english=False)
            
        elif detected_lang == 'en':
            print("🔄 Processing English input...")
            english_text = input_text
            final_answer = RAGSystem.process_query(english_text)
            
        else:
            print(f"🔄 Processing {detected_lang} input...")
            english_text = LanguageProcessor.translate_text(input_text, detected_lang, 'en')
            answer_en = RAGSystem.process_query(english_text)
            final_answer = LanguageProcessor.translate_text(answer_en, 'en', detected_lang)
        
        print(f"✅ Final answer: '{final_answer[:50]}...'")
        return english_text, final_answer

# =====================================================
# MESSAGE HANDLING
# =====================================================

class MessageHandler:
    """Handles WhatsApp message processing and responses"""
    
    @staticmethod
    def send_audio_message(to: str, text: str, audio_bytes: bytes) -> bool:
        """Send audio message via WhatsApp"""
        print(f"📤 Sending audio message to {to}")
        print(f"🔊 Audio size: {len(audio_bytes)} bytes")
        
        try:
            media_url = f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}"
            
            message = twilio_client.messages.create(
                from_=config.TWILIO_WHATSAPP_NUMBER,
                to=to,
                body="🔊 Here's your audio response:",
                media_url=[media_url]
            )
            
            print(f"✅ Audio message sent successfully, Message SID: {message.sid}")
            return True
            
        except Exception as e:
            print(f"💥 Failed to send audio message: {e}")
            return False
    
    @staticmethod
    def process_voice_message(media_url: str, from_number: str) -> PlainTextResponse:
        """Process incoming voice message"""
        print(f"🎤 Processing voice message from {from_number}")
        
        resp = MessagingResponse()
        
        # Convert speech to text
        transcription = ElevenLabsService.speech_to_text(media_url)
        if not transcription:
            print("❌ STT failed, sending error response")
            resp.message("Sorry, I couldn't understand the audio. Please try again.")
            return PlainTextResponse(str(resp))
        
        print(f"📝 Transcription: '{transcription}'")
        
        # Process the transcribed text
        english_text, final_answer = LanguageProcessor.process_multilingual_input(transcription)
        
        # Generate and send audio response
        audio_bytes = ElevenLabsService.text_to_speech(final_answer)
        if audio_bytes:
            success = MessageHandler.send_audio_message(from_number, final_answer, audio_bytes)
            if success:
                print("✅ Voice message processed successfully")
            else:
                resp.message("Sorry, I couldn't generate audio for your response.")
        else:
            print("❌ TTS failed, sending text response")
            resp.message("Sorry, I couldn't generate audio for your response.")
        
        return PlainTextResponse(str(resp))
    
    @staticmethod
    def process_text_message(text: str) -> str:
        """Process incoming text message"""
        print(f"💬 Processing text message: '{text[:50]}...'")
        
        # Process multilingual text
        english_text, final_answer = LanguageProcessor.process_multilingual_input(text)
        
        print(f"✅ Text message processed successfully")
        return final_answer

# =====================================================
# FASTAPI APPLICATION
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    print("🚀 Starting WhatsApp Bot with RAG, TTS, and STT...")
    
    # Initialize RAG system
    rag_success = RAGSystem.initialize()
    if rag_success:
        print("🎉 WhatsApp Bot is ready and fully functional!")
    else:
        print("⚠️ Bot started but RAG system initialization failed")
    
    yield
    
    print("🛑 Shutting down WhatsApp Bot...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Health check endpoint"""
    print("🏥 Health check requested")
    return {
        "status": "ok",
        "features": ["RAG", "TTS", "STT", "Multilingual", "Hinglish"],
        "models": {
            "llm": config.LLM_MODEL,
            "embeddings": config.EMBEDDING_MODEL,
            "tts": config.TTS_MODEL,
            "stt": config.STT_MODEL
        }
    }

@app.post("/webhook")
async def whatsapp_webhook(
    Body: str = Form(None),
    From: str = Form(...),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    NumMedia: str = Form("0")
):
    """Main webhook endpoint for WhatsApp messages"""
    print("=" * 60)
    print(f"📨 INCOMING MESSAGE FROM: {From}")
    print(f"📊 Media count: {NumMedia}")
    print(f"💬 Body: {Body}")
    print(f"🎵 Media URL: {MediaUrl0}")
    print(f"📎 Media Type: {MediaContentType0}")
    print("=" * 60)
    
    # Determine message type
    is_voice_message = (
        NumMedia and 
        int(NumMedia) > 0 and 
        MediaContentType0 and 
        "audio" in MediaContentType0
    )
    
    if is_voice_message:
        print("🎤 VOICE MESSAGE DETECTED")
        return MessageHandler.process_voice_message(MediaUrl0, From)
    
    elif Body:
        print("💬 TEXT MESSAGE DETECTED")
        final_answer = MessageHandler.process_text_message(Body)
        resp = MessagingResponse()
        resp.message(final_answer)
        print(f"📤 Sending text response: '{final_answer[:50]}...'")
        return PlainTextResponse(str(resp))
    
    else:
        print("❓ UNKNOWN MESSAGE TYPE")
        resp = MessagingResponse()
        resp.message("Please send a text message or voice note.")
        return PlainTextResponse(str(resp))

# =====================================================
# UTILITY ENDPOINTS
# =====================================================

@app.get("/voices")
async def get_available_voices():
    """Get available ElevenLabs voices"""
    print("🎤 Fetching available voices...")
    try:
        response = elevenlabs_client.voices.get_all()
        voices = [{"voice_id": voice.voice_id, "name": voice.name} for voice in response.voices]
        print(f"✅ Retrieved {len(voices)} voices")
        return {"voices": voices}
    except Exception as e:
        print(f"💥 Failed to fetch voices: {e}")
        return {"error": str(e)}

@app.post("/test-tts")
async def test_tts(text: str = Form(...)):
    """Test TTS functionality"""
    print(f"🧪 Testing TTS with text: '{text[:50]}...'")
    try:
        audio_bytes = ElevenLabsService.text_to_speech(text)
        if audio_bytes:
            print(f"✅ TTS test successful, audio size: {len(audio_bytes)} bytes")
            return {"status": "success", "audio_length": len(audio_bytes)}
        else:
            print("❌ TTS test failed")
            return {"status": "failed"}
    except Exception as e:
        print(f"💥 TTS test error: {e}")
        return {"error": str(e)}

@app.post("/test-stt")
async def test_stt(audio_url: str = Form(...)):
    """Test STT functionality"""
    print(f"🧪 Testing STT with audio URL: {audio_url}")
    try:
        transcription = ElevenLabsService.speech_to_text(audio_url)
        if transcription:
            print(f"✅ STT test successful: '{transcription}'")
            return {"status": "success", "transcription": transcription}
        else:
            print("❌ STT test failed")
            return {"status": "failed"}
    except Exception as e:
        print(f"💥 STT test error: {e}")
        return {"error": str(e)}

# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)