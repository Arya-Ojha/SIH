import os
import base64
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from elevenlabs import VoiceSettings, generate,set_api_key
from elevenlabs.client import ElevenLabs
import tempfile
import urllib.request
from pydub import AudioSegment

load_dotenv()

# Environment variables
ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients
client = Client(ACCOUNT_SID, AUTH_TOKEN)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
set_api_key(ELEVENLABS_API_KEY)

# Global variables
vector_store = None
qa_chain = None

# ElevenLabs configuration
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice - you can change this
TTS_MODEL = "eleven_multilingual_v2"
STT_MODEL = "eleven_english_sts_v2"

def initialize_rag_system():
    global vector_store, qa_chain
    
    try:
        print("Initializing RAG system...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  
            temperature=0.7,
            max_tokens=1024
        )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        knowledge_file = "knowledge_base.txt"
        
        if os.path.exists(knowledge_file):
            print(f"Loading knowledge base from {knowledge_file}...")
            
            loader = TextLoader(knowledge_file, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            print(f"Created {len(texts)} text chunks")
            
            vector_store = FAISS.from_documents(texts, embeddings)
            print("Vector store created successfully")
            
            prompt_template = """You are a helpful assistant for college students answering queries related to their college. Use the following context to answer the question. 
            If you don't know the answer based on the context, say so politely.
            
            Context: {context}
            
            Question: {question}
            
            Answer in a clear, concise, and friendly manner:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}  
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("RAG system initialized successfully")
            return True
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return False

def text_to_speech_elevenlabs(text: str, voice_id: str = VOICE_ID) -> bytes:
    """
    Convert text to speech using ElevenLabs TTS
    """
    try:
        print(f"🔊 Converting text to speech: {text[:50]}...")
        
        # Generate audio using ElevenLabs
        audio = generate(
            text=text,
            voice=voice_id,
            model=TTS_MODEL,
            voice_settings=VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        print("✅ TTS generation successful")
        return audio
        
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return None

def speech_to_text_elevenlabs(audio_url: str) -> str:
    """
    Convert speech to text using ElevenLabs STT
    """
    try:
        print(f"🎤 Converting speech to text from: {audio_url}")
        
        # Download the audio file from WhatsApp
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            urllib.request.urlretrieve(audio_url, temp_file.name)
            
            # Convert OGG to MP3 using pydub (ElevenLabs works better with MP3)
            audio = AudioSegment.from_file(temp_file.name, format="ogg")
            mp3_path = temp_file.name.replace(".ogg", ".mp3")
            audio.export(mp3_path, format="mp3")
            
            # Use ElevenLabs STT
            with open(mp3_path, "rb") as audio_file:
                response = elevenlabs_client.speech_to_text.v1.speech_to_text_post(
                    audio=audio_file,
                    model_id=STT_MODEL,
                )
                
            # Clean up temporary files
            os.unlink(temp_file.name)
            os.unlink(mp3_path)
            
            transcription = response.text
            print(f"✅ STT transcription: {transcription}")
            return transcription
            
    except Exception as e:
        print(f"❌ STT conversion failed: {e}")
        return None

def upload_audio_to_twilio(audio_bytes: bytes) -> str:
    """
    Upload audio to Twilio and return the media URL
    """
    try:
        print("📤 Uploading audio to Twilio...")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Upload to Twilio
        with open(temp_file_path, 'rb') as audio_file:
            media = client.messages.media.create(
                parent_sid=None,  # Will be set when sending message
                content_type='audio/mpeg',
                body=audio_file.read()
            )
        
        # Clean up
        os.unlink(temp_file_path)
        
        media_url = f"https://api.twilio.com{media.uri.replace('.json', '')}"
        print(f"✅ Audio uploaded: {media_url}")
        return media_url
        
    except Exception as e:
        print(f"❌ Audio upload failed: {e}")
        return None

def smart_translate(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    try:
        print(f"Translating using Google: {source_lang} -> {target_lang}")
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        result = translator.translate(text)
        
        if result:
            print(f"Google translation successful")
            return result
            
    except Exception as e:
        print(f"Google translator failed: {e}")
    
    print("All translators failed, returning original text")
    return text

def detect_language_smart(text: str) -> str:
    detected = detect(text)
    print(f"Initial detection: {detected}")
    
    if detected == 'hi' and text.isascii():
        print("Detected Hinglish (Hindi words in Latin script)")
        return 'hinglish'
    elif detected == 'en':
        hinglish_words = ['hai', 'hain', 'kya', 'aap', 'main', 'kar', 'kaise', 'kahan', 'kab', 'kyun']
        text_lower = text.lower()
        hinglish_count = sum(1 for word in hinglish_words if word in text_lower)
        
        if hinglish_count >= 1:  
            print("Detected Hinglish (misclassified as English)")
            return 'hinglish'
    
    return detected

def handle_hinglish(text: str, to_english: bool = True) -> str:
    try:
        if to_english:
            print("Converting Hinglish to Devanagari for translation...")
            devanagari_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            print(f"Devanagari: {devanagari_text}")
            
            english_result = smart_translate(devanagari_text, 'hi', 'en')
            return english_result
        else:
            print("Converting English to Hinglish...")
            hindi_result = smart_translate(text, 'en', 'hi')
            hinglish_result = transliterate(hindi_result, sanscript.DEVANAGARI, sanscript.ITRANS)
            return hinglish_result
            
    except Exception as e:
        print(f"Hinglish processing failed: {e}")
        return text

def process_user_query(english_text: str) -> str:
    global qa_chain
    
    try:
        if qa_chain is None:
            print("RAG system not initialized, falling back to basic response")
            return f"You asked: {english_text}\n\nI understand your message. How can I assist you further?"
        
        print(f"Processing query with Gemini RAG: {english_text}")
        
        response = qa_chain.invoke({"query": english_text})
        
        answer = response.get('result', 'I apologize, but I couldn\'t find relevant information to answer your question.')
        
        if 'source_documents' in response:
            print(f"Used {len(response['source_documents'])} source documents")
        
        print(f"Generated response: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"Error in RAG processing: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again later."

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting WhatsApp Bot with Gemini, RAG, and ElevenLabs...")
    success = initialize_rag_system()
    if success:
        print("WhatsApp Bot is ready!")
    else:
        print("Bot started but RAG system initialization failed")

    yield 
    print("Shutting down WhatsApp Bot...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "ok", "features": ["RAG", "Multilingual", "TTS", "STT"]}

@app.post("/webhook")
async def whatsapp_webhook(
    Body: str = Form(None), 
    From: str = Form(...),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    NumMedia: str = Form("0")
):
    print(f"📩 Message from {From}")
    print(f"📱 Media count: {NumMedia}")
    print(f"📝 Body: {Body}")
    print(f"🎵 Media URL: {MediaUrl0}")
    print(f"📎 Media Type: {MediaContentType0}")
    
    try:
        # Initialize response
        resp = MessagingResponse()
        input_text = ""
        
        # Handle voice messages (STT)
        if NumMedia and int(NumMedia) > 0 and MediaContentType0 and "audio" in MediaContentType0:
            print("🎤 Processing voice message...")
            
            # Convert speech to text using ElevenLabs STT
            transcription = speech_to_text_elevenlabs(MediaUrl0)
            
            if transcription:
                input_text = transcription
                print(f"🎤➡️📝 Transcribed: {input_text}")
                # Add a note that this was transcribed
                resp.message(f"🎤 I heard: \"{input_text}\"")
            else:
                resp.message("Sorry, I couldn't understand the audio. Please try again or send a text message.")
                return PlainTextResponse(str(resp))
        
        # Handle text messages
        elif Body:
            input_text = Body
        else:
            resp.message("Please send a text message or voice note.")
            return PlainTextResponse(str(resp))
        
        # Process the input text (whether from transcription or direct text)
        if input_text:
            detected_lang = detect_language_smart(input_text)
            print(f"🔎 Detected language: {detected_lang}")
            
            # Handle misdetected languages
            if detected_lang in ["sw", "id", "ms", "so", "af", "tl"]:
                detected_lang = "hinglish"
            
            # Process based on language
            if detected_lang == 'hinglish':
                print("⚡ Processing Hinglish input...")
                english_text = handle_hinglish(input_text, to_english=True)
                print(f"🌐 English version: {english_text}")
                
                answer_en = process_user_query(english_text)
                final_answer = handle_hinglish(answer_en, to_english=False)
                
            elif detected_lang == 'en':
                print("🌐 Processing English input...")
                english_text = input_text
                final_answer = process_user_query(english_text)
                
            else:
                print(f"🌐 Processing {detected_lang} input...")
                english_text = smart_translate(input_text, detected_lang, 'en')
                print(f"🌐 English version: {english_text}")
                
                answer_en = process_user_query(english_text)
                final_answer = smart_translate(answer_en, 'en', detected_lang)
            
            # Send text response
            resp.message(final_answer)
            
            # Generate and send audio response using ElevenLabs TTS
            print("🔊 Generating audio response...")
            audio_bytes = text_to_speech_elevenlabs(final_answer)
            
            if audio_bytes:
                # Save audio to temporary file and create media URL
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file_path = temp_file.name
                
                # Send audio message via Twilio
                try:
                    message = client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        to=From,
                        body="🔊 Here's the audio version:",
                        media_url=[f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}"]
                    )
                    print("✅ Audio message sent successfully")
                except Exception as audio_error:
                    print(f"❌ Failed to send audio: {audio_error}")
                
                # Clean up
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        return PlainTextResponse(str(resp))
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        resp = MessagingResponse()
        resp.message("Sorry, I encountered an error processing your message. Please try again.")
        return PlainTextResponse(str(resp))


@app.post("/test-tts")
async def test_tts(text: str = Form(...)):
    """
    Test TTS functionality
    """
    try:
        audio_bytes = text_to_speech_elevenlabs(text)
        if audio_bytes:
            return {"status": "success", "audio_length": len(audio_bytes)}
        else:
            return {"status": "failed"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)