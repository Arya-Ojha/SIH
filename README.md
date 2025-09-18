# Multilingual WhatsApp Chatbot Backend

A complete minimal MVP backend for a multilingual WhatsApp chatbot that answers questions from institutional PDFs using FastAPI, semantic search, and OpenAI.

## 🚀 Quick Start (Simplified Version)

The simplified version works immediately without heavy dependencies:

```bash
# 1. Install minimal dependencies
pip install fastapi uvicorn requests

# 2. Start the server
python app.py

# 3. Test the system (in another terminal)
python test_simple.py
```

## 📁 Clean Project Structure

```
├── app.py               # ✅ Main application (works with minimal dependencies)
├── translator.py        # Language detection and translation
├── pdf_loader.py        # PDF extraction and chunking
├── retriever.py         # Embeddings and FAISS search
├── llm.py              # Google Gemini AI integration
├── requirements.txt    # All Python dependencies
├── test_simple.py     # Test script for simplified version
├── data/              # Directory for PDF files
│   └── sample_document.pdf
└── README.md          # This file
```

## 🎯 Features

### Simplified Version (Working Now)

- ✅ **Multilingual Support**: English, Hindi, Bengali detection
- ✅ **WhatsApp Webhook**: Ready-to-use webhook endpoint
- ✅ **Knowledge Base**: Built-in institutional information
- ✅ **Language Detection**: Basic character-based detection
- ✅ **Mock Translation**: Demonstrates translation flow

### Full Version (Requires Dependencies)

- 🔄 **PDF Processing**: Extract text from PDFs and create semantic chunks
- 🔄 **Semantic Search**: Use FAISS and SentenceTransformers for fast retrieval
- 🔄 **LLM Integration**: Google Gemini AI for intelligent responses
- 🔄 **Real Translation**: Google Translate API integration

## 🛠️ Setup Instructions

### Option 1: Simplified Version (Recommended for Demo)

```bash
# Install minimal dependencies
pip install fastapi uvicorn requests

# Start the server
python app.py

# Test the system
python test_simple.py
```

### Option 2: With Gemini AI (Advanced Features)

```bash
# Install all dependencies including Gemini AI
pip install -r requirements.txt

# Set Google API key
export GOOGLE_API_KEY="your-google-api-key-here"

# Start the server with Gemini AI
python app.py
```

## 📡 API Endpoints

### WhatsApp Webhook

- **POST** `/whatsapp-webhook`
- Processes incoming WhatsApp messages
- Expected payload:

```json
{
	"message": "What are the admission requirements?",
	"from": "+1234567890",
	"timestamp": "2024-01-01T00:00:00Z"
}
```

### Health Check

- **GET** `/health`
- Returns system status and configuration

### Search (Testing)

- **GET** `/search?query=admission`
- Test the search functionality

### Statistics

- **GET** `/stats`
- Get system statistics and status

## 🧪 Testing

### Test the Simplified Version

```bash
# Start server
python app.py

# In another terminal, run tests
python test_simple.py
```

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Search test
curl "http://localhost:8000/search?query=admission"

# WhatsApp webhook test
curl -X POST "http://localhost:8000/whatsapp-webhook" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What are the admission requirements?",
       "from": "+1234567890"
     }'
```

## 🌍 Multilingual Support

The chatbot automatically:

1. Detects the language of incoming messages
2. Processes the query in English
3. Searches the knowledge base
4. Returns response in original language

Supported languages:

- English (en)
- Hindi (hi)
- Bengali (bn)

### Example Messages

**English**: "What are the admission requirements?"
**Hindi**: "प्रवेश आवश्यकताएं क्या हैं?"
**Bengali**: "ভর্তির প্রয়োজনীয়তা কি?"

## 📊 Knowledge Base

The simplified version includes built-in knowledge about:

- Admission requirements
- Tuition and fees
- Application deadlines
- Contact information
- Academic calendar

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Server configuration
export PORT=8000
export HOST=0.0.0.0

# Required for full version: Google API Key
export GOOGLE_API_KEY="your-google-api-key-here"
```

### Customizing Knowledge Base

Edit the `KNOWLEDGE_BASE` dictionary in `app_simple.py`:

```python
KNOWLEDGE_BASE = {
    "admission": "Your custom admission information...",
    "tuition": "Your custom tuition information...",
    # Add more entries
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"Cannot connect to server"**

   - Make sure the server is running: `python app_simple.py`
   - Check if port 8000 is available

2. **"Module not found"**

   - Install dependencies: `pip install fastapi uvicorn requests`

3. **"No response from webhook"**
   - Check server logs for errors
   - Verify JSON payload format

### Dependencies Issues

If you encounter dependency conflicts:

1. **Use the main version** (`app.py`) - works with minimal dependencies
2. **Create a virtual environment**:
   ```bash
   python -m venv chatbot_env
   chatbot_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

## 📈 Next Steps

1. **Test the main version** to see the basic functionality
2. **Install full dependencies** for PDF processing and LLM integration
3. **Add your Google API key** for intelligent responses
4. **Add more PDFs** to the `data/` directory
5. **Customize the knowledge base** for your institution
6. **Deploy to production** using your preferred hosting platform

## 🎉 Success!

If you can run `python app.py` and `python test_simple.py` successfully, you have a working multilingual WhatsApp chatbot backend!

The system demonstrates:

- ✅ Multilingual message processing
- ✅ Language detection
- ✅ Knowledge base search
- ✅ WhatsApp webhook integration
- ✅ RESTful API endpoints

## 📝 License

This project is provided as-is for educational and demonstration purposes.
