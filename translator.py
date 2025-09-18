"""
Language detection and translation module for multilingual WhatsApp chatbot.
Supports English, Hindi, and Bengali languages.
"""

from googletrans import Translator
from langdetect import detect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTranslator:
    """Handles language detection and translation for multilingual support."""
    
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'bn': 'Bengali'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Language code (en, hi, bn) or 'en' as default
        """
        try:
            detected_lang = detect(text)
            logger.info(f"Detected language: {detected_lang} for text: {text[:50]}...")
            
            # Map detected language to supported languages
            if detected_lang in self.supported_languages:
                return detected_lang
            else:
                logger.warning(f"Unsupported language detected: {detected_lang}, defaulting to English")
                return 'en'
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'en'  # Default to English
    
    def translate_to_english(self, text: str, source_lang: str = None) -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code (optional, will auto-detect if not provided)
            
        Returns:
            Translated text in English
        """
        try:
            if source_lang is None:
                source_lang = self.detect_language(text)
            
            if source_lang == 'en':
                return text  # Already in English
            
            result = self.translator.translate(text, src=source_lang, dest='en')
            logger.info(f"Translated from {source_lang} to English: {text[:30]}... -> {result.text[:30]}...")
            return result.text
            
        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return text  # Return original text if translation fails
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: English text to translate
            target_lang: Target language code (hi, bn)
            
        Returns:
            Translated text in target language
        """
        try:
            if target_lang == 'en':
                return text  # Already in English
            
            if target_lang not in self.supported_languages:
                logger.warning(f"Unsupported target language: {target_lang}, returning English")
                return text
            
            result = self.translator.translate(text, src='en', dest=target_lang)
            logger.info(f"Translated from English to {target_lang}: {text[:30]}... -> {result.text[:30]}...")
            return result.text
            
        except Exception as e:
            logger.error(f"Error translating from English to {target_lang}: {e}")
            return text  # Return original text if translation fails
    
    def get_language_name(self, lang_code: str) -> str:
        """
        Get human-readable language name from language code.
        
        Args:
            lang_code: Language code (en, hi, bn)
            
        Returns:
            Human-readable language name
        """
        return self.supported_languages.get(lang_code, 'English')
    
    def is_supported_language(self, lang_code: str) -> bool:
        """
        Check if language code is supported.
        
        Args:
            lang_code: Language code to check
            
        Returns:
            True if supported, False otherwise
        """
        return lang_code in self.supported_languages

# Global translator instance
translator = MultilingualTranslator()
