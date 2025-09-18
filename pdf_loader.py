"""
PDF loader and semantic chunking module for WhatsApp chatbot.
Extracts text from PDFs and splits them into semantic chunks.
"""

import fitz  # PyMuPDF
import json
import os
import re
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Handles PDF text extraction and semantic chunking."""
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 300):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunks_file = "pdf_chunks.json"
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n"  # Add newline between pages
            
            doc.close()
            logger.info(f"Extracted text from {pdf_path}: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (simple patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better semantic chunking.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def create_semantic_chunks(self, text: str, pdf_name: str) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from text.
        
        Args:
            text: Cleaned text content
            pdf_name: Name of the PDF file
            
        Returns:
            List of chunk dictionaries
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_word_count = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunks.append({
                    "id": f"{pdf_name}_chunk_{chunk_id}",
                    "pdf_name": pdf_name,
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "word_count": current_word_count,
                    "start_sentence": len(chunks) * 10,  # Approximate
                    "end_sentence": len(chunks) * 10 + 10
                })
                chunk_id += 1
                current_chunk = sentence
                current_word_count = sentence_word_count
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_word_count += sentence_word_count
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "id": f"{pdf_name}_chunk_{chunk_id}",
                "pdf_name": pdf_name,
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "word_count": current_word_count,
                "start_sentence": len(chunks) * 10,
                "end_sentence": len(chunks) * 10 + 10
            })
        
        logger.info(f"Created {len(chunks)} chunks from {pdf_name}")
        return chunks
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the data directory.
        
        Returns:
            List of all chunks from all PDFs
        """
        all_chunks = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return all_chunks
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return all_chunks
        
        for pdf_file in pdf_files:
            logger.info(f"Processing PDF: {pdf_file.name}")
            
            # Extract text
            raw_text = self.extract_text_from_pdf(str(pdf_file))
            
            if not raw_text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue
            
            # Clean text
            clean_text = self.clean_text(raw_text)
            
            if not clean_text.strip():
                logger.warning(f"No text after cleaning from {pdf_file.name}")
                continue
            
            # Create chunks
            chunks = self.create_semantic_chunks(clean_text, pdf_file.stem)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
        """
        try:
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(chunks)} chunks to {self.chunks_file}")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file.
        
        Returns:
            List of chunk dictionaries
        """
        try:
            if not os.path.exists(self.chunks_file):
                logger.info(f"No existing chunks file found: {self.chunks_file}")
                return []
            
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []
    
    def process_and_save(self) -> List[Dict[str, Any]]:
        """
        Process all PDFs and save chunks to file.
        
        Returns:
            List of all chunks
        """
        chunks = self.process_all_pdfs()
        if chunks:
            self.save_chunks(chunks)
        return chunks

# Global PDF loader instance
pdf_loader = PDFLoader()
