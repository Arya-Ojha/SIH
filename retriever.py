"""
Retrieval module for semantic search using embeddings and FAISS.
Handles embedding generation and similarity search.
"""

import json
import pickle
import numpy as np
import logging
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticRetriever:
    """Handles embedding generation and FAISS-based semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_file: str = "faiss_index.pkl"):
        self.model_name = model_name
        self.index_file = index_file
        self.model = None
        self.index = None
        self.chunks = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        try:
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension: {dimension}")
            
            # Create FAISS index (using L2 distance)
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index created with {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def build_index_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from PDF chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        if not chunks:
            logger.warning("No chunks provided for index building")
            return
        
        try:
            logger.info(f"Building index from {len(chunks)} chunks")
            
            # Extract texts from chunks
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Create FAISS index
            self.index = self.create_faiss_index(embeddings)
            self.chunks = chunks
            
            logger.info("Index built successfully")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def save_index(self) -> None:
        """Save FAISS index and chunks to file."""
        try:
            if self.index is None or not self.chunks:
                logger.warning("No index or chunks to save")
                return
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file.replace('.pkl', '.faiss'))
            
            # Save chunks and metadata
            index_data = {
                "chunks": self.chunks,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "total_vectors": self.index.ntotal
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"Index saved to {self.index_file}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and chunks from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.index_file):
                logger.info(f"No existing index file found: {self.index_file}")
                return False
            
            # Load index data
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            # Load FAISS index
            faiss_file = self.index_file.replace('.pkl', '.faiss')
            if os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
            else:
                logger.error(f"FAISS index file not found: {faiss_file}")
                return False
            
            # Load chunks and metadata
            self.chunks = index_data["chunks"]
            self.model_name = index_data["model_name"]
            self.dimension = index_data["dimension"]
            
            logger.info(f"Index loaded successfully with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks to the query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of most similar chunks with scores
        """
        if self.index is None or not self.chunks:
            logger.warning("No index available for search")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):  # Valid index
                    chunk = self.chunks[idx].copy()
                    chunk["similarity_score"] = float(score)
                    chunk["rank"] = i + 1
                    results.append(chunk)
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context from top-k chunks for LLM.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(results, 1):
            context_parts.append(f"Context {i} (from {chunk['pdf_name']}):\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def is_index_ready(self) -> bool:
        """
        Check if the index is ready for search.
        
        Returns:
            True if index is ready, False otherwise
        """
        return self.index is not None and len(self.chunks) > 0

# Global retriever instance
retriever = SemanticRetriever()
