"""
Embedding Generation Module
Creates vector embeddings for assessment descriptions using HuggingFace models
"""

import logging
import os
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using HuggingFace models"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name (default from env or 'sentence-transformers/all-MiniLM-L6-v2')
        """
        self.model_name = model_name or os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, generating zero embedding")
            return np.zeros(self.embedding_dimension)
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
        
        # Filter empty texts
        valid_texts = [text if text and text.strip() else "" for text in texts]
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def create_assessment_text(self, assessment: Dict) -> str:
        """
        Create combined text representation of an assessment for embedding
        
        Args:
            assessment: Assessment dictionary with name, description, etc.
            
        Returns:
            Combined text string
        """
        parts = []
        
        # Add name
        if assessment.get('name'):
            parts.append(assessment['name'])
        
        # Add description
        if assessment.get('description'):
            parts.append(assessment['description'])
        
        # Add category
        if assessment.get('category'):
            parts.append(assessment['category'])
        
        # Add test type information
        test_type = assessment.get('test_type', '')
        if test_type:
            type_mapping = {
                'K': 'Knowledge and Skills assessment',
                'P': 'Personality and Behavior assessment',
                'A': 'Ability assessment',
                'C': 'Cognitive assessment'
            }
            type_desc = type_mapping.get(test_type, f'Test type {test_type}')
            parts.append(type_desc)
        
        combined_text = ' '.join(parts)
        return combined_text.strip()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.embedding_dimension


def generate_assessment_embeddings(
    assessments: List[Dict],
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> tuple[List[np.ndarray], List[str]]:
    """
    Generate embeddings for a list of assessments
    
    Args:
        assessments: List of assessment dictionaries
        model_name: Optional model name override
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (embeddings list, assessment texts list)
    """
    generator = EmbeddingGenerator(model_name=model_name)
    
    # Create text representations
    texts = [generator.create_assessment_text(assess) for assess in assessments]
    
    # Generate embeddings
    embeddings = generator.generate_embeddings_batch(texts, batch_size=batch_size)
    
    return embeddings.tolist(), texts


if __name__ == "__main__":
    # Test embedding generation
    test_assessments = [
        {
            'name': 'Java Programming Test',
            'description': 'Assesses knowledge of Java programming language',
            'test_type': 'K',
            'category': 'Technical Skills'
        },
        {
            'name': 'Team Collaboration Assessment',
            'description': 'Measures ability to work effectively in teams',
            'test_type': 'P',
            'category': 'Soft Skills'
        }
    ]
    
    generator = EmbeddingGenerator()
    embeddings, texts = generate_assessment_embeddings(test_assessments)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Sample text: {texts[0][:100]}...")







