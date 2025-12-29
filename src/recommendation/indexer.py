"""
Indexing Module
Creates embeddings and indexes assessments in vector database
"""

import logging
import json
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from .embeddings import EmbeddingGenerator, generate_assessment_embeddings
from .vector_db import VectorDB, initialize_vector_db

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssessmentIndexer:
    """Index assessments into vector database"""
    
    def __init__(
        self,
        embedding_model: str = None,
        vector_db: VectorDB = None
    ):
        """
        Initialize indexer
        
        Args:
            embedding_model: Embedding model name (optional)
            vector_db: VectorDB instance (optional, will create if not provided)
        """
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        # Initialize vector DB
        if vector_db:
            self.vector_db = vector_db
        else:
            self.vector_db = initialize_vector_db(embedding_dimension=embedding_dim)
        
        logger.info(f"Indexer initialized with model: {self.embedding_generator.model_name}")
    
    def index_assessments(
        self,
        assessments: List[Dict],
        batch_size: int = 32
    ):
        """
        Index a list of assessments
        
        Args:
            assessments: List of assessment dictionaries
            batch_size: Batch size for embedding generation
        """
        if not assessments:
            logger.warning("No assessments to index")
            return
        
        logger.info(f"Indexing {len(assessments)} assessments")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings, texts = generate_assessment_embeddings(
            assessments,
            model_name=self.embedding_generator.model_name,
            batch_size=batch_size
        )
        
        # Upsert to vector database
        logger.info("Storing embeddings in vector database...")
        self.vector_db.upsert_assessments(
            assessments=assessments,
            embeddings=embeddings,
            batch_size=100
        )
        
        logger.info(f"Successfully indexed {len(assessments)} assessments")
    
    def index_from_file(self, catalog_path: str):
        """
        Index assessments from a catalog JSON file
        
        Args:
            catalog_path: Path to catalog JSON file
        """
        logger.info(f"Loading catalog from {catalog_path}")
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assessments = data.get('assessments', [])
        
        if not assessments:
            logger.error("No assessments found in catalog file")
            return
        
        logger.info(f"Found {len(assessments)} assessments in catalog")
        
        # Index assessments
        self.index_assessments(assessments)
        
        # Print statistics
        stats = self.vector_db.get_stats()
        logger.info(f"Index statistics: {stats}")


def index_catalog(
    catalog_path: str = "data/processed_catalog.json",
    embedding_model: str = None,
    batch_size: int = 32
):
    """
    Convenience function to index a catalog file
    
    Args:
        catalog_path: Path to catalog JSON file
        embedding_model: Optional embedding model name
        batch_size: Batch size for processing
    """
    indexer = AssessmentIndexer(embedding_model=embedding_model)
    indexer.index_from_file(catalog_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Index SHL assessments in vector database')
    parser.add_argument(
        '--catalog',
        type=str,
        default='data/processed_catalog.json',
        help='Path to catalog JSON file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Embedding model name (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    
    args = parser.parse_args()
    
    try:
        index_catalog(
            catalog_path=args.catalog,
            embedding_model=args.model,
            batch_size=args.batch_size
        )
        print("\n✓ Indexing completed successfully!")
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        exit(1)







