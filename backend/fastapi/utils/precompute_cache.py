# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, List, Dict  # Ensure all type hints are imported
from backend.fastapi.cache.cache_manager_write import save_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_tedx_documents(csv_file_path: str) -> List[Dict[str, str]]:
    """
    Load TEDx talks from the provided CSV file and extract metadata and text content.

    Args:
        csv_file_path (str): Path to the TEDx CSV file.

    Returns:
        List[Dict[str, str]]: List of TEDx talks with metadata.
    """
    # Load the TEDx CSV file into a DataFrame
    tedx_df = pd.read_csv(csv_file_path)

    # Extract metadata (slug, description, presenterDisplayName) and drop missing values
    if 'description' not in tedx_df.columns:
        raise ValueError(f"Column 'description' not found in the CSV file {csv_file_path}")

    documents = tedx_df[['slug', 'description', 'presenterDisplayName']].dropna().to_dict('records')
    logger.info(f"Loaded {len(documents)} TEDx documents from the CSV file.")
    
    return documents

def create_tfidf_matrix(documents: List[Dict[str, str]]) -> Any:
    """
    Create a TF-IDF matrix from the provided document descriptions.

    Args:
        documents (List[Dict[str, str]]): List of document metadata.

    Returns:
        Any: Sparse TF-IDF matrix and the vectorizer instance.
    """
    descriptions = [doc['description'] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer

def save_tfidf_components(tfidf_matrix: np.ndarray, vectorizer: TfidfVectorizer, documents: List[Dict[str, str]], cache_dir: str):
    """Save the TF-IDF matrix, vectorizer metadata, and document metadata."""
    tfidf_cache_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Convert sparse matrix to dense format for saving
    tfidf_dense = tfidf_matrix.toarray()

    # Save the dense TF-IDF matrix
    save_cache({'tfidf_matrix': tfidf_dense}, tfidf_cache_path)

    # Save metadata (vocabulary and IDF values)
    np.savez_compressed(
        tfidf_metadata_path,
        vocabulary=vectorizer.vocabulary_,
        idf_values=vectorizer.idf_
    )
    logger.info(f"TF-IDF matrix and metadata saved to {tfidf_cache_path} and {tfidf_metadata_path}")

    # Save document metadata for semantic search
    np.savez_compressed(document_metadata_path, documents=documents)
    logger.info(f"Document metadata saved to {document_metadata_path}")

def precompute_cache():
    """Main precomputation process to build and save TF-IDF data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")
    csv_file_path = os.path.join(base_dir, "data", "tedx_talks.csv")  # Adjust the path as needed

    # Load TEDx talk documents from the CSV file
    documents = load_tedx_documents(csv_file_path)

    # Create the TF-IDF matrix and vectorizer using the loaded TEDx documents
    tfidf_matrix, vectorizer = create_tfidf_matrix(documents)

    # Save the generated TF-IDF components
    os.makedirs(cache_dir, exist_ok=True)
    save_tfidf_components(tfidf_matrix, vectorizer, documents, cache_dir)
    logger.info("Precompute cache written successfully to the cache directory.")

if __name__ == "__main__":
    precompute_cache()
