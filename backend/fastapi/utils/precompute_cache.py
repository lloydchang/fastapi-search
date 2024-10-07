# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, List, Dict
from backend.fastapi.cache.cache_manager_write import save_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_tedx_documents(csv_file_path: str) -> List[Dict[str, str]]:
    """Load TEDx talks from the provided CSV file and extract metadata and text content."""
    tedx_df = pd.read_csv(csv_file_path)
    if 'description' not in tedx_df.columns:
        raise ValueError(f"Column 'description' not found in the CSV file {csv_file_path}")
    documents = tedx_df[['slug', 'description', 'presenterDisplayName']].dropna().to_dict('records')
    logger.info(f"Loaded {len(documents)} TEDx documents from the CSV file.")
    return documents

def create_tfidf_matrix(documents: List[Dict[str, str]]) -> Any:
    """Create a sparse TF-IDF matrix from the provided document descriptions."""
    descriptions = [doc['description'] for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    logger.info(f"TF-IDF matrix created. Shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

def save_sparse_matrix(tfidf_matrix, cache_dir: str):
    """Save sparse matrix in a numpy-compatible format."""
    tfidf_data_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    np.savez_compressed(
        tfidf_data_path,
        data=tfidf_matrix.data,
        indices=tfidf_matrix.indices,
        indptr=tfidf_matrix.indptr,
        shape=tfidf_matrix.shape
    )
    logger.info(f"Sparse TF-IDF matrix components saved to {tfidf_data_path}")
    # Debug: Confirm saved keys
    with np.load(tfidf_data_path) as data:
        logger.info(f"Saved sparse matrix keys: {data.files}")

def save_tfidf_components(tfidf_matrix, vectorizer: TfidfVectorizer, documents: List[Dict[str, str]], cache_dir: str):
    """Save the TF-IDF matrix and metadata in a format compatible with numpy."""
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Save sparse matrix components
    save_sparse_matrix(tfidf_matrix, cache_dir)

    # Save metadata (vocabulary and IDF values)
    np.savez_compressed(
        tfidf_metadata_path,
        vocabulary=vectorizer.vocabulary_,
        idf_values=vectorizer.idf_
    )
    logger.info(f"TF-IDF metadata saved to {tfidf_metadata_path}")

    # Save document metadata
    np.savez_compressed(document_metadata_path, documents=documents)
    logger.info(f"Document metadata saved to {document_metadata_path}")

def precompute_cache():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")
    csv_file_path = os.path.join(base_dir, "data", "tedx_talks.csv")

    # Load TEDx documents
    documents = load_tedx_documents(csv_file_path)

    # Create the TF-IDF matrix
    tfidf_matrix, vectorizer = create_tfidf_matrix(documents)

    # Save the generated components
    os.makedirs(cache_dir, exist_ok=True)
    save_tfidf_components(tfidf_matrix, vectorizer, documents, cache_dir)
    logger.info("Precompute cache written successfully to the cache directory.")

if __name__ == "__main__":
    precompute_cache()
