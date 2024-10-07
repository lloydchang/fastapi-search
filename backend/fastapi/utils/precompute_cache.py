# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.fastapi.cache.cache_manager_write import save_cache

def load_documents() -> List[str]:
    """
    Load documents from your data source.
    Replace this with your actual document loading logic.
    """
    # Example: You might load documents from a file or database
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

def create_tfidf_matrix(documents: List[str]) -> Any:
    """
    Creates a TF-IDF sparse matrix from the provided documents.

    Args:
        documents (List[str]): List of document strings.

    Returns:
        Any: Sparse TF-IDF matrix and the vectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Consider both unigrams and bigrams
        max_features=10000,  # Limit to top 10,000 features
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

def main():
    # Load your documents
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    # Create TF-IDF matrix and vectorizer
    tfidf_matrix, vectorizer = create_tfidf_matrix(documents)

    # Debug: Print vectorizer attributes
    print("Vectorizer Attributes:")
    print(f"stop_words: {vectorizer.stop_words}")
    print(f"vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"ngram_range: {vectorizer.ngram_range}")
    print(f"max_features: {vectorizer.max_features}")

    # Define cache directory and file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tfidf_cache_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")

    # Save cache using the write cache manager
    save_cache({'tfidf_matrix': tfidf_matrix}, tfidf_cache_path)

    # Save vectorizer metadata
    vectorizer_params = {
        'stop_words': vectorizer.stop_words,
        'ngram_range': vectorizer.ngram_range,
        'max_features': vectorizer.max_features,
        # Add other parameters as needed
    }

    np.savez_compressed(
        tfidf_metadata_path,
        vocabulary=vectorizer.vocabulary_,
        vectorizer_params=vectorizer_params
    )
    print(f"Vectorizer metadata saved to {tfidf_metadata_path}")

    print("Precompute cache written successfully.")

if __name__ == "__main__":
    main()