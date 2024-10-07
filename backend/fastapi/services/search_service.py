# File: backend/fastapi/services/search_service.py

import os
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def semantic_search(query: str, cache_dir: str, top_n: int = 5) -> List[Dict]:
    """
    Performs a semantic search based on the query.

    Args:
        query (str): The search query.
        cache_dir (str): Directory where the cache is stored.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results.
    """
    # Load the TF-IDF matrix
    cache_file_path = os.path.join(cache_dir, 'tfidf_matrix.npz')
    cache = load_cache(cache_file_path)
    if cache is None or 'tfidf_matrix' not in cache:
        raise RuntimeError("TF-IDF matrix not found in cache.")

    tfidf_matrix = cache['tfidf_matrix']  # This is a dense numpy array

    # Load the vectorizer metadata (assuming it's saved during precompute)
    vectorizer_metadata_path = os.path.join(cache_dir, 'vectorizer_metadata.npz')
    with np.load(vectorizer_metadata_path, allow_pickle=True) as data:
        vocabulary = data['vocabulary'].item()
        vectorizer_params = data['vectorizer_params'].item()

    # Initialize the vectorizer with saved parameters
    vectorizer = TfidfVectorizer()
    vectorizer.vocabulary_ = vocabulary
    vectorizer._validate_vocabulary()
    # Note: You may need to set other parameters as needed

    # Transform the query into TF-IDF vector
    query_vector = vectorizer.transform([query]).toarray()  # Dense vector

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'document_id': idx,
            'similarity': similarities[idx],
            # Add more fields as needed, such as document content or metadata
        })

    return results
