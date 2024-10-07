# File: backend/fastapi/services/semantic_search.py

import os
import numpy as np
from typing import List, Dict

def load_tfidf_components(cache_dir: str) -> Dict[str, np.ndarray]:
    """Load the TF-IDF matrix and associated metadata from cache."""
    matrix_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Load the TF-IDF matrix and metadata
    matrix_data = np.load(matrix_path, allow_pickle=True)
    tfidf_matrix = matrix_data['tfidf_matrix']
    metadata = np.load(metadata_path, allow_pickle=True)
    vocabulary = metadata['vocabulary'].item()
    idf_values = metadata['idf_values']

    # Load document metadata
    document_metadata = np.load(document_metadata_path, allow_pickle=True)['documents']

    return {"tfidf_matrix": tfidf_matrix, "vocabulary": vocabulary, "idf_values": idf_values, "documents": document_metadata}

def vectorize_query(query: str, vocabulary: Dict[str, int], idf_values: np.ndarray) -> np.ndarray:
    """Create a TF-IDF vector for the query based on the vocabulary and IDF values."""
    query_vector = np.zeros(len(vocabulary))
    tokens = query.lower().split()
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    # Populate the query vector with term frequencies multiplied by IDF values
    for term, count in token_counts.items():
        if term in vocabulary:
            index = vocabulary[term]
            query_vector[index] = count * idf_values[index]

    return query_vector

def semantic_search(query: str, cache_dir: str, top_n: int = 5) -> List[Dict]:
    """Perform a semantic search using the precomputed TF-IDF matrix."""
    tfidf_data = load_tfidf_components(cache_dir)
    tfidf_matrix = tfidf_data['tfidf_matrix']
    vocabulary = tfidf_data['vocabulary']
    idf_values = tfidf_data['idf_values']
    documents = tfidf_data['documents']

    # Vectorize the input query
    query_vector = vectorize_query(query, vocabulary, idf_values)

    # Compute cosine similarities between the query and document vectors
    dot_products = tfidf_matrix @ query_vector
    doc_norms = np.linalg.norm(tfidf_matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return []

    cosine_similarities = dot_products / (doc_norms * query_norm + 1e-10)

    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    results = [
        {
            'document_id': int(idx),
            'similarity': float(cosine_similarities[idx]),
            'slug': documents[idx]['slug'],
            'description': documents[idx]['description'],
            'presenter': documents[idx]['presenterDisplayName']
        } for idx in top_indices
    ]
    return results
