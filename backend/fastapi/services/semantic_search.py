# File: backend/fastapi/services/semantic_search.py

import os
import numpy as np
from typing import List, Dict

# Debugging helper function for consistent logging
def debug_log(message: str):
    """Utility function to print debug logs with a consistent format."""
    print(f"[DEBUG] {message}")

def load_tfidf_components(cache_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the sparse TF-IDF matrix and associated metadata from cache."""
    matrix_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Load sparse matrix components using numpy
    debug_log(f"Loading sparse TF-IDF matrix from: {matrix_path}")
    matrix_data = np.load(matrix_path)
    debug_log(f"Loaded matrix file: {matrix_path}")
    debug_log(f"Matrix keys found: {matrix_data.files}")

    # Extract components from the npz file
    data = matrix_data['data']
    indices = matrix_data['indices']
    indptr = matrix_data['indptr']
    shape = tuple(matrix_data['shape'])

    debug_log(f"TF-IDF matrix shape: {shape}, data length: {len(data)}, indices length: {len(indices)}")

    # Load metadata
    debug_log(f"Loading metadata from: {metadata_path}")
    metadata = np.load(metadata_path, allow_pickle=True)
    debug_log(f"Loaded metadata keys: {metadata.files}")
    vocabulary = metadata['vocabulary'].item()
    idf_values = metadata['idf_values']

    # Load document metadata and print available keys for debugging
    debug_log(f"Loading document metadata from: {document_metadata_path}")
    document_metadata = np.load(document_metadata_path, allow_pickle=True)
    debug_log(f"Document metadata keys: {document_metadata.files}")

    # Adjust the key access based on actual file contents
    if 'arr_0' in document_metadata:
        debug_log("Loading document metadata using default key 'arr_0'.")
        documents = document_metadata['arr_0']
    elif 'documents' in document_metadata:
        debug_log("Loading document metadata using custom key 'documents'.")
        documents = document_metadata['documents']
    else:
        raise RuntimeError(f"Key missing in document metadata file: {document_metadata.files}")

    debug_log(f"Loaded document metadata with {len(documents)} entries.")
    return {
        "tfidf_matrix": {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "shape": shape
        },
        "vocabulary": vocabulary,
        "idf_values": idf_values,
        "documents": documents
    }

def vectorize_query(query: str, vocabulary: Dict[str, int], idf_values: np.ndarray) -> Dict[int, float]:
    """Create a sparse representation for the query based on the vocabulary and IDF values."""
    debug_log(f"Vectorizing query: '{query}'")
    query_vector = {}
    tokens = query.lower().split()
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    for term, count in token_counts.items():
        if term in vocabulary:
            index = vocabulary[term]
            query_vector[index] = count * idf_values[index]

    debug_log(f"Query vector created with {len(query_vector)} non-zero entries")
    return query_vector

def semantic_search(query: str, cache_dir: str, top_n: int = 5) -> List[Dict]:
    """Perform a semantic search using the precomputed TF-IDF matrix."""
    debug_log(f"Starting semantic search for query: '{query}' in directory: {cache_dir}")
    try:
        # Load the TF-IDF matrix and metadata
        tfidf_data = load_tfidf_components(cache_dir)
        tfidf_matrix = tfidf_data['tfidf_matrix']
        vocabulary = tfidf_data['vocabulary']
        idf_values = tfidf_data['idf_values']
        documents = tfidf_data['documents']
    except RuntimeError as e:
        debug_log(f"Failed to load TF-IDF components: {e}")
        raise RuntimeError(f"Semantic search failed: {e}")

    # Create a vector for the query
    try:
        query_vector = vectorize_query(query, vocabulary, idf_values)
    except Exception as e:
        debug_log(f"Error vectorizing query '{query}': {e}")
        raise RuntimeError(f"Query vectorization failed: {e}")

    # Calculate cosine similarities manually
    try:
        debug_log("Calculating cosine similarities...")
        data, indices, indptr, shape = (
            tfidf_matrix['data'], tfidf_matrix['indices'], tfidf_matrix['indptr'], tfidf_matrix['shape']
        )
        num_docs = shape[0]
        similarities = []

        for doc_id in range(num_docs):
            start = indptr[doc_id]
            end = indptr[doc_id + 1]
            doc_indices = indices[start:end]
            doc_data = data[start:end]

            # Calculate dot product between the document vector and the query vector
            dot_product = sum(query_vector.get(idx, 0) * doc_data[i] for i, idx in enumerate(doc_indices))

            # Calculate norms for cosine similarity
            doc_norm = np.sqrt(sum(val ** 2 for val in doc_data))
            query_norm = np.sqrt(sum(val ** 2 for val in query_vector.values()))

            if query_norm == 0 or doc_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (doc_norm * query_norm)

            similarities.append((doc_id, similarity))

        # Sort by similarity and select the top_n results
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
        results = [
            {
                'document_id': int(doc_id),
                'similarity': float(similarity),
                'slug': documents[doc_id]['slug'],
                'description': documents[doc_id]['description'],
                'presenter': documents[doc_id]['presenterDisplayName'],
                'sdg_tags': documents[doc_id].get('sdg_tags', [])
            } for doc_id, similarity in similarities
        ]

        debug_log(f"Search results: {results}")
    except Exception as e:
        debug_log(f"Error during semantic search: {e}")
        raise RuntimeError(f"Failed to compute similarities: {e}")

    return results