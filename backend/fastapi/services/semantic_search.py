# File: backend/fastapi/services/semantic_search.py

import os
import numpy as np
from typing import List, Dict
from collections import defaultdict
import math

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

    documents = document_metadata['documents']

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
    query_vector = defaultdict(float)
    tokens = query.lower().split()
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    for term, count in token_counts.items():
        if term in vocabulary:
            index = vocabulary[term]
            query_vector[index] = count * idf_values[index]

    debug_log(f"Query vector created with {len(query_vector)} non-zero entries")
    return query_vector

def cosine_similarity_manual(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Manually compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 != 0 and norm_vec2 != 0:
        return dot_product / (norm_vec1 * norm_vec2)
    return 0.0

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
            tfidf_matrix["data"],
            tfidf_matrix["indices"],
            tfidf_matrix["indptr"],
            tfidf_matrix["shape"]
        )

        # Create a sparse vector representation for the query
        query_sparse_vector = np.zeros(shape[1])

        for index, value in query_vector.items():
            query_sparse_vector[index] = value

        # Calculate cosine similarity manually
        cosine_similarities = []
        for i in range(shape[0]):
            document_vector = np.zeros(shape[1])
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            document_vector[indices[start_idx:end_idx]] = data[start_idx:end_idx]

            # Cosine similarity calculation
            cosine_similarity_score = cosine_similarity_manual(document_vector, query_sparse_vector)
            cosine_similarities.append((cosine_similarity_score, i))

        # Sort by similarity in descending order and return top_n results
        cosine_similarities.sort(reverse=True, key=lambda x: x[0])
        top_results = cosine_similarities[:top_n]

        # Retrieve corresponding documents and return results
        results = []
        for score, idx in top_results:
            document = documents[idx]  # Directly access the document from the dict
            results.append({
                "score": score,
                "document": document
            })

        debug_log(f"Search results returned: {len(results)}")
        return results

    except Exception as e:
        debug_log(f"Error during semantic search: {e}")
        raise RuntimeError(f"Semantic search failed: {e}")
