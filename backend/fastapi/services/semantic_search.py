# File: backend/fastapi/services/semantic_search.py

import os
import numpy as np
from typing import List, Dict

# Debugging helper function for consistent logging
def debug_log(message: str):
    """Utility function to print debug logs with a consistent format."""
    print(f"[DEBUG] {message}")

def load_tfidf_components(cache_dir: str) -> Dict[str, np.ndarray]:
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

    # Reconstruct the dense matrix using the components
    debug_log(f"Reconstructing dense matrix from sparse components...")
    tfidf_matrix = np.zeros(shape)
    for row in range(shape[0]):
        start = indptr[row]
        end = indptr[row + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        tfidf_matrix[row, row_indices] = row_data
    debug_log(f"Reconstructed dense matrix with shape: {tfidf_matrix.shape}")

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
    return {"tfidf_matrix": tfidf_matrix, "vocabulary": vocabulary, "idf_values": idf_values, "documents": documents}

def vectorize_query(query: str, vocabulary: Dict[str, int], idf_values: np.ndarray) -> np.ndarray:
    """Create a TF-IDF vector for the query based on the vocabulary and IDF values."""
    debug_log(f"Vectorizing query: '{query}'")
    query_vector = np.zeros(len(vocabulary))
    tokens = query.lower().split()
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    for term, count in token_counts.items():
        if term in vocabulary:
            index = vocabulary[term]
            query_vector[index] = count * idf_values[index]

    debug_log(f"Query vector created: {query_vector}")
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

    # Calculate similarities using numpy operations
    try:
        debug_log("Calculating cosine similarities...")
        dot_products = np.dot(tfidf_matrix, query_vector)
        doc_norms = np.linalg.norm(tfidf_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            debug_log("Query vector norm is zero; returning empty results.")
            return []

        cosine_similarities = dot_products / (doc_norms * query_norm + 1e-10)
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]

        debug_log(f"Top document indices: {top_indices}")
        results = [
            {
                'document_id': int(idx),
                'similarity': float(cosine_similarities[idx]),
                'slug': documents[idx]['slug'],
                'description': documents[idx]['description'],
                'presenter': documents[idx]['presenterDisplayName']
            } for idx in top_indices
        ]
        debug_log(f"Search results: {results}")
    except Exception as e:
        debug_log(f"Error during semantic search: {e}")
        raise RuntimeError(f"Failed to compute similarities: {e}")

    return results
