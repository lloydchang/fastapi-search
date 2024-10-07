# File: backend/fastapi/services/search_service.py

from typing import List, Dict, Any, Optional
import numpy as np
import os
from threading import Lock
import logging
from scipy.sparse import csr_matrix

from backend.fastapi.cache.cache_manager import load_cache

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global variables to hold the loaded resources
tfidf_matrix: Optional[csr_matrix] = None
vocabulary: Optional[Dict[str, int]] = None
idf_values: Optional[np.ndarray] = None
data: Optional[List[Dict[str, Any]]] = None
resources_initialized = False
load_lock = Lock()

def load_resources(cache_dir: str):
    """
    Loads precomputed resources from .npz files.

    Args:
        cache_dir (str): Directory where cache files are stored.
    """
    global tfidf_matrix, vocabulary, idf_values, data, resources_initialized

    with load_lock:
        if resources_initialized:
            logger.debug("Resources already initialized. Skipping load.")
            return

        logger.info(f"Starting to load precomputed data from {cache_dir}...")

        # Load TF-IDF matrix (sparse)
        logger.info("Loading TF-IDF matrix...")
        tfidf_matrix_path = os.path.join(cache_dir, 'tfidf_matrix.npz')
        tfidf_data = load_cache(tfidf_matrix_path)
        if tfidf_data is None or 'tfidf_matrix' not in tfidf_data:
            raise RuntimeError("TF-IDF matrix not found or corrupted.")
        tfidf_matrix = tfidf_data['tfidf_matrix']
        logger.info(f"TF-IDF matrix loaded successfully with shape {tfidf_matrix.shape}.")

        # Load TF-IDF metadata (vocabulary and IDF)
        logger.info("Loading TF-IDF metadata...")
        tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')
        metadata = load_cache(tfidf_metadata_path)
        if metadata is None or 'vocabulary' not in metadata or 'idf' not in metadata:
            raise RuntimeError("TF-IDF metadata not found or corrupted.")
        vocabulary = metadata['vocabulary'].item()  # Convert from numpy object to dict
        idf_values = metadata['idf']
        logger.info(f"TF-IDF metadata loaded successfully. Vocabulary size: {len(vocabulary)}.")

        # Load document data
        logger.info("Loading document data...")
        data_path = os.path.join(cache_dir, 'data.npz')
        document_data = load_cache(data_path)
        if document_data is None or 'data' not in document_data:
            raise RuntimeError("Document data not found or corrupted.")

        # Correctly convert each numpy.void object to a dictionary
        try:
            data = [dict(zip(doc.dtype.names, doc)) for doc in document_data['data']]
        except AttributeError as e:
            logger.error(f"Error converting document data to dictionaries: {e}")
            raise RuntimeError("Document data conversion failed.")

        logger.info(f"Document data loaded successfully. Number of documents: {len(data)}.")

        resources_initialized = True
        logger.info("Resources successfully loaded.")

def compute_query_vector(query: str, vocabulary: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    """
    Computes the TF-IDF vector for the query using precomputed vocabulary and IDF values.

    Args:
        query (str): The search query.
        vocabulary (Dict[str, int]): Precomputed vocabulary dictionary.
        idf (np.ndarray): Precomputed IDF values.

    Returns:
        np.ndarray: The computed query vector.
    """
    # Initialize a zero vector for the query
    query_vector = np.zeros(len(vocabulary))

    # Tokenize the query
    tokens = query.lower().split()
    logger.debug(f"Tokenized query: {tokens}")

    # Compute term frequency (TF)
    term_counts = {}
    for token in tokens:
        if token in vocabulary:
            term_counts[token] = term_counts.get(token, 0) + 1

    total_terms = len(tokens)
    logger.debug(f"Term counts: {term_counts}")
    logger.debug(f"Total terms in query: {total_terms}")

    for term, count in term_counts.items():
        index = vocabulary[term]
        query_vector[index] = (count / total_terms) * idf[index]
        logger.debug(f"Term: {term}, Index: {index}, TF-IDF Value: {query_vector[index]}")

    return query_vector

def semantic_search(query: str, cache_dir: str, top_n: int = 1) -> List[Dict]:
    """
    Perform semantic search on the precomputed data using numpy-based implementations.

    Args:
        query (str): The search query.
        cache_dir (str): Directory where cache files are stored.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    global resources_initialized

    try:
        if not resources_initialized:
            logger.debug("Resources not initialized. Loading resources...")
            load_resources(cache_dir)

        # Compute the query vector
        logger.debug(f"Computing query vector for: '{query}'")
        query_vector = compute_query_vector(query, vocabulary, idf_values)
        logger.debug(f"Query vector computed with shape: {query_vector.shape}")

        # Compute cosine similarities using sparse matrix operations
        logger.debug("Computing cosine similarities...")
        # Normalize the query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            logger.warning("Query vector norm is zero. Returning empty results.")
            return []
        normalized_query = query_vector / query_norm

        # Compute dot product
        dot_products = tfidf_matrix.dot(normalized_query)

        # Since tfidf_matrix is sparse and normalized_query is dense, dot_products is a numpy array
        cosine_similarities = dot_products.flatten()

        logger.debug("Cosine similarities computed.")

        # Get the top N results
        logger.debug(f"Fetching top {top_n} results...")
        top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
        logger.debug(f"Top indices: {top_indices}")

        # Prepare the search results
        results = []
        for idx in top_indices:
            doc = data[idx]
            result = {
                'title': doc['slug'].replace('_', ' '),
                'description': doc['description'],
                'presenter': doc['presenter'],
                'sdg_tags': doc['sdg_tags'],
                'similarity_score': float(cosine_similarities[idx]),
                'url': f"https://www.ted.com/talks/{doc['slug']}"
            }
            results.append(result)
            logger.debug(f"Result appended: {result}")

        logger.info(f"Semantic search completed for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        raise RuntimeError("Semantic search failed.")
