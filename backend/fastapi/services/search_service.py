# File: backend/fastapi/services/search_service.py

from typing import List, Dict
import numpy as np
from backend.fastapi.utils.text_processing import preprocess, compute_tf, compute_idf, compute_tfidf, cosine_similarity

# Load the precomputed IDF values (saved during the precompute phase)
# Replace with a path to your IDF values or use the ones you computed earlier
precomputed_idf = {}  # Assuming this is populated from a saved file or precomputed during cache initialization

def compute_query_vector(query: str, idf_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Computes the TF-IDF vector for a query string.

    Args:
        query (str): The search query.
        idf_dict (Dict[str, float]): Precomputed IDF values.

    Returns:
        Dict[str, float]: The TF-IDF vector representation of the query.
    """
    # Preprocess the query to get tokens
    tokens = preprocess(query)

    # Compute term frequency (TF)
    tf_query = compute_tf(tokens)

    # Compute TF-IDF for the query
    return compute_tfidf(tf_query, idf_dict)

def semantic_search(query: str, tfidf_matrix: List[Dict[str, float]], data: list, top_n: int = 1) -> List[Dict]:
    """
    Performs semantic search on the TEDx dataset using manually computed TF-IDF vectors.

    Args:
        query (str): The search query.
        tfidf_matrix: The list of precomputed TF-IDF vectors for the documents.
        data (list): The preloaded data containing documents and their metadata.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    try:
        # Compute the TF-IDF vector for the query
        query_vector = compute_query_vector(query, precomputed_idf)

        # Calculate cosine similarities manually
        similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in tfidf_matrix]

        # Get the indices of the top N similar documents
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        # Prepare the search results
        results = []
        for idx in top_indices:
            doc = data[idx]
            result = {
                'title': doc.get('slug', '').replace('_', ' '),
                'description': doc.get('description', ''),
                'presenter': doc.get('presenter', ''),
                'sdg_tags': doc.get('sdg_tags', []),
                'similarity_score': float(similarities[idx]),
                'url': f"https://www.ted.com/talks/{doc.get('slug', '')}"
            }
            results.append(result)

        return results

    except Exception as e:
        return [{"error": f"Search error: {str(e)}"}]
