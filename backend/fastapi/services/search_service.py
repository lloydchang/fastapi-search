# File: backend/fastapi/services/search_service.py

from typing import List, Dict
from backend.fastapi.utils.logger import logger
from backend.fastapi.utils.text_processing import (
    preprocess,
    compute_tf,
    compute_tfidf,
    cosine_similarity,
)

number_of_results = 100

def semantic_search(query: str, data: list, idf_dict, document_tfidf_vectors, top_n: int = number_of_results) -> List[Dict]:
    """
    Performs semantic search on the TEDx dataset using custom TF-IDF vectors.

    Args:
        query (str): The search query.
        data (list): The dataset containing TEDx talks.
        idf_dict: The IDF dictionary.
        document_tfidf_vectors: Precomputed TF-IDF vectors for the documents.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    logger.info(f"Performing semantic search for the query: '{query}'.")

    try:
        query_tokens = preprocess(query)
        query_tf = compute_tf(query_tokens)
        query_tfidf = compute_tfidf(query_tf, idf_dict)

        similarities = []
        for idx, doc_tfidf in enumerate(document_tfidf_vectors):
            sim = cosine_similarity(query_tfidf, doc_tfidf)
            similarities.append((idx, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_n]]
        logger.info(f"Top {top_n} indices identified.")

        # Prepare the search results
        results = []
        for idx in top_indices:
            doc = data[idx]
            # Check if 'sdg_tags' is present, otherwise use an empty list as a placeholder
            sdg_tags = doc.get('sdg_tags', [])

            result = {
                'title': doc.get('slug', '').replace('_', ' '),
                'description': doc.get('description', ''),
                'presenter': doc.get('presenterDisplayName', ''),
                'sdg_tags': sdg_tags,
                'similarity_score': float(similarities[idx][1]),
                'url': f"https://www.ted.com/talks/{doc.get('slug', '')}"
            }
            results.append(result)

        logger.info(f"Semantic search completed successfully for query: '{query}'. Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return [{"error": f"Search error: {str(e)}"}]
