# File: backend/fastapi/services/search_service.py

from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.fastapi.utils.logger import logger  # Import the centralized logger

number_of_results = 100

def semantic_search(query: str, data: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = number_of_results) -> List[Dict]:
    """
    Performs semantic search on the TEDx dataset using TF-IDF vectors.

    Args:
        query (str): The search query.
        data (pd.DataFrame): The dataset containing TEDx talks.
        vectorizer: The TF-IDF vectorizer.
        tfidf_matrix: The TF-IDF matrix for the descriptions.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    logger.info(f"Performing semantic search for the query: '{query}'.")

    try:
        # Encode the query using the TF-IDF vectorizer
        query_vector = vectorizer.transform([query])
        logger.info("Query encoded successfully using TF-IDF.")

        # Compute cosine similarities between the query and all documents
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        logger.info("Cosine similarities computed successfully.")

        # Get top N indices
        top_indices = np.argsort(-similarities)[:top_n]
        logger.info(f"Top {top_n} indices identified.")

        # Prepare the search results
        results = []
        for idx in top_indices:
            # Check if 'sdg_tags' is present, otherwise use an empty list as a placeholder
            sdg_tags = data.iloc[idx].get('sdg_tags', []) if 'sdg_tags' in data.columns else []

            result = {
                'title': data.iloc[idx]['slug'].replace('_', ' '),
                'description': data.iloc[idx]['description'],
                'presenter': data.iloc[idx]['presenterDisplayName'],
                'sdg_tags': sdg_tags,
                'similarity_score': float(similarities[idx]),
                'url': f"https://www.ted.com/talks/{data.iloc[idx]['slug']}"
            }
            results.append(result)

        logger.info(f"Semantic search completed successfully for query: '{query}'. Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return [{"error": f"Search error: {str(e)}"}]
