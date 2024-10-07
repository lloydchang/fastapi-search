# File: backend/fastapi/services/search_service.py

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

number_of_results = 1  # Set to return only 1 result

def semantic_search(query: str, tfidf_vectorizer, tfidf_matrix, data: list, top_n: int = number_of_results) -> List[Dict]:
    """
    Performs semantic search on the TEDx dataset using precomputed TF-IDF vectors.

    Args:
        query (str): The search query.
        tfidf_vectorizer: The preloaded TF-IDF vectorizer.
        tfidf_matrix: The preloaded TF-IDF matrix.
        data (list): The preloaded data containing documents and their metadata.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    try:
        # Transform the query using the loaded vectorizer
        query_vector = tfidf_vectorizer.transform([query])

        # Compute cosine similarities using vectorized operations
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get the indices of the top N similar documents
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]

        # Prepare the search results
        results = []
        for idx in top_indices:
            doc = data[idx]
            result = {
                'title': doc.get('slug', '').replace('_', ' '),
                'description': doc.get('description', ''),
                'presenter': doc.get('presenter', ''),
                'sdg_tags': doc.get('sdg_tags', []),
                'similarity_score': float(cosine_similarities[idx]),
                'url': f"https://www.ted.com/talks/{doc.get('slug', '')}"
            }
            results.append(result)

        return results

    except Exception as e:
        return [{"error": f"Search error: {str(e)}"}]
