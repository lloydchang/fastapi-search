# File: backend/fastapi/services/search_service.py

from typing import List, Dict
from backend.fastapi.utils.text_processing import preprocess, compute_tf, compute_tfidf, cosine_similarity

number_of_results = 1  # Set to return only 1 result

async def semantic_search(query: str, data: list, idf_dict, document_tfidf_vectors, top_n: int = number_of_results) -> List[Dict]:
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
    try:
        query_tokens = preprocess(query)
        query_tf = compute_tf(query_tokens)
        query_tfidf = compute_tfidf(query_tf, idf_dict)

        similarities = []
        for idx, doc_tfidf in enumerate(document_tfidf_vectors):
            sim = cosine_similarity(query_tfidf, doc_tfidf)
            similarities.append((idx, sim))

        # Sort by similarity and get top 1 result
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_n]]

        # Prepare the single top search result
        results = []
        for idx in top_indices:
            doc = data[idx]
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

        return results

    except Exception as e:
        return [{"error": f"Search error: {str(e)}"}]
