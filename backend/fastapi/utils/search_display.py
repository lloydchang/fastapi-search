# File: backend/fastapi/utils/search_display.py

import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_cached_data(cache_dir: str):
    """Load the cached TF-IDF matrix, vectorizer, and documents."""
    # Paths to cached files
    tfidf_data_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Load sparse TF-IDF matrix
    with np.load(tfidf_data_path, allow_pickle=True) as data:
        tfidf_matrix = scipy.sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    logger.info(f"Loaded TF-IDF matrix with shape {tfidf_matrix.shape}")

    # Load vectorizer metadata
    with np.load(tfidf_metadata_path, allow_pickle=True) as data:
        vocabulary = data['vocabulary'].item()
        idf_values = data['idf_values']
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocabulary)
    vectorizer.idf_ = idf_values
    logger.info("Loaded TF-IDF vectorizer metadata")

    # Load document metadata
    with np.load(document_metadata_path, allow_pickle=True) as data:
        documents = data['documents'].tolist()
    logger.info(f"Loaded {len(documents)} documents")

    return tfidf_matrix, vectorizer, documents

def search_documents(query: str, tfidf_matrix, vectorizer: TfidfVectorizer, documents: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for documents matching the query and return top_k results."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    top_docs = []
    for idx in top_indices:
        similarity = similarities[idx]
        if similarity > 0:  # Only include documents with non-zero similarity
            top_docs.append({
                'slug': documents[idx]['slug'],
                'description': documents[idx]['description'],
                'presenterDisplayName': documents[idx]['presenterDisplayName'],
                'sdg_tags': documents[idx].get('sdg_tags', []),
                'similarity': similarity
            })
    return top_docs

def display_search_results(keyword: str, results: List[Dict[str, Any]]):
    """Display the search results in a readable format."""
    print(f"\nSearch Results for Keyword: '{keyword}'")
    print("-" * (25 + len(keyword)))
    if not results:
        print("No results found.")
        return
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(f"Slug: {doc['slug']}")
        print(f"Presenter: {doc['presenterDisplayName']}")
        print(f"Description: {doc['description']}")
        print(f"SDG Tags: {', '.join(doc['sdg_tags'])}")
        print(f"Similarity Score: {doc['similarity']:.4f}")

def main():
    # Define the base and cache directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")

    # Load cached data
    tfidf_matrix, vectorizer, documents = load_cached_data(cache_dir)

    # Define the keywords to search
    keywords_to_search = [
        "TED AI",
        'poverty', 'hunger', 'health', 'education', 'gender', 'water', 'energy', 'work',
        'industry', 'inequality', 'city', 'consumption', 'climate', 'ocean', 'land',
        'peace', 'partnership'
    ]

    # Iterate through each keyword and display search results
    for keyword in keywords_to_search:
        results = search_documents(keyword, tfidf_matrix, vectorizer, documents, top_k=100)
        display_search_results(keyword, results)

if __name__ == "__main__":
    main()
