# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, List, Dict
from backend.fastapi.data.sdg_keywords import sdg_keywords  # Import SDG keywords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_tedx_documents(csv_file_path: str) -> List[Dict[str, str]]:
    """Load TEDx talks from the provided CSV file and extract metadata and text content."""
    tedx_df = pd.read_csv(csv_file_path)
    if 'description' not in tedx_df.columns or 'slug' not in tedx_df.columns:
        raise ValueError(f"Required columns 'description' or 'slug' not found in the CSV file {csv_file_path}")
    
    documents = tedx_df[['slug', 'description', 'presenterDisplayName']].dropna().to_dict('records')
    logger.info(f"Loaded {len(documents)} TEDx documents from the CSV file.")
    return documents

def create_tfidf_matrix(documents: List[Dict[str, str]]) -> Any:
    """Create a sparse TF-IDF matrix from the combined 'description' and 'slug' fields."""
    # Combine slug and description fields to improve semantic search capabilities.
    combined_texts = [f"{doc['slug']} {doc['description']}" for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    logger.info(f"TF-IDF matrix created. Shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

def get_sdg_tags_for_documents(documents: List[Dict[str, str]], sdg_keywords: Dict[str, List[str]]) -> None:
    """Assign SDG tags to documents based on semantic similarity to SDG keywords."""
    # Flatten SDG keywords for vectorization
    sdg_keyword_list = [keyword for keywords in sdg_keywords.values() for keyword in keywords]
    
    # Create TF-IDF vectorizer and transform SDG keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    sdg_tfidf_matrix = vectorizer.fit_transform(sdg_keyword_list)

    for doc in documents:
        description_vector = vectorizer.transform([doc['description']])
        # Calculate cosine similarity with SDG keywords
        cosine_similarities = cosine_similarity(description_vector, sdg_tfidf_matrix).flatten()
        
        # Assign SDG tags based on high similarity
        matched_tags = []
        for i in np.argsort(cosine_similarities)[::-1]:  # Sort indices in descending order
            if cosine_similarities[i] > 0.1:  # Threshold can be adjusted
                # Identify which SDG tag this keyword belongs to
                for sdg, keywords in sdg_keywords.items():
                    if i < len(keywords):
                        matched_tags.append(sdg)
                        break
            else:
                break  # Stop if the similarity is below the threshold

        # If no tags matched, find the closest one
        if not matched_tags:
            closest_index = np.argmax(cosine_similarities)  # Find index of the highest similarity
            closest_sdg = list(sdg_keywords.keys())[closest_index // len(list(sdg_keywords.values())[0])]  # Determine SDG tag
            matched_tags.append(closest_sdg)  # Assign the closest SDG tag

        doc['sdg_tags'] = matched_tags  # Add matched SDG tags to the document
        logger.info(f"Document '{doc['slug']}' assigned SDG tags: {matched_tags}")  # Log the assigned tags

def save_sparse_matrix(tfidf_matrix, cache_dir: str):
    """Save sparse matrix in a numpy-compatible format."""
    tfidf_data_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    np.savez_compressed(
        tfidf_data_path,
        data=tfidf_matrix.data,
        indices=tfidf_matrix.indices,
        indptr=tfidf_matrix.indptr,
        shape=tfidf_matrix.shape
    )
    logger.info(f"Sparse TF-IDF matrix components saved to {tfidf_data_path}")
    # Debug: Confirm saved keys
    with np.load(tfidf_data_path) as data:
        logger.info(f"Saved sparse matrix keys: {data.files}")

def save_tfidf_components(tfidf_matrix, vectorizer: TfidfVectorizer, documents: List[Dict[str, str]], cache_dir: str):
    """Save the TF-IDF matrix and metadata in a format compatible with numpy."""
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Save sparse matrix components
    save_sparse_matrix(tfidf_matrix, cache_dir)

    # Save metadata (vocabulary and IDF values)
    np.savez_compressed(
        tfidf_metadata_path,
        vocabulary=vectorizer.vocabulary_,
        idf_values=vectorizer.idf_
    )
    logger.info(f"TF-IDF metadata saved to {tfidf_metadata_path}")

    # Save document metadata including SDG tags
    np.savez_compressed(document_metadata_path, documents=documents)
    logger.info(f"Document metadata saved to {document_metadata_path}")

def precompute_cache():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")
    csv_file_path = os.path.join(base_dir, "data", "tedx_talks.csv")

    # Load TEDx documents
    documents = load_tedx_documents(csv_file_path)

    # Create the TF-IDF matrix using both 'slug' and 'description'
    tfidf_matrix, vectorizer = create_tfidf_matrix(documents)

    # Get SDG tags for each document based on semantic matching
    get_sdg_tags_for_documents(documents, sdg_keywords)

    # Save the generated components
    os.makedirs(cache_dir, exist_ok=True)
    save_tfidf_components(tfidf_matrix, vectorizer, documents, cache_dir)
    logger.info("Precompute cache written successfully to the cache directory.")

if __name__ == "__main__":
    precompute_cache()
