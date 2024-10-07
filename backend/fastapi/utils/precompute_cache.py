# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from backend.fastapi.cache.cache_manager import save_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def precompute_cache():
    cache_dir = '/Users/lloyd/github/fastapi-search/backend/fastapi/cache'
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Created cache directory: {cache_dir}")

    # Load the dataset
    from backend.fastapi.data.data_loader import load_dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'github-mauropelucchi-tedx_dataset-update_2024-details.csv')  # Adjust path as needed
    data_to_save = load_dataset(dataset_path)

    # Check dataset size
    if len(data_to_save) < 2:
        logger.warning("Dataset contains less than two documents. Please provide a larger dataset for meaningful TF-IDF computation.")

    # Extract descriptions
    descriptions = [doc['description'] for doc in data_to_save]

    logger.info("Preprocessing the dataset...")

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=30603, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    except ValueError as ve:
        logger.error(f"TF-IDF Vectorization failed: {ve}")
        raise ve

    vocabulary = vectorizer.vocabulary_
    idf_values = vectorizer.idf_

    logger.info(f"TF-IDF vectors computed. Matrix shape: {tfidf_matrix.shape}")

    # Compute SDG tags
    from backend.fastapi.services.sdg_manager import get_sdg_keywords
    from backend.fastapi.data.sdg_utils import compute_sdg_tags

    sdg_keywords = get_sdg_keywords()
    sdg_tags_computed = compute_sdg_tags(descriptions=descriptions, sdg_keywords=sdg_keywords)

    # Assign SDG tags to documents
    for idx, doc in enumerate(data_to_save):
        doc['sdg_tags'] = sdg_tags_computed[idx]

    logger.info("SDG tags computed and assigned to documents.")

    # Save TF-IDF matrix
    tfidf_matrix_path = os.path.join(cache_dir, 'tfidf_matrix.npz')
    save_npz(tfidf_matrix_path, tfidf_matrix)
    logger.info(f"TF-IDF matrix saved to {tfidf_matrix_path}")

    # Save TF-IDF metadata
    tfidf_metadata = {
        'vocabulary': vocabulary,
        'idf': idf_values
    }
    tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')
    save_cache(tfidf_metadata, tfidf_metadata_path)
    logger.info(f"TF-IDF metadata saved to {tfidf_metadata_path}")

    # Save processed document data
    dtype = [('slug', 'U100'), ('description', 'U1000'), ('presenter', 'U100'), ('sdg_tags', 'O')]
    structured_data = np.array([
        (
            str(doc.get('slug', 'Unknown_Slug')),
            str(doc.get('description', 'No description provided.')),
            str(doc.get('presenter', 'Unknown')),
            doc['sdg_tags']
        ) for doc in data_to_save
    ], dtype=dtype)

    data_path = os.path.join(cache_dir, 'data.npz')
    save_cache({'data': structured_data}, data_path)
    logger.info(f"Processed document data saved to {data_path}")

    logger.info("Precomputation complete and data saved in .npz format!")

if __name__ == "__main__":
    precompute_cache()
