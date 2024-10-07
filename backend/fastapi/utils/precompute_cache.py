# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.utils.text_processing import preprocess
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.sdg_manager import get_sdg_keywords

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths to data and cache files
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent.parent  # backend/fastapi/utils -> backend/fastapi -> backend
file_path = base_dir / "fastapi" / "data" / "github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_dir = base_dir / "fastapi" / "cache"

# Ensure the cache directory exists
def ensure_cache_directory_exists():
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        logger.info(f"Created cache directory: {cache_dir}")

def precompute_and_save_cache():
    try:
        ensure_cache_directory_exists()

        logger.info("Loading the TEDx dataset...")
        data = load_dataset(str(file_path))
        if not data:
            logger.warning("No data loaded. Exiting precompute process.")
            return

        logger.info("Preprocessing the dataset...")
        documents = [preprocess(doc.get('description', '')) for doc in data]
        texts = [' '.join(doc) for doc in documents]

        if not any(texts):
            logger.warning("All documents are empty after preprocessing. Exiting precompute process.")
            return

        logger.info("Computing TF-IDF vectors...")
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        logger.info("Computing SDG tags...")
        sdg_keywords_dict = get_sdg_keywords()
        sdg_keywords = {sdg: [kw.lower() for kw in kws] for sdg, kws in sdg_keywords_dict.items()}
        sdg_tags = compute_sdg_tags(documents, sdg_keywords, list(sdg_keywords.keys()))

        # Save the TF-IDF matrix
        logger.info("Saving TF-IDF matrix to .npz format...")
        np.savez_compressed(cache_dir / 'tfidf_matrix.npz', tfidf_matrix=tfidf_matrix.toarray())

        # Save the vocabulary and IDF values separately
        logger.info("Saving TF-IDF metadata to .npz format...")
        metadata = {
            "vocabulary": tfidf_vectorizer.vocabulary_,
            "idf": tfidf_vectorizer.idf_
        }
        np.savez_compressed(cache_dir / 'tfidf_metadata.npz', **metadata)

        # Save processed data (with SDG tags)
        logger.info("Saving processed data to .npz format...")
        data_to_save = []
        for idx, doc in enumerate(data):
            doc_fields_to_keep = ['slug', 'description', 'presenter']
            pruned_doc = {field: doc.get(field, '') for field in doc_fields_to_keep}
            pruned_doc['sdg_tags'] = sdg_tags[idx] if idx < len(sdg_tags) else []
            data_to_save.append(pruned_doc)

        # Convert list of dicts to a structured numpy array
        dtype = [('slug', 'U100'), ('description', 'U1000'), ('presenter', 'U100'), ('sdg_tags', 'O')]
        structured_data = np.array([
            (doc['slug'], doc['description'], doc['presenter'], doc['sdg_tags']) for doc in data_to_save
        ], dtype=dtype)

        # **Use 'data' as the key instead of 'structured_data'**
        np.savez_compressed(cache_dir / 'data.npz', data=structured_data)

        logger.info("Precomputation complete and data saved in .npz format!")
    except Exception as e:
        logger.error(f"An error occurred during precomputation: {e}")
        raise

if __name__ == "__main__":
    precompute_and_save_cache()
