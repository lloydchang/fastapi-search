# File: backend/fastapi/utils/precompute_cache.py

import os
import sys
from pathlib import Path
import json
from joblib import dump

current_dir = Path(__file__).resolve().parent
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(current_dir.parent.parent.parent))

from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.utils.text_processing import preprocess
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.sdg_manager import get_sdg_keywords

from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to data and cache files
base_dir = current_dir.parent
file_path = base_dir / "data" / "github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_dir = base_dir / "cache"

# Ensure the cache directory exists
def ensure_cache_directory_exists():
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        print(f"Created cache directory: {cache_dir}")

def precompute_and_save_cache():
    ensure_cache_directory_exists()

    print("Loading the TEDx dataset...")
    data = load_dataset(str(file_path))

    # Optional: Limit the number of documents for testing
    # data = data[:1000]

    print("Preprocessing the dataset...")
    documents = [preprocess(doc.get('description', '')) for doc in data]
    texts = [' '.join(doc) for doc in documents]

    print("Computing TF-IDF vectors...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    print("Computing SDG tags...")
    sdg_keywords_dict = get_sdg_keywords()
    sdg_keywords = {sdg: [kw.lower() for kw in kws] for sdg, kws in sdg_keywords_dict.items()}
    sdg_tags = compute_sdg_tags(documents, sdg_keywords, list(sdg_keywords.keys()))

    # Save the TF-IDF matrix and vectorizer
    print("Saving TF-IDF matrix and vectorizer...")
    dump(tfidf_matrix, cache_dir / 'tfidf_matrix.joblib')
    dump(tfidf_vectorizer, cache_dir / 'tfidf_vectorizer.joblib')

    # Prepare data to save
    print("Preparing data to save...")
    data_to_save = []
    for idx, doc in enumerate(data):
        # Keep only necessary fields
        doc_fields_to_keep = ['slug', 'description', 'presenter']
        pruned_doc = {field: doc.get(field, '') for field in doc_fields_to_keep}
        pruned_doc['sdg_tags'] = sdg_tags[idx] if idx < len(sdg_tags) else []
        data_to_save.append(pruned_doc)

    # Save the processed data (documents with metadata)
    print("Saving processed data...")
    with open(cache_dir / 'data.json', 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False)

    print("Precomputation complete!")

if __name__ == "__main__":
    precompute_and_save_cache()
