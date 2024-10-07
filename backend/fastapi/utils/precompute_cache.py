# File: backend/fastapi/utils/precompute_cache.py

import os
import sys
import asyncio
import pickle
from pathlib import Path

# Ensure the script can import modules properly when run as a standalone script
current_dir = Path(__file__).resolve().parent
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(current_dir.parent.parent.parent))  # Add the root directory of the project to `sys.path`

# Now you can import using absolute paths
from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.utils.text_processing import preprocess, compute_tf, compute_idf, compute_tfidf
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.sdg_manager import get_sdg_keywords

# Paths to data and cache files
base_dir = current_dir.parent
file_path = base_dir / "data" / "github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_dir = base_dir / "cache"
cache_file_path = cache_dir / "tedx_dataset_with_sdg_tags.pkl"

# Ensure the cache directory exists
def ensure_cache_directory_exists():
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        print(f"Created cache directory: {cache_dir}")

async def precompute_and_save_cache():
    # Ensure the cache directory exists before saving the cache file
    ensure_cache_directory_exists()

    # Step 1: Load the TEDx dataset
    print("Loading the TEDx dataset...")
    data = await load_dataset(file_path, None)

    # Step 2: Preprocess the dataset to create tokenized documents
    print("Preprocessing the dataset...")
    documents = [preprocess(doc.get('description', '')) for doc in data]

    # Step 3: Compute the IDF dictionary
    print("Computing IDF dictionary...")
    idf_dict = compute_idf(documents)

    # Step 4: Compute TF-IDF vectors for the documents
    print("Computing TF-IDF vectors...")
    document_tfidf_vectors = [compute_tfidf(compute_tf(doc), idf_dict) for doc in documents]

    # Step 5: Compute SDG tags for each document
    print("Computing SDG tags...")
    sdg_keywords_dict = get_sdg_keywords()
    sdg_keywords = {sdg: [kw.lower() for kw in kws] for sdg, kws in sdg_keywords_dict.items()}
    sdg_tags = compute_sdg_tags(documents, sdg_keywords, list(sdg_keywords.keys()))

    # Step 6: Associate SDG tags with each document
    for idx, doc in enumerate(data):
        doc['sdg_tags'] = sdg_tags[idx] if idx < len(sdg_tags) else []

    # Step 7: Save the complete dataset with SDG tags, IDF dictionary, and TF-IDF vectors to a single cache file
    cache_data = {
        'data': data,
        'idf_dict': idf_dict,
        'document_tfidf_vectors': document_tfidf_vectors,
    }
    print(f"Saving the dataset and computed values to {cache_file_path}...")
    with open(cache_file_path, 'wb') as cache_file:
        pickle.dump(cache_data, cache_file)

    print("Precomputation complete! All resources have been cached successfully.")


if __name__ == "__main__":
    asyncio.run(precompute_and_save_cache())
