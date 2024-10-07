# File: backend/fastapi/utils/precompute_cache.py

import os
import asyncio
import pickle
from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.utils.text_processing import preprocess, compute_tf, compute_idf, compute_tfidf

# Paths to data and cache files
file_path = os.path.join(os.path.dirname(__file__), "../data/github-mauropelucchi-tedx_dataset-update_2024-details.csv")
cache_file_path = os.path.join(os.path.dirname(__file__), "../cache/tedx_dataset_with_sdg_tags.pkl")

async def precompute_and_save_cache():
    """
    Precompute and save the cache for TEDx data including TF-IDF vectors and SDG tags.
    """
    # Step 1: Load the TEDx Dataset
    print("Loading the TEDx dataset...")
    data = await load_dataset(file_path, cache_file_path)  # Fix: Provide a cache path here

    # Step 2: Preprocess the dataset to create tokenized documents
    print("Preprocessing the dataset...")
    documents = [preprocess(doc.get('description', '')) for doc in data]

    # Step 3: Compute the IDF dictionary
    print("Computing IDF dictionary...")
    idf_dict = compute_idf(documents)

    # Step 4: Compute TF-IDF vectors for the documents
    print("Computing TF-IDF vectors...")
    document_tfidf_vectors = [compute_tfidf(compute_tf(doc), idf_dict) for doc in documents]

    # Step 5: Combine all components into a single dictionary
    cache_data = {
        'data': data,
        'idf_dict': idf_dict,
        'document_tfidf_vectors': document_tfidf_vectors
    }

    # Step 6: Save the single cache file
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    with open(cache_file_path, 'wb') as cache_file:
        pickle.dump(cache_data, cache_file)

    print("Precomputation complete! Resources have been cached successfully.")

if __name__ == "__main__":
    # Run the async function using asyncio
    asyncio.run(precompute_and_save_cache())
