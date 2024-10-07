# File: backend/fastapi/data/data_loader.py

import csv
from typing import List, Dict
from backend.fastapi.cache.cache_manager import load_cache, save_cache

def load_dataset(file_path: str, cache_file_path: str) -> List[Dict]:
    """
    Loads the TEDx dataset with caching.

    Args:
        file_path (str): Path to the CSV file.
        cache_file_path (str): Path to the cache file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    data = load_cache(cache_file_path)

    if data is not None:
        return data
    else:
        try:
            data = []
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            # Cache the dataset for future use
            save_cache(data, cache_file_path)
        except Exception:
            data = []

    return data
