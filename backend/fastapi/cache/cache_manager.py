# File: backend/fastapi/cache/cache_manager.py

import pickle
import os
from typing import Any, Optional

# Ensure the cache directory exists
CACHE_DIRECTORY = './backend/fastapi/cache'

def ensure_cache_directory_exists():
    """
    Ensures that the cache directory exists.
    If it doesn't exist, the directory is created.
    """
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

def load_cache(cache_file_path: str) -> Optional[Any]:
    """
    Loads data from a cache file if it exists.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        Optional[Any]: Loaded data or None if loading fails.
    """
    ensure_cache_directory_exists()  # Ensure directory exists before loading cache

    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as cache_file:
                data = pickle.load(cache_file)
            return data
        except Exception:
            return None
    else:
        return None

def save_cache(data: Any, cache_file_path: str) -> None:
    """
    Saves data to a cache file.

    Args:
        data (Any): Data to be cached.
        cache_file_path (str): Path to the cache file.
    """
    ensure_cache_directory_exists()  # Ensure directory exists before saving cache

    try:
        with open(cache_file_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
    except Exception:
        pass
