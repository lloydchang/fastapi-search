# File: backend/fastapi/cache/cache_manager.py

import os
from joblib import dump, load
from typing import Any, Optional

# Ensure the cache directory exists
CACHE_DIRECTORY = os.path.dirname(__file__)

def load_cache(cache_file_path: str) -> Optional[Any]:
    """
    Loads data from a cache file if it exists.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        Optional[Any]: Loaded data or None if loading fails.
    """
    if os.path.exists(cache_file_path):
        try:
            return load(cache_file_path)
        except Exception as e:
            print(f"Failed to load cache from {cache_file_path}: {e}")
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
    try:
        dump(data, cache_file_path)
    except Exception as e:
        print(f"Failed to save cache to {cache_file_path}: {e}")
