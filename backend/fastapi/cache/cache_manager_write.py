# File: backend/fastapi/cache/cache_manager_write.py

import os
from typing import Any, Optional, Dict
import numpy as np
from scipy.sparse import load_npz, save_npz

def load_cache(cache_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads data from a cache file if it exists.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        Optional[Dict[str, Any]]: Loaded data as a dictionary or None if loading fails.
    """
    if os.path.exists(cache_file_path):
        try:
            if cache_file_path.endswith('tfidf_matrix.npz'):
                # Load as sparse matrix
                data = load_npz(cache_file_path)
                return {'tfidf_matrix': data}
            else:
                # Load as regular .npz
                with np.load(cache_file_path, allow_pickle=True) as data:
                    loaded_data = {key: data[key] for key in data.files}
                    return loaded_data
        except Exception as e:
            print(f"Failed to load cache from {cache_file_path}: {e}")
            return None
    else:
        print(f"Cache file {cache_file_path} does not exist.")
        return None

def save_cache(data: Any, cache_file_path: str) -> None:
    """
    Saves data to a cache file.

    Args:
        data (Any): Data to be cached.
        cache_file_path (str): Path to the cache file.
    """
    try:
        if isinstance(data, dict):
            if 'tfidf_matrix' in data and hasattr(data['tfidf_matrix'], 'save_npz'):
                save_npz(cache_file_path, data['tfidf_matrix'])
            else:
                np.savez_compressed(cache_file_path, **data)
        else:
            np.savez_compressed(cache_file_path, data=data)
        print(f"Cache saved successfully to {cache_file_path}")
    except Exception as e:
        print(f"Failed to save cache to {cache_file_path}: {e}")
