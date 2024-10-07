# File: backend/fastapi/cache/cache_manager_read.py

import os
from typing import Any, Optional, Dict
import numpy as np

def load_cache(cache_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads data from a cache file without using scipy.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        Optional[Dict[str, Any]]: Loaded data as a dictionary or None if loading fails.
    """
    if os.path.exists(cache_file_path):
        try:
            if cache_file_path.endswith('tfidf_matrix.npz'):
                # Load sparse matrix components
                with np.load(cache_file_path, allow_pickle=True) as data:
                    tfidf_data = data['data']
                    tfidf_indices = data['indices']
                    tfidf_indptr = data['indptr']
                    tfidf_shape = tuple(data['shape'])
                # Convert to dense matrix
                tfidf_matrix = np.zeros(tfidf_shape)
                for row in range(tfidf_shape[0]):
                    start = tfidf_indptr[row]
                    end = tfidf_indptr[row + 1]
                    indices = tfidf_indices[start:end]
                    data_vals = tfidf_data[start:end]
                    tfidf_matrix[row, indices] = data_vals
                return {'tfidf_matrix': tfidf_matrix}
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
    Placeholder function. Saving cache in runtime does not require writing.

    Args:
        data (Any): Data to be cached.
        cache_file_path (str): Path to the cache file.
    """
    raise NotImplementedError("Runtime cache manager does not support writing cache.")
