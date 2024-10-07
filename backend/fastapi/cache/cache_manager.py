# File: backend/fastapi/cache/cache_manager.py

import pickle
import os
from typing import Any, Optional
from backend.fastapi.utils.logger import logger  # Import the centralized logger

# Ensure the cache directory exists
CACHE_DIRECTORY = './backend/fastapi/cache'


def ensure_cache_directory_exists():
    """
    Ensures that the cache directory exists.
    If it doesn't exist, the directory is created.
    """
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)
        logger.info(f"Cache directory created at {CACHE_DIRECTORY}.")
    else:
        logger.info(f"Cache directory already exists at {CACHE_DIRECTORY}.")


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
        logger.info(f"Loading cached data from {cache_file_path}.")
        try:
            with open(cache_file_path, 'rb') as cache_file:
                data = pickle.load(cache_file)
            logger.info(f"Cached data loaded successfully from {cache_file_path}.")
            return data
        except Exception as e:
            logger.error(f"Error loading cache from {cache_file_path}: {e}")
            return None
    else:
        logger.info(f"Cache file {cache_file_path} does not exist.")
        return None


def save_cache(data: Any, cache_file_path: str) -> None:
    """
    Saves data to a cache file.

    Args:
        data (Any): Data to be cached.
        cache_file_path (str): Path to the cache file.
    """
    ensure_cache_directory_exists()  # Ensure directory exists before saving cache

    logger.info(f"Saving data to cache file {cache_file_path}.")
    try:
        with open(cache_file_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logger.info(f"Data cached successfully at {cache_file_path}.")
    except Exception as e:
        logger.error(f"Error saving cache to {cache_file_path}: {e}")
