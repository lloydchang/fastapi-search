# File: backend/fastapi/cache/cache_manager.py

import os
import aiofiles
import pickle
from typing import Any, Optional

# Ensure the cache directory exists
CACHE_DIRECTORY = './backend/fastapi/cache'

async def ensure_cache_directory_exists():
    """
    Ensures that the cache directory exists.
    If it doesn't exist, the directory is created.
    """
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

async def load_cache(cache_file_path: str) -> Optional[Any]:
    """
    Loads data from a cache file if it exists.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        Optional[Any]: Loaded data or None if loading fails.
    """
    await ensure_cache_directory_exists()

    if os.path.exists(cache_file_path):
        try:
            async with aiofiles.open(cache_file_path, 'rb') as cache_file:
                data = await cache_file.read()
                return pickle.loads(data)
        except Exception:
            return None
    else:
        return None

async def save_cache(data: Any, cache_file_path: str) -> None:
    """
    Saves data to a cache file.

    Args:
        data (Any): Data to be cached.
        cache_file_path (str): Path to the cache file.
    """
    await ensure_cache_directory_exists()

    try:
        async with aiofiles.open(cache_file_path, 'wb') as cache_file:
            await cache_file.write(pickle.dumps(data))
    except Exception:
        pass
