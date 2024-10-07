# File: backend/fastapi/data/data_loader.py

import csv
import aiofiles
from typing import List, Dict, Optional
from backend.fastapi.cache.cache_manager import load_cache, save_cache

async def load_dataset(file_path: str, cache_file_path: Optional[str] = None) -> List[Dict]:
    """
    Loads the TEDx dataset with caching.

    Args:
        file_path (str): Path to the CSV file.
        cache_file_path (Optional[str]): Path to the cache file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    # Skip cache loading if cache_file_path is None
    if cache_file_path:
        data = await load_cache(cache_file_path)
        if data is not None:
            return data

    try:
        data = []
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as csvfile:
            content = await csvfile.read()
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                data.append(row)
        # Save the cache only if cache_file_path is provided
        if cache_file_path:
            await save_cache(data, cache_file_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        data = []

    return data
