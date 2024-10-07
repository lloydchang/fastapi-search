# File: backend/fastapi/data/data_loader.py

import csv
import aiofiles
from typing import List, Dict
from backend.fastapi.cache.cache_manager import load_cache, save_cache

async def load_dataset(file_path: str, cache_file_path: str) -> List[Dict]:
    """
    Loads the TEDx dataset with caching.

    Args:
        file_path (str): Path to the CSV file.
        cache_file_path (str): Path to the cache file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    data = await load_cache(cache_file_path)

    if data is not None:
        return data
    else:
        try:
            data = []
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as csvfile:
                content = await csvfile.read()
                reader = csv.DictReader(content.splitlines())
                for row in reader:
                    data.append(row)
            await save_cache(data, cache_file_path)
        except Exception:
            data = []

    return data
