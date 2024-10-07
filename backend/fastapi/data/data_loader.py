# File: backend/fastapi/data/data_loader.py

import csv
from backend.fastapi.utils.logger import logger
from backend.fastapi.cache.cache_manager import load_cache, save_cache

def load_dataset(file_path: str, cache_file_path: str) -> list:
    """
    Loads the TEDx dataset with caching.

    Args:
        file_path (str): Path to the CSV file.
        cache_file_path (str): Path to the cache file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    logger.info("Loading the TEDx Dataset with a caching mechanism.")
    data = load_cache(cache_file_path)

    if data is not None:
        logger.info("Dataset loaded from cache.")
    else:
        logger.info("Loading dataset from CSV file.")
        try:
            data = []
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            logger.info(f"Dataset successfully loaded with {len(data)} records.")
            # Cache the dataset for future use
            save_cache(data, cache_file_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            data = []

    return data
