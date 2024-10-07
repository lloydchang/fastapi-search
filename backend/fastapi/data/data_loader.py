# File: backend/fastapi/data/data_loader.py

import pandas as pd
from backend.fastapi.utils.logger import logger
from backend.fastapi.cache.cache_manager import load_cache, save_cache

def load_dataset(file_path: str, cache_file_path: str) -> pd.DataFrame:
    """
    Loads the TEDx dataset with caching.

    Args:
        file_path (str): Path to the CSV file.
        cache_file_path (str): Path to the cache file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger.info("Loading the TEDx Dataset with a caching mechanism.")
    data = load_cache(cache_file_path)

    if data is not None:
        logger.info("Dataset loaded from cache.")
    else:
        logger.info("Loading dataset from CSV file.")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Dataset successfully loaded with {len(data)} records.")
            # Cache the dataset for future use
            save_cache(data, cache_file_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")

    return data
