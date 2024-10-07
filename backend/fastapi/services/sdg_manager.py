# File: backend/fastapi/services/sdg_manager.py

from typing import List, Dict
from backend.fastapi.utils.logger import logger  # Import the centralized logger

def get_sdg_keywords() -> Dict[str, List[str]]:
    """
    Retrieves the predefined list of SDG keywords for all 17 SDGs.

    Returns:
        Dict[str, List[str]]: Dictionary mapping SDG names to their keywords.
    """
    logger.info("Retrieving SDG keywords.")
    # Assuming sdg_keywords.py contains a dictionary named sdg_keywords
    try:
        from backend.fastapi.data.sdg_keywords import sdg_keywords
        logger.info("SDG keywords retrieved successfully.")
        return sdg_keywords
    except Exception as e:
        logger.error(f"Error retrieving SDG keywords: {e}")
        return {}
