# File: backend/fastapi/services/sdg_manager.py

from typing import Dict, List

def get_sdg_keywords() -> Dict[str, List[str]]:
    """
    Retrieves the predefined list of SDG keywords for all 17 SDGs.

    Returns:
        Dict[str, List[str]]: Dictionary mapping SDG names to their keywords.
    """
    try:
        from backend.fastapi.data.sdg_keywords import sdg_keywords
        return sdg_keywords
    except Exception as e:
        print(f"Error loading SDG keywords: {e}")
        return {}
