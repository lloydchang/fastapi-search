# File: backend/fastapi/services/sdg_manager.py

from typing import List, Dict

def get_sdg_keywords() -> Dict[str, List[str]]:
    """
    Retrieves the predefined list of SDG keywords for all 17 SDGs.

    Returns:
        Dict[str, List[str]]: Dictionary mapping SDG names to their keywords.
    """
    try:
        from backend.fastapi.data.sdg_keywords import sdg_keywords
        return sdg_keywords
    except Exception:
        return {}
