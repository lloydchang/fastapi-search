# File: backend/fastapi/data/sdg_utils.py

from backend.fastapi.utils.logger import logger  # Import the centralized logger

def compute_sdg_tags(documents, sdg_keywords, sdg_names):
    """
    Compute SDG tags for each document based on keyword presence.

    Args:
        documents: List of tokenized documents.
        sdg_keywords: Dictionary of SDG keywords.
        sdg_names: List of SDG names.

    Returns:
        List[List[str]]: List of lists containing SDG tags for each document.
    """
    logger.info("Computing SDG tags based on keyword matching.")

    sdg_tags_list = []
    for tokens in documents:
        tags = []
        token_set = set(tokens)
        for sdg_name, keywords in sdg_keywords.items():
            if token_set.intersection(keywords):
                tags.append(sdg_name)
        if not tags:
            tags.append('sdg0')  # Assign a default SDG if none matched
        sdg_tags_list.append(tags)

    logger.info("SDG tags computed successfully.")
    return sdg_tags_list
