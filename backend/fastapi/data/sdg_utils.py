# File: backend/fastapi/data/sdg_utils.py

from typing import List, Dict

def compute_sdg_tags(documents: List[List[str]], sdg_keywords: Dict[str, List[str]], sdg_names: List[str]) -> List[List[str]]:
    """
    Compute SDG tags for each document based on keyword presence.

    Args:
        documents (List[List[str]]): List of tokenized documents.
        sdg_keywords (Dict[str, List[str]]): Dictionary of SDG keywords.
        sdg_names (List[str]): List of SDG names.

    Returns:
        List[List[str]]: List of lists containing SDG tags for each document.
    """
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

    return sdg_tags_list
