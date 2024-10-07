# File: backend/fastapi/data/sdg_utils.py

import re
from typing import List, Dict

def compute_sdg_tags(descriptions: List[str], sdg_keywords: Dict[str, List[str]]) -> List[List[str]]:
    """
    Compute SDG tags for each document based on keyword presence.

    Args:
        descriptions (List[str]): List of document descriptions.
        sdg_keywords (Dict[str, List[str]]): Dictionary mapping SDGs to their keywords.

    Returns:
        List[List[str]]: List of SDG tags for each document.
    """
    sdg_tags_list = []
    for description in descriptions:
        tags = []
        # Normalize the description
        description_lower = description.lower()
        # Tokenize the description
        tokens = re.findall(r'\b\w+\b', description_lower)
        token_set = set(tokens)
        for sdg, keywords in sdg_keywords.items():
            for keyword in keywords:
                # For multi-word keywords, perform a substring search
                if ' ' in keyword:
                    if keyword in description_lower:
                        tags.append(sdg)
                        break  # Avoid duplicate tags for the same SDG
                else:
                    if keyword in token_set:
                        tags.append(sdg)
                        break
        # Exclude 'sdg0' by not assigning any default tag
        sdg_tags_list.append(tags)
    return sdg_tags_list
