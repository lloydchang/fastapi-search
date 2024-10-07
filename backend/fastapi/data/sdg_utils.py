# File: backend/fastapi/data/sdg_utils.py

import torch
from typing import List
from backend.fastapi.utils.logger import logger  # Import the centralized logger

def compute_sdg_tags(cosine_similarities: torch.Tensor, sdg_names: List[str]) -> List[List[str]]:
    """
    Compute SDG tags for each TEDx talk based on cosine similarities.

    Args:
        cosine_similarities (torch.Tensor): Tensor containing cosine similarities between descriptions and SDG keywords.
        sdg_names (List[str]): List of SDG names.

    Returns:
        List[List[str]]: List of lists containing SDG tags for each TEDx talk.
    """
    logger.info("Computing SDG tags based on cosine similarities.")
    sdg_tags_list = []
    try:
        for row in cosine_similarities:
            sdg_indices = torch.where(row > 0.5)[0]
            if len(sdg_indices) == 0:
                top_n = row.topk(1).indices
                sdg_indices = top_n

            sdg_tags = [f"sdg{i.item() + 1}" for i in sdg_indices]  # Generates tags like 'sdg1', 'sdg2', etc.
            sdg_tags_list.append(sdg_tags)
        logger.info("SDG tags computed successfully.")
    except Exception as e:
        logger.error(f"Error computing SDG tags: {e}")
    return sdg_tags_list
