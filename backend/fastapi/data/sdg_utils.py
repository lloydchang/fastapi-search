# File: backend/fastapi/data/sdg_utils.py

from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from backend.fastapi.utils.logger import logger  # Import the centralized logger
import numpy as np

def compute_sdg_tags(tfidf_matrix, sdg_tfidf_matrix, sdg_names: List[str]) -> List[List[str]]:
    """
    Compute SDG tags for each TEDx talk based on cosine similarities.

    Args:
        tfidf_matrix: TF-IDF matrix of descriptions.
        sdg_tfidf_matrix: TF-IDF matrix of SDG keywords.
        sdg_names (List[str]): List of SDG names.

    Returns:
        List[List[str]]: List of lists containing SDG tags for each TEDx talk.
    """
    logger.info("Computing SDG tags based on cosine similarities.")

    try:
        # Compute cosine similarities between descriptions and SDG keywords
        similarities = cosine_similarity(tfidf_matrix, sdg_tfidf_matrix)
        logger.info("Cosine similarities between descriptions and SDG keywords computed successfully.")

        sdg_tags_list = []
        for row in similarities:
            sdg_indices = np.where(row > 0.1)[0]  # Adjust the threshold as needed
            if len(sdg_indices) == 0:
                # Get the index of the highest similarity
                sdg_indices = [np.argmax(row)]

            sdg_tags = [sdg_names[i] for i in sdg_indices]
            sdg_tags_list.append(sdg_tags)
        logger.info("SDG tags computed successfully.")
    except Exception as e:
        logger.error(f"Error computing SDG tags: {e}")
        sdg_tags_list = []
    return sdg_tags_list
