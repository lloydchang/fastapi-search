# File: backend/fastapi/utils/embedding_utils.py

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from backend.fastapi.utils.logger import logger  # Import the centralized logger

def encode_descriptions(descriptions: List[str], model: SentenceTransformer) -> List[List[float]]:
    """
    Encodes a list of descriptions using the provided model.

    Args:
        descriptions (List[str]): List of TEDx talk descriptions.
        model (SentenceTransformer): The loaded Sentence-BERT model.

    Returns:
        List[List[float]]: List of encoded description vectors.
    """
    logger.info("Encoding TEDx talk descriptions.")
    try:
        # Batch encode descriptions for efficiency
        encoded = model.encode(descriptions, clean_up_tokenization_spaces=True, convert_to_tensor=True, batch_size=32)
        encoded_list = encoded.cpu().numpy().tolist()
        logger.info("Descriptions encoded successfully.")
        return encoded_list
    except Exception as e:
        logger.error(f"Error encoding descriptions: {e}")
        return []

def encode_sdg_keywords(sdg_keyword_list: List[str], model: SentenceTransformer) -> Optional[List[List[float]]]:
    """
    Encodes a list of SDG keywords using the provided model.

    Args:
        sdg_keyword_list (List[str]): List of SDG keyword strings.
        model (SentenceTransformer): The loaded Sentence-BERT model.

    Returns:
        Optional[List[List[float]]]: List of encoded SDG keyword vectors or None if encoding fails.
    """
    logger.info("Encoding SDG keywords.")
    try:
        # Batch encode SDG keywords for efficiency
        sdg_embeddings = model.encode(sdg_keyword_list, convert_to_tensor=True, batch_size=16)
        sdg_embeddings_list = sdg_embeddings.cpu().numpy().tolist()
        logger.info("SDG keywords encoded successfully.")
        return sdg_embeddings_list
    except Exception as e:
        logger.error(f"Error encoding SDG keywords: {e}")
        return None
