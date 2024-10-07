# File: backend/fastapi/utils/embedding_utils.py

from sklearn.feature_extraction.text import TfidfVectorizer
from backend.fastapi.utils.logger import logger  # Import the centralized logger

def encode_descriptions(descriptions):
    """
    Encodes a list of descriptions using TF-IDF vectorizer.

    Args:
        descriptions (List[str]): List of TEDx talk descriptions.

    Returns:
        TfidfVectorizer, sparse matrix: The vectorizer and the TF-IDF matrix.
    """
    logger.info("Encoding TEDx talk descriptions using TF-IDF.")
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        logger.info("Descriptions encoded successfully.")
        return vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Error encoding descriptions: {e}")
        return None, None

def encode_sdg_keywords(sdg_keyword_list, vectorizer):
    """
    Encodes a list of SDG keywords using the provided TF-IDF vectorizer.

    Args:
        sdg_keyword_list (List[str]): List of SDG keyword strings.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer trained on the descriptions.

    Returns:
        sparse matrix: The TF-IDF matrix for SDG keywords.
    """
    logger.info("Encoding SDG keywords using TF-IDF.")
    try:
        sdg_tfidf_matrix = vectorizer.transform(sdg_keyword_list)
        logger.info("SDG keywords encoded successfully.")
        return sdg_tfidf_matrix
    except Exception as e:
        logger.error(f"Error encoding SDG keywords: {e}")
        return None
