# File: backend/fastapi/models/model_definitions.py

from sentence_transformers import SentenceTransformer
from backend.fastapi.utils.logger import logger  # Import the centralized logger

def load_model(model_name: str = 'paraphrase-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Loads the Sentence-BERT model.

    Args:
        model_name (str): The name of the Sentence-BERT model to load.

    Returns:
        SentenceTransformer: The loaded Sentence-BERT model.
    """
    logger.info(f"Loading Sentence-BERT model: {model_name}.")
    try:
        model = SentenceTransformer(model_name)
        logger.info("Sentence-BERT model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Error initializing Sentence-BERT model: {e}")
        return None
