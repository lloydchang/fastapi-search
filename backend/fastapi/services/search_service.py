# File: backend/fastapi/services/search_service.py

from typing import List, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from backend.fastapi.utils.logger import logger  # Import the centralized logger

number_of_results = 100

async def semantic_search(query: str, data: pd.DataFrame, model: SentenceTransformer, sdg_embeddings, top_n: int = number_of_results) -> List[Dict]:
    """
    Performs semantic search on the TEDx dataset.

    Args:
        query (str): The search query.
        data (pd.DataFrame): The dataset containing TEDx talks.
        model (SentenceTransformer): The loaded Sentence-BERT model.
        sdg_embeddings: The precomputed SDG embeddings.
        top_n (int): Number of top results to return.

    Returns:
        List[Dict]: List of search results with metadata.
    """
    logger.info(f"Performing semantic search for the query: '{query}'.")

    # Log the current columns in the data for debugging
    logger.info(f"Available columns in the data: {list(data.columns)}")

    # Check if the model and required data are available
    if model is None or 'description_vector' not in data.columns:
        logger.error(f"Model is None: {model is None}, 'description_vector' in data columns: {'description_vector' in data.columns}")
        logger.error("Model or data not available. Make sure 'description_vector' column is present in the dataset.")
        return [{"error": "Model or data not available. Make sure 'description_vector' column is present in the dataset."}]

    try:
        # Encode the query asynchronously
        logger.info(f"Encoding query: '{query}' using the model.")
        query_vector = await asyncio.to_thread(
            model.encode,
            query,
            clean_up_tokenization_spaces=True,
            convert_to_tensor=True
        )
        query_vector = query_vector.cpu().numpy()
        logger.info(f"Query encoded successfully. Shape: {query_vector.shape}")

        # Convert description vectors to numpy array
        description_vectors_np = np.array([np.array(vec) for vec in data['description_vector']])
        logger.info(f"Converted description vectors to numpy array. Shape: {description_vectors_np.shape}")

        # Compute cosine similarities
        similarities = np.dot(description_vectors_np, query_vector) / (
            np.linalg.norm(description_vectors_np, axis=1) * np.linalg.norm(query_vector)
        )
        logger.info("Cosine similarities computed successfully.")

        # Handle any NaN values resulting from zero division
        similarities = np.nan_to_num(similarities)

        # Get top N indices
        top_indices = np.argsort(-similarities)[:top_n]
        logger.info(f"Top {top_n} indices identified.")

        # Prepare the search results
        results = []
        for idx in top_indices:
            # Check if 'sdg_tags' is present, otherwise use an empty list as a placeholder
            sdg_tags = data.iloc[idx].get('sdg_tags', []) if 'sdg_tags' in data.columns else []

            result = {
                'title': data.iloc[idx]['slug'].replace('_', ' '),
                'description': data.iloc[idx]['description'],
                'presenter': data.iloc[idx]['presenterDisplayName'],
                'sdg_tags': sdg_tags,
                'similarity_score': float(similarities[idx]),
                'url': f"https://www.ted.com/talks/{data.iloc[idx]['slug']}"
            }
            results.append(result)

        logger.info(f"Semantic search completed successfully for query: '{query}'. Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return [{"error": f"Search error: {str(e)}"}]
