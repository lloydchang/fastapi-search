# File: backend/fastapi/api/index.py

import asyncio
import os
import pickle
import warnings
from typing import List, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.search_service import semantic_search
from backend.fastapi.services.sdg_manager import get_sdg_keywords
from backend.fastapi.utils.embedding_utils import encode_descriptions, encode_sdg_keywords
from backend.fastapi.utils.logger import logger

# Create a FastAPI app instance
app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

# Enable CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded resources
vectorizer = None
tfidf_matrix = None
sdg_tfidf_matrix = None
data = None
resources_initialized = False  # Flag to track if resources are fully initialized

# Event to wait for resource initialization
resource_event = asyncio.Event()

# Suppress warnings for specific libraries
warnings.filterwarnings("ignore", category=FutureWarning)

# File paths for data and cache
file_path = "backend/fastapi/data/github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_file_path = "backend/fastapi/cache/tedx_dataset.pkl"
vectorizer_cache_path = "backend/fastapi/cache/tfidf_vectorizer.pkl"
tfidf_matrix_cache_path = "backend/fastapi/cache/tfidf_matrix.pkl"
sdg_tfidf_matrix_cache_path = "backend/fastapi/cache/sdg_tfidf_matrix.pkl"
sdg_tags_cache_path = "backend/fastapi/cache/sdg_tags.pkl"

# Background task to load the necessary resources
async def load_resources():
    global vectorizer, tfidf_matrix, sdg_tfidf_matrix, data, resources_initialized

    # Load TEDx Dataset
    logger.info("Loading TEDx dataset.")
    data = load_dataset(file_path, cache_file_path)
    logger.info(f"TEDx dataset loaded successfully! Data: {data is not None}")

    # Check if 'sdg_tags' column is in the dataset and add if missing
    if 'sdg_tags' not in data.columns:
        logger.info("Adding missing 'sdg_tags' column to the dataset with default empty lists.")
        data['sdg_tags'] = [[] for _ in range(len(data))]

    # Load or create TF-IDF vectorizer and matrix
    if os.path.exists(vectorizer_cache_path) and os.path.exists(tfidf_matrix_cache_path):
        logger.info("Loading TF-IDF vectorizer and matrix from cache.")
        with open(vectorizer_cache_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(tfidf_matrix_cache_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        logger.info("TF-IDF vectorizer and matrix loaded from cache.")
    else:
        logger.info("Creating TF-IDF vectorizer and matrix.")
        descriptions = data['description'].fillna('').tolist()
        vectorizer, tfidf_matrix = encode_descriptions(descriptions)
        # Save vectorizer and matrix to cache
        with open(vectorizer_cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(tfidf_matrix_cache_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        logger.info("TF-IDF vectorizer and matrix created and cached.")

    # Load or compute SDG TF-IDF vectors
    if os.path.exists(sdg_tfidf_matrix_cache_path):
        logger.info("Loading SDG TF-IDF matrix from cache.")
        with open(sdg_tfidf_matrix_cache_path, 'rb') as f:
            sdg_tfidf_matrix = pickle.load(f)
    else:
        logger.info("Encoding SDG keywords.")
        sdg_keywords = get_sdg_keywords()
        sdg_keyword_list = [', '.join(keywords) for keywords in sdg_keywords.values()]
        sdg_tfidf_matrix = encode_sdg_keywords(sdg_keyword_list, vectorizer)
        with open(sdg_tfidf_matrix_cache_path, 'wb') as f:
            pickle.dump(sdg_tfidf_matrix, f)
        logger.info("SDG TF-IDF matrix created and cached.")

    # Compute or load SDG Tags
    if os.path.exists(sdg_tags_cache_path):
        logger.info("Loading cached SDG tags.")
        with open(sdg_tags_cache_path, 'rb') as f:
            data['sdg_tags'] = pickle.load(f)
        logger.info("SDG tags loaded from cache.")
    else:
        logger.info("Computing SDG tags.")
        sdg_keywords = get_sdg_keywords()
        sdg_names = list(sdg_keywords.keys())
        data['sdg_tags'] = compute_sdg_tags(tfidf_matrix, sdg_tfidf_matrix, sdg_names)
        with open(sdg_tags_cache_path, 'wb') as f:
            pickle.dump(data['sdg_tags'], f)
        logger.info("SDG tags computed and cached successfully.")

    # Set the resources initialized flag and notify waiting coroutines
    resources_initialized = True
    resource_event.set()  # Signal that resources are ready
    logger.info("All resources are fully loaded and ready for use.")

# On startup, load resources in a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources())

# Create a Search Endpoint for TEDx Talks
@app.get("/api/search")
async def search(query: str = Query(..., min_length=1)) -> List[Dict]:
    await resource_event.wait()  # Wait until resources are fully initialized
    logger.info(f"Search request received. Data is None: {data is None}")

    if data is None or vectorizer is None or tfidf_matrix is None:
        logger.error("Data or TF-IDF vectorizer not available.")
        return [{"error": "Data or TF-IDF vectorizer not available."}]

    logger.info(f"Performing semantic search for the query: '{query}'.")
    try:
        result = semantic_search(query, data, vectorizer, tfidf_matrix)
        return result
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return [{"error": str(e)}]
