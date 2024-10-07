# File: backend/fastapi/api/index.py

import asyncio
import warnings
from typing import List, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.services.search_service import semantic_search
from backend.fastapi.utils.text_processing import (
    preprocess,
    compute_tf,
    compute_idf,
    compute_tfidf,
)
from backend.fastapi.utils.logger import logger

# Add imports for SDG tagging
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.sdg_manager import get_sdg_keywords

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
data = None
idf_dict = None
document_tfidf_vectors = None
resources_initialized = False  # Flag to track if resources are fully initialized

# Event to wait for resource initialization
resource_event = asyncio.Event()

# Suppress warnings for specific libraries
warnings.filterwarnings("ignore", category=FutureWarning)

# File paths for data and cache
file_path = "backend/fastapi/data/github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_file_path = "backend/fastapi/cache/tedx_dataset.pkl"

# Background task to load the necessary resources
async def load_resources():
    global data, idf_dict, document_tfidf_vectors, resources_initialized

    # Load TEDx Dataset
    logger.info("Loading TEDx dataset.")
    data = load_dataset(file_path, cache_file_path)
    logger.info(f"TEDx dataset loaded successfully! Data: {data is not None}")

    # Preprocess documents
    documents = [preprocess(doc.get('description', '')) for doc in data]
    logger.info("Documents preprocessed.")

    # Compute IDF dictionary
    idf_dict = compute_idf(documents)
    logger.info("IDF dictionary computed.")

    # Compute TF-IDF vectors for documents
    document_tfidf_vectors = [compute_tfidf(compute_tf(doc), idf_dict) for doc in documents]
    logger.info("Document TF-IDF vectors computed.")

    # Compute SDG tags
    sdg_keywords_dict = get_sdg_keywords()
    sdg_keywords = {sdg: [kw.lower() for kw in kws] for sdg, kws in sdg_keywords_dict.items()}
    sdg_names = list(sdg_keywords.keys())
    data_with_sdg_tags = []

    for idx, doc_tokens in enumerate(documents):
        sdg_tags = compute_sdg_tags([doc_tokens], sdg_keywords, sdg_names)[0]
        data[idx]['sdg_tags'] = sdg_tags
        data_with_sdg_tags.append(data[idx])

    data = data_with_sdg_tags
    logger.info("SDG tags assigned to documents.")

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

    if data is None or idf_dict is None or document_tfidf_vectors is None:
        logger.error("Data or TF-IDF vectors not available.")
        return [{"error": "Data or TF-IDF vectors not available."}]

    logger.info(f"Performing semantic search for the query: '{query}'.")
    try:
        result = semantic_search(query, data, idf_dict, document_tfidf_vectors)
        return result
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return [{"error": str(e)}]
