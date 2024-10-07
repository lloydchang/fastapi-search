# File: api/index.py

import asyncio
import warnings
from typing import List, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.services.search_service import semantic_search
from backend.fastapi.utils.text_processing import preprocess, compute_tf, compute_idf, compute_tfidf
from backend.fastapi.cache.cache_manager import load_cache

# Create a FastAPI app instance
app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

# Global variables to hold the loaded resources
data = None
idf_dict = None
document_tfidf_vectors = None
resources_initialized = False

# Event to wait for resource initialization
resource_event = asyncio.Event()

# File paths for data and cache
file_path = "backend/fastapi/data/github-mauropelucchi-tedx_dataset-update_2024-details.csv"
data_cache_path = "backend/fastapi/cache/tedx_dataset.pkl"
documents_cache_path = "backend/fastapi/cache/documents.pkl"
idf_cache_path = "backend/fastapi/cache/idf_dict.pkl"
tfidf_vectors_cache_path = "backend/fastapi/cache/document_tfidf_vectors.pkl"

# Load the necessary resources from precomputed caches
async def load_resources():
    global data, idf_dict, document_tfidf_vectors, resources_initialized

    # Step 1: Load from precomputed caches
    data, documents, idf_dict, document_tfidf_vectors = await asyncio.gather(
        load_cache(data_cache_path),
        load_cache(documents_cache_path),
        load_cache(idf_cache_path),
        load_cache(tfidf_vectors_cache_path)
    )

    # Step 2: Check if any resource is missing; if so, raise an error
    if data is None or documents is None or idf_dict is None or document_tfidf_vectors is None:
        raise RuntimeError("One or more required cache files are missing. Run `precompute_cache.py` to generate them.")

    resources_initialized = True
    resource_event.set()

# On startup, load resources in a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources())

# Create a Search Endpoint for TEDx Talks
@app.get("/api/search")
async def search(query: str = Query(..., min_length=1)) -> List[Dict]:
    await resource_event.wait()

    if data is None or idf_dict is None or document_tfidf_vectors is None:
        return [{"error": "Data or TF-IDF vectors not available."}]

    result = await semantic_search(query, data, idf_dict, document_tfidf_vectors)
    return result
