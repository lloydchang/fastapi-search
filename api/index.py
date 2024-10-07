# File: api/index.py

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
from backend.fastapi.data.sdg_utils import compute_sdg_tags
from backend.fastapi.services.sdg_manager import get_sdg_keywords
from backend.fastapi.cache.cache_manager import load_cache, save_cache

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
data_cache_path = "backend/fastapi/cache/tedx_dataset.pkl"
documents_cache_path = "backend/fastapi/cache/documents.pkl"
idf_cache_path = "backend/fastapi/cache/idf_dict.pkl"
tfidf_vectors_cache_path = "backend/fastapi/cache/document_tfidf_vectors.pkl"
sdg_tags_cache_path = "backend/fastapi/cache/sdg_tags.pkl"

# Background task to load the necessary resources
async def load_resources():
    global data, idf_dict, document_tfidf_vectors, resources_initialized

    data = await load_dataset(file_path, data_cache_path)

    documents = await load_cache(documents_cache_path)
    if documents is None:
        documents = [preprocess(doc.get('description', '')) for doc in data]
        await save_cache(documents, documents_cache_path)

    idf_dict = await load_cache(idf_cache_path)
    if idf_dict is None:
        idf_dict = compute_idf(documents)
        await save_cache(idf_dict, idf_cache_path)

    document_tfidf_vectors = await load_cache(tfidf_vectors_cache_path)
    if document_tfidf_vectors is None:
        document_tfidf_vectors = [compute_tfidf(compute_tf(doc), idf_dict) for doc in documents]
        await save_cache(document_tfidf_vectors, tfidf_vectors_cache_path)

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
