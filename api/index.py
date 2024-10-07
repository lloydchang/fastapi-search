# File: api/index.py

import time
import uuid
from threading import Lock
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.fastapi.services.search_service import semantic_search

from joblib import load
import json

# Generate a script-wide unique ID for general sections
script_uuid = uuid.uuid4()

# Measure the overall script execution time
print(f"{script_uuid} [Script Initialization] Starting script initialization...")
script_initialization_start = time.time()

# 1. Creating the FastAPI App Instance
print(f"{script_uuid} [App Initialization] Starting FastAPI app creation...")
app_start_time = time.time()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app_end_time = time.time()
print(f"{script_uuid} [App Initialization] FastAPI app instance created in {app_end_time - app_start_time:.4f} seconds.")

# 2. Setting Up CORS Configuration
print(f"{script_uuid} [CORS Setup] Starting CORS configuration...")
cors_start_time = time.time()
origins = [
    "http://localhost:3000",  # React app running on localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
cors_end_time = time.time()
print(f"{script_uuid} [CORS Setup] CORS configuration completed in {cors_end_time - cors_start_time:.4f} seconds.")

# 3. Resolving File Paths
print(f"{script_uuid} [Path Resolution] Starting file path resolution...")
path_resolution_start_time = time.time()
base_dir = Path(__file__).resolve().parent.parent
cache_dir = base_dir / "backend" / "fastapi" / "cache"
path_resolution_end_time = time.time()
print(f"{script_uuid} [Path Resolution] File paths resolved in {path_resolution_end_time - path_resolution_start_time:.4f} seconds.")

# Global variables to hold the loaded resources
tfidf_vectorizer = None
tfidf_matrix = None
data = None
resources_initialized = False
load_lock = Lock()

# 4. Loading Resources
print(f"{script_uuid} [Resource Loading] Starting resource loading...")

def load_resources(log_uuid):
    global tfidf_vectorizer, tfidf_matrix, data, resources_initialized

    with load_lock:
        if resources_initialized:
            return

        print(f"{log_uuid} [Resource Loading] Starting to load precomputed data...")
        load_start_time = time.time()
        try:
            # Load tfidf_matrix and tfidf_vectorizer
            tfidf_matrix = load(cache_dir / 'tfidf_matrix.joblib')
            tfidf_vectorizer = load(cache_dir / 'tfidf_vectorizer.joblib')

            # Load data (the documents)
            with open(cache_dir / 'data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)

            resources_initialized = True
            load_end_time = time.time()
            print(f"{log_uuid} [Resource Loading] Loaded precomputed data in {load_end_time - load_start_time:.4f} seconds.")
        except FileNotFoundError as e:
            print(f"{log_uuid} [Resource Loading Error] {e}")
            raise RuntimeError("Failed to load precomputed data due to missing file.")
        except Exception as e:
            print(f"{log_uuid} [Resource Loading Error] Unexpected error: {e}.")
            raise RuntimeError("Failed to load precomputed data due to an unexpected error.")

# 5. Creating Search Endpoint
print(f"{script_uuid} [Endpoint Creation] Starting Search Endpoint creation...")
endpoint_creation_start_time = time.time()

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1)) -> List[Dict]:
    # Generate a unique ID for each request
    request_uuid = uuid.uuid4()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    search_request_start_time = time.time()

    if not resources_initialized:
        print(f"{request_uuid} [Resource Initialization] Starting resource initialization before search...")
        resource_init_start_time = time.time()
        try:
            load_resources(request_uuid)  # Pass request-specific UUID to load_resources()
        except RuntimeError as e:
            print(f"{request_uuid} [Resource Initialization Error] Failed to initialize resources: {e}.")
            raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
        resource_init_end_time = time.time()
        print(f"{request_uuid} [Resource Initialization] Resources initialized in {resource_init_end_time - resource_init_start_time:.4f} seconds.")

    if tfidf_vectorizer is None or tfidf_matrix is None or data is None:
        print(f"{request_uuid} [Search Endpoint Error] Precomputed data is not available.")
        raise HTTPException(
            status_code=503, detail="Precomputed data not available."
        )

    print(f"{request_uuid} [Semantic Search] Starting semantic search processing...")
    semantic_search_start_time = time.time()
    try:
        result = semantic_search(query, tfidf_vectorizer, tfidf_matrix, data)
    except Exception as e:
        print(f"{request_uuid} [Semantic Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Semantic search failed.")
    semantic_search_end_time = time.time()
    print(f"{request_uuid} [Semantic Search] Semantic search completed in {semantic_search_end_time - semantic_search_start_time:.4f} seconds.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    return result

endpoint_creation_end_time = time.time()
print(f"{script_uuid} [Endpoint Creation] Search Endpoint created in {endpoint_creation_end_time - endpoint_creation_start_time:.4f} seconds.")
