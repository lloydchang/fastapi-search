# File: api/index.py

import time
import uuid
import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.fastapi.services.search_service import semantic_search

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
base_dir = Path(__file__).resolve().parent.parent  # api/index.py -> api/ -> backend/fastapi
cache_dir = base_dir / "backend" / "fastapi" / "cache"
print(f"{script_uuid} [Path Resolution] Cache directory set to: {cache_dir}")
path_resolution_end_time = time.time()
print(f"{script_uuid} [Path Resolution] File paths resolved in {path_resolution_end_time - path_resolution_start_time:.4f} seconds.")

# Global variables to hold the loaded resources are now managed within search_service.py

# 4. Creating Search Endpoint
print(f"{script_uuid} [Endpoint Creation] Starting Search Endpoint creation...")
endpoint_creation_start_time = time.time()

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1)) -> List[Dict]:
    # Generate a unique ID for each request
    request_uuid = uuid.uuid4()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    search_request_start_time = time.time()

    try:
        # Perform semantic search
        results = semantic_search(query, str(cache_dir), top_n=1)
    except RuntimeError as e:
        print(f"{request_uuid} [Resource Initialization Error] {e}")
        raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Semantic Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Semantic search failed.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    return results

endpoint_creation_end_time = time.time()
print(f"{script_uuid} [Endpoint Creation] Search Endpoint created in {endpoint_creation_end_time - endpoint_creation_start_time:.4f} seconds.")

script_initialization_end = time.time()
print(f"{script_uuid} [Script Initialization] Script initialized in {script_initialization_end - script_initialization_start:.4f} seconds.")
