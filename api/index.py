# File: api/index.py

import asyncio
import os
import subprocess
from typing import List, Dict
from fastapi import FastAPI, Query, HTTPException
from backend.fastapi.cache.cache_manager import load_cache
from backend.fastapi.services.search_service import semantic_search

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
cache_file_path = "backend/fastapi/cache/tedx_dataset_with_sdg_tags.pkl"
precompute_script_path = "backend/fastapi/utils/precompute_cache.py"

# Load resources from the single cache file or trigger precompute
async def load_resources():
    global data, idf_dict, document_tfidf_vectors, resources_initialized

    # Step 1: Check if the cache file exists
    if not os.path.exists(cache_file_path):
        print(f"Cache file '{cache_file_path}' not found. Running precompute_cache.py script...")

        # Run the precompute script with the correct PYTHONPATH
        try:
            subprocess.run(
                ["python3", precompute_script_path],
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Precomputation script failed: {e}")

    # Step 2: Attempt to load the resources from the single cache file
    print("Loading resources from the single cache file...")
    cache_data = await load_cache(cache_file_path)

    # Step 3: Unpack the cached data into the corresponding variables
    if cache_data:
        data, idf_dict, document_tfidf_vectors = cache_data['data'], cache_data['idf_dict'], cache_data['document_tfidf_vectors']
        resources_initialized = True
        resource_event.set()
    else:
        raise RuntimeError("Failed to load cache after precomputation. Please check the precompute script.")

# On startup, load resources in a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources())

# Create a Search Endpoint for TEDx Talks
@app.get("/api/search")
async def search(query: str = Query(..., min_length=1)) -> List[Dict]:
    await resource_event.wait()

    if not resources_initialized or data is None or idf_dict is None or document_tfidf_vectors is None:
        raise HTTPException(
            status_code=503, detail="Resources not available. Check precomputation and cache loading."
        )

    result = await semantic_search(query, data, idf_dict, document_tfidf_vectors)
    return result
