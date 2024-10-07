# File: api/index.py

import asyncio
import warnings
import subprocess
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, Query, HTTPException
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

# File paths for data and cache, relative to base directory
file_path = Path("backend/fastapi/data/github-mauropelucchi-tedx_dataset-update_2024-details.csv")
cache_directory = Path("backend/fastapi/cache")
cache_file_path = cache_directory / "tedx_dataset_with_sdg_tags.pkl"

# Function to load resources from cache or run the precompute script
async def load_resources():
    global data, idf_dict, document_tfidf_vectors, resources_initialized

    # Check for cache file existence
    if not cache_file_path.exists():
        print(f"Cache file '{cache_file_path}' not found. Running precompute_cache.py script...")

        # Run the precompute script from the base directory
        precompute_script_path = Path("backend/fastapi/utils/precompute_cache.py")
        try:
            subprocess.run(["python3", str(precompute_script_path)], check=True, cwd=Path(".").resolve())
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Precomputation script failed: {e}")

    # Load the cache file
    print("Loading resources from the single cache file...")
    cache_data = await load_cache(str(cache_file_path))
    if cache_data is None:
        raise RuntimeError("Failed to load cache file even after precomputation.")

    # Unpack the cache data
    data, idf_dict, document_tfidf_vectors = cache_data["data"], cache_data["idf_dict"], cache_data["document_tfidf_vectors"]

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
        raise HTTPException(
            status_code=503, detail="Data or TF-IDF vectors not available. Run `precompute_cache.py`."
        )

    result = await semantic_search(query, data, idf_dict, document_tfidf_vectors)
    return result
