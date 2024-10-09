# File: api/index.py

import time
import uuid
import os
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.fastapi.services.semantic_search import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Configure CORS
origins = [
    "http://localhost:3000",  # Local frontend
    "https://nextjs-fastapi-wheat-kappa.vercel.app"  # Deployed frontend URL
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Resolve cache directory path
base_dir = Path(__file__).resolve().parent.parent
cache_dir = str(base_dir / "backend" / "fastapi" / "cache")

def load_vocabulary(cache_dir: str) -> Dict[str, int]:
    """Load vocabulary metadata from the precomputed TF-IDF cache."""
    tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')
    metadata = load_cache(tfidf_metadata_path)
    if metadata is None or 'vocabulary' not in metadata:
        raise RuntimeError("TF-IDF metadata not found or corrupted.")
    return metadata['vocabulary'].item()

# Load vocabulary on startup
vocabulary = load_vocabulary(cache_dir)

@lru_cache(maxsize=100)
def cached_semantic_search(query: str, top_n: int = 3) -> List[Dict]:
    """
    Cached wrapper for performing a semantic search.
    This function uses LRU cache to store results for repeated queries.
    """
    print(f"DEBUG: Using LRU cache for query: '{query}'")
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 3) -> List[Dict]:
    """Perform a new semantic search for the given query and return the top `top_n` results."""
    print(f"DEBUG: Performing semantic search for query: '{query}'...")
    # Directly retrieve only the top `top_n` results
    results = semantic_search(query, cache_dir, top_n=top_n)

    # Check if results are available
    if results is None:
        print(f"DEBUG: No results returned for query: '{query}'.")
        return []

    print(f"DEBUG: Retrieved {len(results)} results for query: '{query}'")

    # Return only the top `top_n` results
    return results

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    """Handle the search endpoint by performing a new semantic search."""
    request_uuid = uuid.uuid4()
    search_request_start_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    try:
        # Perform a new search using the LRU cache
        results = cached_semantic_search(query, top_n=3)

        # Ensure `sdg_tags` are included in the results
        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])  # Add empty sdg_tags if not present

    except RuntimeError as e:
        print(f"{request_uuid} [Cache Error] {e}")
        raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Semantic Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Semantic search failed.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    # Return the results
    return {"results": results}
