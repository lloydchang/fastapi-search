# File: api/index.py

import time
import uuid
import os

from pathlib import Path
from typing import List, Dict, Optional, Union

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from functools import lru_cache
from rapidfuzz import process, fuzz  # Ensure rapidfuzz is installed if using spell-checking

from backend.fastapi.services.search_service import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache

# Initialize FastAPI app
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Configure CORS
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

# Resolve cache directory path
base_dir = Path(__file__).resolve().parent.parent
cache_dir = base_dir / "backend" / "fastapi" / "cache"
cache_dir = str(cache_dir)
# print(cache_dir)

# Load vocabulary for spell-checking
def load_vocabulary(cache_dir: str) -> Dict[str, int]:
    tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')  # Ensure correct filename
    metadata = load_cache(tfidf_metadata_path)
    if metadata is None or 'vocabulary' not in metadata:
        raise RuntimeError("TF-IDF metadata not found or corrupted.")
    vocabulary = metadata['vocabulary'].item()
    return vocabulary

vocabulary = load_vocabulary(cache_dir)

# Implement Query Caching using lru_cache
@lru_cache(maxsize=1024)
def get_cached_results(query: str) -> List[Dict]:
    return semantic_search(query, cache_dir, top_n=1)

def get_spell_suggestions(term: str, vocabulary: Dict[str, int], threshold: int = 80) -> Optional[str]:
    """
    Suggests a spell correction for a given term based on similarity.

    Args:
        term (str): The term to correct.
        vocabulary (Dict[str, int]): The vocabulary dictionary.
        threshold (int): The minimum similarity score to consider a suggestion.

    Returns:
        Optional[str]: The suggested correction or None if no suitable suggestion found.
    """
    suggestions = process.extract(term, vocabulary.keys(), scorer=fuzz.WRatio, limit=1)
    for suggestion, score, _ in suggestions:
        if score >= threshold:
            return suggestion
    return None

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    # Generate a unique ID for each request
    request_uuid = uuid.uuid4()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    search_request_start_time = time.time()

    try:
        # Retrieve cached results or perform a new search
        results = get_cached_results(query)

        if not results:
            # Identify unmatched terms
            tokens = query.lower().split()
            unmatched_terms = [token for token in tokens if token not in vocabulary]

            if unmatched_terms:
                # Suggest corrections for unmatched terms
                suggestions = {term: get_spell_suggestions(term, vocabulary) for term in unmatched_terms}

                # Build a user-friendly message
                message_parts = ["No results found for your query."]
                for term, suggestion in suggestions.items():
                    if suggestion:
                        message_parts.append(f"Did you mean '{suggestion}' instead of '{term}'?")
                    else:
                        message_parts.append(f"No suggestions found for '{term}'.")

                message = " ".join(message_parts)
            else:
                message = "No results found for your query. Please try different keywords."

            print(f"{request_uuid} [Search Endpoint Handling] {message}")
            return JSONResponse(
                status_code=200,
                content={"message": message}
            )

    except RuntimeError as e:
        print(f"{request_uuid} [Resource Initialization Error] {e}")
        raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Semantic Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Semantic search failed.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    # Return results under the "results" key to maintain consistent response structure
    return {"results": results}