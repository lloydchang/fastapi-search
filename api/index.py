# File: api/index.py

import time
import uuid
import os
import re
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from backend.fastapi.services.semantic_search import semantic_search  # Ensure we're using the correct import
from backend.fastapi.cache.cache_manager_read import load_cache
from backend.fastapi.data.sdg_keywords import sdg_keywords

# Initialize FastAPI app
app = FastAPI(docs_url="/api/search/docs", redoc_url="/api/search/redoc", openapi_url="/api/search/openapi.json")

# Configure CORS
origins = [
    "*"  # Any origin (not recommended for production)
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

@lru_cache(maxsize=1000)  # Increased cache size for better performance
def cached_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    """Cached wrapper for performing a semantic search."""
    print(f"DEBUG: Using LRU cache for query: '{query}'")
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    """Perform a new semantic search for the given query and return the top `top_n` results."""
    print(f"DEBUG: Performing semantic search for query: '{query}'...")
    results = semantic_search(query, cache_dir, top_n=top_n)

    if results is None:
        print(f"DEBUG: No results returned for query: '{query}'.")
        return []

    print(f"DEBUG: Retrieved {len(results)} results for query: '{query}'")
    return results

def filter_by_sdg_tag(tag: str) -> List[Dict]:
    """Filter cached results based on SDG tags."""
    print(f"DEBUG: Filtering results by SDG tag: '{tag}'...")
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        metadata = load_cache(document_metadata_path)
        if metadata is None or 'documents' not in metadata:
            print("DEBUG: Document metadata not found or corrupted.")
            return []

        # Extract documents metadata and handle different data structures
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):  # If it's a numpy array, convert to list
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported documents structure type: {type(documents)}")

        # If the tag is simply 'sdg', include all documents related to any SDGs
        if tag.lower() == "sdg":
            filtered_results = [
                doc for doc in doc_dict.values() 
                if isinstance(doc, dict) and 
                any(sdgt in doc.get('sdg_tags', []) for sdgt in sdg_keywords.keys())
            ]
        else:
            # Filter documents based on the provided SDG tag
            filtered_results = [
                doc for doc in doc_dict.values() 
                if isinstance(doc, dict) and tag in doc.get('sdg_tags', [])
            ]
        
        print(f"DEBUG: Found {len(filtered_results)} results for SDG tag: '{tag}'")
        return filtered_results[:10]  # Limit results to 10

    except Exception as e:
        print(f"ERROR: Failed to filter by SDG tag '{tag}': {e}")
        return []

def presenter_search(query: str) -> List[Dict]:
    """Perform a search for TEDx talks where the presenter's name matches the query."""
    print(f"DEBUG: Performing presenter search for: '{query}'...")
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')

    try:
        metadata = load_cache(document_metadata_path)
        if metadata is None or 'documents' not in metadata:
            print("DEBUG: Document metadata not found or corrupted.")
            return []

        # Extract documents metadata
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):  # If it's a numpy array, convert to list
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported documents structure type: {type(documents)}")

        # Perform exact or partial matching in presenterDisplayName
        filtered_results = [
            doc for doc in doc_dict.values() 
            if isinstance(doc, dict) and query.lower() in doc.get('presenterDisplayName', '').lower()
        ]
        
        print(f"DEBUG: Found {len(filtered_results)} results for presenter search: '{query}'")
        return filtered_results[:10]  # Limit results to 10
    except Exception as e:
        print(f"ERROR: Failed to perform presenter search: {e}")
        return []

def normalize_sdg_query(query: str) -> str:
    """Normalize the query if it matches the SDG pattern: 'sdg', one or more spaces, and a digit."""
    sdg_pattern = r"^sdg\s*\d*$"  # Match 'sdg', followed by optional spaces and an optional digit (e.g., 'sdg', 'sdg 1')
    if re.match(sdg_pattern, query, re.IGNORECASE):
        # Normalize to format: 'sdg' followed directly by the digit (e.g., 'sdg 1' -> 'sdg1')
        return re.sub(r"\s+", "", query.lower())  # Remove spaces and convert to lowercase
    return query

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    """Handle the search endpoint by performing semantic search, SDG tag search, or presenter search."""
    request_uuid = uuid.uuid4()
    search_request_start_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    try:
        # Normalize query if it matches SDG pattern (e.g., 'sdg 1' -> 'sdg1')
        normalized_query = normalize_sdg_query(query)

        if normalized_query.startswith("sdg"):
            # Perform SDG tag-based search
            results = filter_by_sdg_tag(normalized_query)
        elif query.lower().startswith("presenter:") or query.lower().startswith("presenter "):
            # Perform presenter search when query starts with 'presenter: ' or 'presenter '
            presenter_query = query.split(":", 1)[-1].strip()  # Extract the presenter's name
            results = presenter_search(presenter_query)
        else:
            # Perform standard semantic search using the LRU cache
            results = cached_semantic_search(query, top_n=10)

            # Boost presenter matches if query appears to be a name
            presenter_matches = presenter_search(query)
            # Add presenter matches to the results and rank them higher, ensuring no duplicates
            results = presenter_matches + [res for res in results if res not in presenter_matches]

            # Limit the total results to 10
            results = results[:10]

        # Ensure sdg_tags and transcripts are included in the results
        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])  # Add empty sdg_tags if not present
            result['transcript'] = result.get('transcript', '')

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
