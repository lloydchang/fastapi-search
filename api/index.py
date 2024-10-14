# File: api/index.py

import time
import uuid
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from functools import lru_cache
from backend.fastapi.services.semantic_search import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache
from backend.fastapi.data.sdg_keywords import sdg_keywords

# Initialize FastAPI app
app = FastAPI(docs_url="/api/search/docs", redoc_url="/api/search/redoc", openapi_url="/api/search/openapi.json")

# Configure CORS
origins = ["*"]
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

@lru_cache(maxsize=1000)
def cached_semantic_search(query: str, top_n: int = 100) -> List[Dict]:
    """Cached wrapper for performing a semantic search."""
    print(f"DEBUG: Using LRU cache for query: '{query}'")
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 100) -> List[Dict]:
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
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported documents structure type: {type(documents)}")
        filtered_results = []
        for doc in doc_dict.values():
            sdg_tags = doc.get('sdg_tags', [])
            print(f"DEBUG: Document '{doc.get('slug', 'unknown')}' SDG tags: {sdg_tags}")
            if isinstance(doc, dict) and tag in sdg_tags:
                filtered_results.append(doc)
        print(f"DEBUG: Found {len(filtered_results)} results for SDG tag: '{tag}'")
        return filtered_results[:100]
    except Exception as e:
        print(f"ERROR: Failed to filter by SDG tag '{tag}': {e}")
        return []

def normalize_sdg_query(query: str) -> str:
    """Normalize the query if it matches the SDG pattern: 'sdg', one or more spaces, and a digit."""
    sdg_pattern = r"^sdg\s*\d{1,2}$"
    if re.match(sdg_pattern, query, re.IGNORECASE):
        return re.sub(r"\s+", "", query.lower())
    return query

def extract_sdg_number(query: str) -> Optional[int]:
    """Extract SDG number from queries like "SDG 7" or "SDG 7: Affordable and Clean Energy"."""
    match = re.search(r"SDG\s*(\d{1,2})", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def rank_and_combine_results(semantic_results: List[Dict], tag_results: List[Dict], sdg_number: int) -> List[Dict]:
    """Combine and rank results, prioritizing those with the correct SDG tag."""
    combined_results = semantic_results + tag_results
    def rank_key(result):
        has_matching_tag = f"sdg{sdg_number}" in result.get('sdg_tags', [])
        return (has_matching_tag, result.get('score', 0))
    return sorted(combined_results, key=rank_key, reverse=True)

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    """Handle the search endpoint by performing either a semantic search or an SDG tag search."""
    request_uuid = uuid.uuid4()
    search_request_start_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request processing for query: '{query}'...")

    try:
        normalized_query = normalize_sdg_query(query)
        sdg_number = extract_sdg_number(query)

        if sdg_number:
            # Perform both semantic and tag-based search
            semantic_results = cached_semantic_search(query, top_n=100)
            tag_results = filter_by_sdg_tag(f"sdg{sdg_number}")
            results = rank_and_combine_results(semantic_results, tag_results, sdg_number)
        elif normalized_query.startswith("sdg"):
            # Perform SDG tag-based search for any SDG from 'sdg1' to 'sdg17'
            results = filter_by_sdg_tag(normalized_query)
        else:
            # Perform standard semantic search using the LRU cache
            results = cached_semantic_search(query, top_n=100)

        # Ensure sdg_tags and transcripts are included in the results
        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])
            result['transcript'] = result.get('transcript', '')

        # Limit to top 10 results
        results = results[:10]

    except RuntimeError as e:
        print(f"{request_uuid} [Cache Error] {e}")
        raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Semantic Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Semantic search failed.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    # Return the filtered results
    return {"results": results}
