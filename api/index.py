# File: api/index.py

import time
import uuid
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution
from functools import lru_cache  # For caching semantic search results
from backend.fastapi.services.semantic_search import semantic_search  # Correct import
from backend.fastapi.cache.cache_manager_read import load_cache
from backend.fastapi.data.sdg_keywords import sdg_keywords  # SDG keywords mapping

# Initialize FastAPI app
app = FastAPI(
    docs_url="/api/search/docs",
    redoc_url="/api/search/redoc",
    openapi_url="/api/search/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Resolve cache directory path
base_dir = Path(__file__).resolve().parent.parent
cache_dir = str(base_dir / "backend" / "fastapi" / "cache")

def load_vocabulary(cache_dir: str) -> Dict[str, int]:
    """Load vocabulary from the precomputed TF-IDF cache."""
    tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')
    metadata = load_cache(tfidf_metadata_path)
    if not metadata or 'vocabulary' not in metadata:
        raise RuntimeError("TF-IDF metadata not found or corrupted.")
    return metadata['vocabulary'].item()

# Load vocabulary on startup
vocabulary = load_vocabulary(cache_dir)

@lru_cache(maxsize=1000)
def cached_semantic_search(query: str, top_n: int = 100) -> List[Dict]:
    """Cached semantic search."""
    print(f"DEBUG: Using LRU cache for query: '{query}'")
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 100) -> List[Dict]:
    """Perform semantic search."""
    print(f"DEBUG: Performing semantic search for query: '{query}'...")
    results = semantic_search(query, cache_dir, top_n=top_n)
    if not results:
        print(f"DEBUG: No results returned for query: '{query}'.")
        return []
    print(f"DEBUG: Retrieved {len(results)} results for query: '{query}'")
    return results

def filter_out_null_transcripts(results: List[Dict]) -> List[Dict]:
    """Filter out results with null or empty transcripts."""
    return [result for result in results if result.get('transcript')]

def filter_by_sdg_tag_from_cache(tag: str) -> List[Dict]:
    """Filter results based on SDG tags from cache."""
    print(f"DEBUG: Filtering results by SDG tag: '{tag}'...")
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        metadata = load_cache(document_metadata_path)
        if not metadata or 'documents' not in metadata:
            print("DEBUG: Document metadata not found or corrupted.")
            return []
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported document structure type: {type(documents)}")

        if tag.lower() == "sdg":
            filtered_results = [
                doc for doc in doc_dict.values()
                if any(sdgt in doc.get('sdg_tags', []) for sdgt in sdg_keywords.keys())
            ]
        else:
            filtered_results = [
                doc for doc in doc_dict.values()
                if tag in doc.get('sdg_tags', [])
            ]
        print(f"DEBUG: Found {len(filtered_results)} results for SDG tag: '{tag}'")
        return filtered_results[:100]
    except Exception as e:
        print(f"ERROR: Failed to filter by SDG tag '{tag}': {e}")
        return []

def normalize_sdg_query(query: str) -> str:
    """Normalize SDG queries."""
    sdg_pattern = r"^sdg\s*\d*$"
    if re.match(sdg_pattern, query, re.IGNORECASE):
        return re.sub(r"\s+", "", query.lower())
    return query

def extract_sdg_number(query: str) -> Optional[int]:
    """Extract SDG number from queries."""
    match = re.match(r"sdg\s*(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def filter_by_presenter(presenter_name: str) -> List[Dict]:
    """Filter results based on presenter name."""
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        metadata = load_cache(document_metadata_path)
        if not metadata or 'documents' not in metadata:
            return []
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported document structure type: {type(documents)}")

        filtered_results = [
            doc for doc in doc_dict.values()
            if presenter_name.lower() in doc.get('presenterDisplayName', '').lower()
        ]
        return filtered_results[:100]
    except Exception as e:
        print(f"ERROR: Failed to filter by presenter name '{presenter_name}': {e}")
        return []

def rank_and_combine_results(presenter_results: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    """Combine and rank results, removing duplicates."""
    semantic_results = [result for result in semantic_results if result not in presenter_results]
    combined_results = presenter_results + semantic_results
    return combined_results

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    """Handle search requests."""
    request_uuid = uuid.uuid4()
    search_start_time = time.time()
    print(f"{request_uuid} [Search] Starting search for query: '{query}'...")

    try:
        # Normalize query
        normalized_query = normalize_sdg_query(query)
        is_sdg_query = normalized_query.startswith("sdg")

        if is_sdg_query:
            # Handle SDG tag-based search
            sdg_number = extract_sdg_number(normalized_query)
            sdg_tag = f"sdg{sdg_number}" if sdg_number else "sdg"
            print(f"DEBUG: SDG tag detected: '{sdg_tag}'")

            # Retrieve SDG-based results
            results = filter_by_sdg_tag_from_cache(sdg_tag)
            results = filter_out_null_transcripts(results)
            results = results[:10]
        else:
            # Run presenter and semantic searches in parallel
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(filter_by_presenter, query): "presenter",
                    executor.submit(cached_semantic_search, query): "semantic"
                }

                presenter_results = []
                semantic_results = []

                for future in as_completed(futures):
                    tag = futures[future]
                    if tag == "presenter":
                        presenter_results = future.result()
                    elif tag == "semantic":
                        semantic_results = future.result()

            # Filter out null transcripts
            presenter_results = filter_out_null_transcripts(presenter_results)
            semantic_results = filter_out_null_transcripts(semantic_results)

            # Combine and rank results
            results = rank_and_combine_results(presenter_results, semantic_results)
            results = results[:10]

        # Ensure 'sdg_tags' and 'transcript' keys exist
        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])
            result['transcript'] = result.get('transcript', '')

    except RuntimeError as e:
        print(f"{request_uuid} [Cache Error] {e}")
        raise HTTPException(status_code=503, detail="Cache initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Search Error] {e}")
        raise HTTPException(status_code=500, detail="Search failed.")

    search_end_time = time.time()
    print(f"{request_uuid} [Search] Completed in {search_end_time - search_start_time:.4f} seconds.")

    return {"results": results}
