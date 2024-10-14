# File: api/index.py

import time
import uuid
import os
import re
import asyncio  # For parallel execution
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from backend.fastapi.services.semantic_search import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache
from backend.fastapi.data.sdg_keywords import sdg_keywords  # Import the SDG keywords

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

vocabulary = load_vocabulary(cache_dir)

@lru_cache(maxsize=1000)
def cached_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    """Cached wrapper for performing a semantic search."""
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    """Perform a new semantic search for the given query and return the top `top_n` results."""
    results = semantic_search(query, cache_dir, top_n=top_n)
    if results is None:
        return []
    return results

def filter_out_null_transcripts(results: List[Dict]) -> List[Dict]:
    """Filter out results that have null or empty transcripts."""
    return [result for result in results if result.get('transcript')]

def filter_by_sdg_tag(results: List[Dict], tag: str) -> List[Dict]:
    """Filter cached results based on SDG tags."""
    return [result for result in results if tag in result.get('sdg_tags', [])]

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

def filter_by_presenter(presenter_name: str) -> List[Dict]:
    """Filter results based on presenter names."""
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        metadata = load_cache(document_metadata_path)
        if metadata is None or 'documents' not in metadata:
            return []
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported documents structure type: {type(documents)}")

        filtered_results = [
            doc for doc in doc_dict.values()
            if presenter_name.lower() in doc.get('presenterDisplayName', '').lower()
        ]
        return filtered_results[:10]
    except Exception as e:
        print(f"ERROR: Failed to filter by presenter name '{presenter_name}': {e}")
        return []

def filter_by_sdg_keywords(query: str) -> List[Dict]:
    """Perform SDG tag-based search using keywords mapped from sdg_keywords."""
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        metadata = load_cache(document_metadata_path)
        if metadata is None or 'documents' not in metadata:
            return []
        documents = metadata['documents']
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported documents structure type: {type(documents)}")
        
        sdg_number = extract_sdg_number(query)
        if not sdg_number or f"sdg{sdg_number}" not in sdg_keywords:
            return []
        keywords = sdg_keywords[f"sdg{sdg_number}"]

        # Filter documents based on SDG keywords
        filtered_results = [
            doc for doc in doc_dict.values()
            if any(keyword.lower() in doc.get('transcript', '').lower() for keyword in keywords)
        ]
        return filtered_results[:10]
    except Exception as e:
        print(f"ERROR: Failed to filter by SDG keywords: {e}")
        return []

def rank_and_combine_results(presenter_results: List[Dict], semantic_results: List[Dict], sdg_results: List[Dict]) -> List[Dict]:
    """Combine and rank results, giving preference to presenter results first, followed by SDG and semantic matches."""
    combined_results = presenter_results + [result for result in sdg_results if result not in presenter_results] + [result for result in semantic_results if result not in presenter_results and result not in sdg_results]
    return combined_results

@app.get("/api/search")
async def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    """Handle the search endpoint by performing presenter lookup, semantic search, and SDG keyword-based search."""
    request_uuid = uuid.uuid4()
    search_request_start_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request for query: '{query}'...")

    try:
        # Step 1: Run presenter search, semantic search, and SDG keyword-based search in parallel
        presenter_task = asyncio.create_task(filter_by_presenter(query))
        semantic_task = asyncio.create_task(cached_semantic_search(query, top_n=10))
        sdg_task = asyncio.create_task(filter_by_sdg_keywords(query))

        # Await results of all tasks
        presenter_results, semantic_results, sdg_results = await asyncio.gather(presenter_task, semantic_task, sdg_task)

        # Step 2: Filter out results with null transcripts from all search results
        presenter_results = filter_out_null_transcripts(presenter_results)
        semantic_results = filter_out_null_transcripts(semantic_results)
        sdg_results = filter_out_null_transcripts(sdg_results)

        # Step 3: If query is SDG-related, filter by SDG tag
        sdg_number = extract_sdg_number(query)
        if sdg_number:
            sdg_tag = f"sdg{sdg_number}"
            presenter_results = filter_by_sdg_tag(presenter_results, sdg_tag)
            semantic_results = filter_by_sdg_tag(semantic_results, sdg_tag)

        # Step 4: Combine and rank results (presenter results first, followed by SDG and semantic results)
        results = rank_and_combine_results(presenter_results, semantic_results, sdg_results)

        # Ensure sdg_tags and transcripts are included in the results
        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])
            result['transcript'] = result.get('transcript', '')

        # Limit results to top 10
        results = results[:10]

    except RuntimeError as e:
        print(f"{request_uuid} [Cache Error] {e}")
        raise HTTPException(status_code=503, detail="Precomputed data initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Search Error] Failed to process query '{query}': {e}.")
        raise HTTPException(status_code=500, detail="Search failed.")

    search_request_end_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Search request handled in {search_request_end_time - search_request_start_time:.4f} seconds.")

    return {"results": results}
