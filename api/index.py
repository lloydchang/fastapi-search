# File: api/index.py

import time
import uuid
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from functools import lru_cache
from backend.fastapi.services.semantic_search import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache

app = FastAPI(docs_url="/api/search/docs", redoc_url="/api/search/redoc", openapi_url="/api/search/openapi.json")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

base_dir = Path(__file__).resolve().parent.parent
cache_dir = str(base_dir / "backend" / "fastapi" / "cache")

def load_vocabulary(cache_dir: str) -> Dict[str, int]:
    tfidf_metadata_path = os.path.join(cache_dir, 'tfidf_metadata.npz')
    metadata = load_cache(tfidf_metadata_path)
    if metadata is None or 'vocabulary' not in metadata:
        raise RuntimeError("TF-IDF metadata not found or corrupted.")
    return metadata['vocabulary'].item()

vocabulary = load_vocabulary(cache_dir)

@lru_cache(maxsize=1000)
def cached_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 10) -> List[Dict]:
    results = semantic_search(query, cache_dir, top_n=top_n)
    if results is None:
        return []
    return results

def filter_by_sdg_tag(tag: str) -> List[Dict]:
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
        filtered_results = [doc for doc in doc_dict.values() if tag in doc.get('sdg_tags', [])]
        return filtered_results[:10]
    except Exception as e:
        print(f"ERROR: Failed to filter by SDG tag '{tag}': {e}")
        return []

def normalize_sdg_query(query: str) -> str:
    sdg_pattern = r"^sdg\s*\d{1,2}$"
    if re.match(sdg_pattern, query, re.IGNORECASE):
        return re.sub(r"\s+", "", query.lower())
    return query

def extract_sdg_number(query: str) -> Optional[int]:
    match = re.search(r"SDG\s*(\d{1,2})", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def rank_and_combine_results(semantic_results: List[Dict], tag_results: List[Dict], sdg_number: int) -> List[Dict]:
    combined_results = semantic_results + tag_results
    def rank_key(result):
        has_matching_tag = f"sdg{sdg_number}" in result.get('sdg_tags', [])
        return (has_matching_tag, result.get('score', 0))
    return sorted(combined_results, key=rank_key, reverse=True)

@app.get("/api/search")
def search(request: Request, query: str = Query(..., min_length=1, max_length=100)) -> Dict:
    request_uuid = uuid.uuid4()
    search_request_start_time = time.time()
    print(f"{request_uuid} [Search Endpoint Handling] Starting search request for query: '{query}'...")

    try:
        normalized_query = normalize_sdg_query(query)
        sdg_number = extract_sdg_number(query)

        if sdg_number:
            semantic_results = cached_semantic_search(query, top_n=100)
            tag_results = filter_by_sdg_tag(f"sdg{sdg_number}")
            results = rank_and_combine_results(semantic_results, tag_results, sdg_number)
        elif normalized_query.startswith("sdg"):
            results = filter_by_sdg_tag(normalized_query)
        else:
            results = cached_semantic_search(query, top_n=10)

        for result in results:
            result['sdg_tags'] = result.get('sdg_tags', [])
            result['transcript'] = result.get('transcript', '')

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
