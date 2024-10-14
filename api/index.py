# File: api/index.py

import time
import uuid
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from backend.fastapi.services.semantic_search import semantic_search
from backend.fastapi.cache.cache_manager_read import load_cache
from backend.fastapi.data.sdg_keywords import sdg_keywords  # Ensure this includes 'sdg1' to 'sdg17'

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
def cached_semantic_search(query: str, top_n: int = 100) -> List[Dict[str, Any]]:
    """Cached semantic search."""
    print(f"DEBUG: Using LRU cache for query: '{query}'")
    return perform_semantic_search(query, top_n)

def perform_semantic_search(query: str, top_n: int = 100) -> List[Dict[str, Any]]:
    """Perform semantic search."""
    print(f"DEBUG: Performing semantic search for query: '{query}'...")
    results = semantic_search(query, cache_dir, top_n=top_n)
    if not results:
        print(f"DEBUG: No results returned for query: '{query}'.")
        return []
    print(f"DEBUG: Retrieved {len(results)} results for query: '{query}'")
    return results

def filter_out_null_transcripts(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out results with null or empty transcripts."""
    filtered_results = []
    for result in results:
        transcript = result.get('document', {}).get('transcript')
        if transcript:
            filtered_results.append(result)
    return filtered_results

def filter_out_zero_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out results with a score of 0.0."""
    filtered_results = []
    for result in results:
        score = result.get('score', 0.0)
        if score != 0.0:
            filtered_results.append(result)
    return filtered_results

def filter_by_sdg_tag_from_cache(tag: str) -> List[Dict[str, Any]]:
    """Filter results based on SDG tags from cache."""
    print(f"DEBUG: Filtering results by SDG tag: '{tag}'...")
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        # Load documents from cache
        metadata = load_cache(document_metadata_path)
        if not metadata or 'documents' not in metadata:
            print("DEBUG: Document metadata not found or corrupted.")
            return []
        documents = metadata['documents']

        # Convert documents to a consistent dictionary format
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported document structure type: {type(documents)}")

        # Normalize the tag
        tag_normalized = tag.lower().strip()

        # Prepare valid SDG tags set
        valid_sdg_tags = set(sdg_keywords.keys())

        # Filter documents based on SDG tags
        filtered_results = []
        for doc in doc_dict.values():
            sdg_tags = doc.get('sdg_tags', [])
            # Normalize sdg_tags
            sdg_tags_normalized = [str(sdg_tag).lower().strip() for sdg_tag in sdg_tags]

            # Debugging: Print the SDG tags of the document
            print(f"DEBUG: Document SDG Tags: {sdg_tags_normalized}")

            if tag_normalized == "sdg":
                # Include all documents related to any SDG
                if any(sdg_tag in valid_sdg_tags for sdg_tag in sdg_tags_normalized):
                    filtered_results.append({'document': doc})
            else:
                # Ensure exact match of SDG tag
                if tag_normalized in sdg_tags_normalized:
                    filtered_results.append({'document': doc})

        print(f"DEBUG: Found {len(filtered_results)} results for SDG tag: '{tag}'")
        return filtered_results[:100]
    except Exception as e:
        print(f"ERROR: Failed to filter by SDG tag '{tag}': {e}")
        return []

def normalize_sdg_query(query: str) -> str:
    """Normalize SDG queries."""
    sdg_pattern = r"^sdg\s*\d+.*"  # Match 'sdg' followed by digits and any text
    if re.match(sdg_pattern, query, re.IGNORECASE):
        return re.sub(r"\s+", "", query.lower())
    return query

def extract_sdg_number(query: str) -> Optional[int]:
    """Extract SDG number from queries."""
    match = re.search(r"sdg\s*(\d{1,2})", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def filter_by_presenter(presenter_name: str) -> List[Dict[str, Any]]:
    """Filter results based on presenter name."""
    document_metadata_path = os.path.join(cache_dir, 'document_metadata.npz')
    try:
        # Load documents from cache
        metadata = load_cache(document_metadata_path)
        if not metadata or 'documents' not in metadata:
            return []
        documents = metadata['documents']

        # Convert documents to a consistent dictionary format
        if isinstance(documents, dict):
            doc_dict = documents
        elif hasattr(documents, 'tolist'):
            doc_list = documents.tolist()
            doc_dict = {i: doc for i, doc in enumerate(doc_list)}
        else:
            raise TypeError(f"Unsupported document structure type: {type(documents)}")

        # Filter documents based on presenter name
        filtered_results = []
        for doc in doc_dict.values():
            presenter_display_name = doc.get('presenterDisplayName', '')
            if presenter_name.lower() in presenter_display_name.lower():
                # Wrap the document in a consistent result format
                filtered_results.append({'document': doc})

        return filtered_results[:100]
    except Exception as e:
        print(f"ERROR: Failed to filter by presenter name '{presenter_name}': {e}")
        return []

def rank_and_combine_results(*args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine and rank results, removing duplicates."""
    seen_docs = set()
    combined_results = []

    for results in args:
        for result in results:
            doc = result.get('document', {})
            doc_id = doc.get('document_id') or doc.get('slug')
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                combined_results.append(result)

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
        sdg_number = extract_sdg_number(query)
        is_sdg_query = normalized_query.lower().startswith("sdg")

        if is_sdg_query and sdg_number:
            # Handle SDG tag-based search
            sdg_tag = f"sdg{sdg_number}"
            print(f"DEBUG: SDG tag detected: '{sdg_tag}'")

            # Retrieve SDG-based results
            sdg_results = filter_by_sdg_tag_from_cache(sdg_tag)
            sdg_results = filter_out_null_transcripts(sdg_results)

            # Also perform semantic search with augmented query
            sdg_keywords_list = sdg_keywords.get(sdg_tag.lower(), [])
            augmented_query = f"{query} {' '.join(sdg_keywords_list)}"

            # Run semantic search
            semantic_results = cached_semantic_search(augmented_query)
            # Filter out results with null transcripts and zero scores
            semantic_results = filter_out_null_transcripts(semantic_results)
            semantic_results = filter_out_zero_scores(semantic_results)

            # Combine SDG and semantic results
            results = rank_and_combine_results(semantic_results, sdg_results)
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

            # Filter out results with null transcripts and zero scores
            presenter_results = filter_out_null_transcripts(presenter_results)
            semantic_results = filter_out_null_transcripts(semantic_results)
            semantic_results = filter_out_zero_scores(semantic_results)

            # Combine and rank results
            results = rank_and_combine_results(presenter_results, semantic_results)
            results = results[:10]

        # Ensure 'sdg_tags' and 'transcript' keys exist
        for result in results:
            doc = result.get('document', {})
            doc['sdg_tags'] = doc.get('sdg_tags', [])
            doc['transcript'] = doc.get('transcript', '')
            result['document'] = doc  # Update the document

    except RuntimeError as e:
        print(f"{request_uuid} [Cache Error] {e}")
        raise HTTPException(status_code=503, detail="Cache initialization failed.")
    except Exception as e:
        print(f"{request_uuid} [Search Error] {e}")
        raise HTTPException(status_code=500, detail="Search failed.")

    search_end_time = time.time()
    print(f"{request_uuid} [Search] Completed in {search_end_time - search_start_time:.4f} seconds.")

    return {"results": results}
