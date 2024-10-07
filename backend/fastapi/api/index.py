# File: backend/fastapi/api/index.py

import subprocess
import sys

# Install required packages if not present (in the specified order)
required_packages = ["sentence_transformers", "torch"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found, installing at runtime...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

# Continue with the rest of your imports and application code
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import importlib
import os
import pickle
import warnings
import asyncio
import numpy as np
import torch  # Import torch to work with Tensors
from sentence_transformers import SentenceTransformer  # Import Sentence-BERT model after installing
from backend.fastapi.utils.logger import logger
from backend.fastapi.data.data_loader import load_dataset
from backend.fastapi.models.model_definitions import load_model
from backend.fastapi.utils.embedding_utils import encode_descriptions, encode_sdg_keywords
from backend.fastapi.services.sdg_manager import get_sdg_keywords
from backend.fastapi.data.sdg_utils import compute_sdg_tags

# Create a FastAPI app instance
app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

# Enable CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the loaded resources
model = None
data = None
sdg_embeddings = None
resources_initialized = False  # Flag to track if resources are fully initialized

# Event to wait for resource initialization
resource_event = asyncio.Event()

# Suppress warnings for specific libraries
logger.info("Suppressing FutureWarnings for transformers and torch libraries.")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*", module="torch.storage")

# File paths for data and cache
file_path = "backend/fastapi/data/github-mauropelucchi-tedx_dataset-update_2024-details.csv"
cache_file_path = "backend/fastapi/cache/tedx_dataset.pkl"
sdg_embeddings_cache = "backend/fastapi/cache/sdg_embeddings.pkl"
sdg_tags_cache = "backend/fastapi/cache/sdg_tags.pkl"
description_embeddings_cache = "backend/fastapi/cache/description_embeddings.pkl"

# Background task to load the necessary resources
async def load_resources():
    global model, data, sdg_embeddings, resources_initialized

    # Load TEDx Dataset
    logger.info("Loading TEDx dataset.")
    data = load_dataset(file_path, cache_file_path)
    logger.info(f"TEDx dataset loaded successfully! Data: {data is not None}")

    # Check if 'sdg_tags' column is in the dataset and add if missing
    if 'sdg_tags' not in data.columns:
        logger.info("Adding missing 'sdg_tags' column to the dataset with default empty lists.")
        data['sdg_tags'] = [[] for _ in range(len(data))]

    # Load the Sentence-BERT model
    logger.info("Loading the Sentence-BERT model for semantic search.")
    model = load_model('paraphrase-MiniLM-L6-v2')
    logger.info(f"Sentence-BERT model loaded successfully! Model: {model is not None}")

    # Check if 'description_vector' is present, if not, compute and add it
    if 'description_vector' not in data.columns:
        logger.info("'description_vector' column missing. Computing description embeddings.")
        description_vectors = await asyncio.to_thread(encode_descriptions, data['description'].tolist(), model)
        
        # Assign the computed vectors to the 'description_vector' column
        data['description_vector'] = description_vectors

        # Save the updated dataset with description embeddings to cache
        with open(cache_file_path, 'wb') as cache_file:
            pickle.dump(data, cache_file)
        logger.info("Description vectors computed and added to the dataset. Data cached successfully.")

    # Load or compute SDG Embeddings
    logger.info("Loading or computing SDG embeddings.")
    if os.path.exists(sdg_embeddings_cache):
        logger.info("Loading cached SDG keyword embeddings.")
        try:
            with open(sdg_embeddings_cache, 'rb') as cache_file:
                sdg_embeddings = pickle.load(cache_file)
            logger.info(f"SDG embeddings loaded from cache. Embeddings: {sdg_embeddings is not None}")
        except Exception as e:
            logger.error(f"Error loading cached SDG embeddings: {e}")
            sdg_embeddings = None
    else:
        logger.info("Computing SDG embeddings.")
        sdg_keywords = get_sdg_keywords()
        sdg_keyword_list = [keywords for keywords in sdg_keywords.keys()]  # Only take the keys (sdg1, sdg2, etc.)
        sdg_embeddings = await asyncio.to_thread(encode_sdg_keywords, sdg_keyword_list, model)
        if sdg_embeddings:
            with open(sdg_embeddings_cache, 'wb') as cache_file:
                pickle.dump(sdg_embeddings, cache_file)
            logger.info("SDG embeddings computed and cached successfully.")
        else:
            logger.error("Failed to encode SDG keywords.")
            sdg_embeddings = None

    # Compute or load SDG Tags
    logger.info("Loading or computing SDG tags.")
    if os.path.exists(sdg_tags_cache):
        logger.info("Loading cached SDG tags.")
        try:
            with open(sdg_tags_cache, 'rb') as cache_file:
                data['sdg_tags'] = pickle.load(cache_file)
            logger.info("SDG tags loaded from cache.")
        except Exception as e:
            logger.error(f"Error loading cached SDG tags: {e}")
    else:
        logger.info("Computing SDG tags.")
        if not data.empty and 'description_vector' in data.columns and sdg_embeddings is not None:
            description_vectors_tensor = torch.tensor(np.array(data['description_vector'].tolist()))  # Ensure this is a Tensor
            sdg_embeddings_tensor = torch.tensor(np.array(sdg_embeddings))  # Ensure this is a Tensor
            cosine_similarities = torch.nn.functional.cosine_similarity(description_vectors_tensor.unsqueeze(1), sdg_embeddings_tensor.unsqueeze(0), dim=-1)

            # Get SDG names to pass as a parameter
            sdg_keywords = get_sdg_keywords()
            sdg_names = list(sdg_keywords.keys())

            # Call compute_sdg_tags with cosine similarities and sdg_names
            data['sdg_tags'] = compute_sdg_tags(cosine_similarities, sdg_names)

            with open(sdg_tags_cache, 'wb') as cache_file:
                pickle.dump(data['sdg_tags'], cache_file)
            logger.info("SDG tags computed and cached successfully.")

    # Set the resources initialized flag and notify waiting coroutines
    resources_initialized = True
    resource_event.set()  # Signal that resources are ready
    logger.info("All resources are fully loaded and ready for use.")

# On startup, load resources in a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources())

# Create a Search Endpoint for TEDx Talks
@app.get("/api/search")
async def search(query: str = Query(..., min_length=1)) -> List[Dict]:
    await resource_event.wait()  # Wait until resources are fully initialized
    logger.info(f"Search request received: Model is None: {model is None}, Data is None: {data is None}")

    if model is None or data is None:
        logger.error("Model or data not available.")
        return [{"error": "Model or data not available."}]

    logger.info(f"Performing semantic search for the query: '{query}'.")
    try:
        search_module = importlib.import_module("backend.fastapi.services.search_service")
        result = await search_module.semantic_search(query, data, model, sdg_embeddings)
        return result
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return [{"error": str(e)}]
