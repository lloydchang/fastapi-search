# File: backend/fastapi/utils/precompute_cache.py

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, List, Dict
from backend.fastapi.data.sdg_keywords import sdg_keywords  # Import SDG keywords
import logging
import requests  # For fetching transcripts
import re  # For regex extraction of transcripts from HTML

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TRANSCRIPT_CSV_PATH = 'precompute-data/transcripts.csv'  # Path to save and load transcripts
TRANSCRIPT_CSV_PATH = 'precompute-data/transcripts.csv'  # Path to save and load transcripts

def load_tedx_documents(csv_file_path: str) -> List[Dict[str, str]]:
    """Load TEDx talks from the provided CSV file and extract metadata and text content."""
    tedx_df = pd.read_csv(csv_file_path)
    transcripts_df = pd.read_csv(TRANSCRIPT_CSV_PATH) if os.path.exists(TRANSCRIPT_CSV_PATH) else pd.DataFrame(columns=['slug', 'transcript'])

    if 'description' not in tedx_df.columns or 'slug' not in tedx_df.columns:
        raise ValueError(f"Required columns 'description' or 'slug' not found in the CSV file {csv_file_path}")
    
    # Merge transcripts if already downloaded
    tedx_df = pd.merge(tedx_df, transcripts_df[['slug', 'transcript']], on='slug', how='left')

    # Fetch missing transcripts for individual talks
    tedx_df['transcript'] = tedx_df.apply(lambda row: fetch_transcript_if_missing(row['slug'], row['transcript']), axis=1)

    documents = tedx_df[['slug', 'description', 'presenterDisplayName', 'transcript']].dropna().to_dict('records')
    logger.info(f"Loaded {len(documents)} TEDx documents from the CSV file, with transcripts.")
    return documents, tedx_df[['slug', 'transcript']]

def fetch_transcript_if_missing(slug: str, existing_transcript: str) -> str:
    """Check if the transcript already exists, if not, download it."""
    if pd.notna(existing_transcript) and existing_transcript != 'Transcript not available':
        logger.info(f"Transcript already exists for {slug}, skipping download.")
        return existing_transcript

    logger.info(f"Transcript not found for {slug}, downloading...")
    transcript = fetch_transcript(slug)
    
    # Save transcript immediately after fetching
    save_transcript(slug, transcript)
    
    return transcript

def fetch_transcript(slug: str) -> str:
    """Fetch and extract transcript from TED by slug."""
    transcript_url = f"https://www.ted.com/talks/{slug}/transcript?subtitle=en"
    try:
        response = requests.get(transcript_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
        })

        if response.status_code == 200:
            return extract_transcript(response.text)
        else:
            logger.error(f"Failed to fetch transcript for {slug}. Status: {response.status_code}")
            return 'Transcript not available'
    except requests.RequestException as e:
        logger.error(f"Error fetching transcript for {slug}: {e}")
        return 'Transcript not available'

def extract_transcript(html: str) -> str:
    """Extract the transcript text from HTML using regex, adapted from JS logic."""
    logger.info("Starting transcript extraction from HTML...")

    # Match the JSON-like structure that may contain the transcript
    transcript_match = re.search(r'"transcript":\s*"(.*?)",', html)

    if not transcript_match:
        logger.error("Could not find transcript in the HTML. The transcript structure may have changed.")
        logger.debug(f"HTML Snippet for Debugging: {html[:500]}")  # Log the first 500 characters for debugging
        return 'Transcript not available'

    # Extract and clean the transcript, handling special characters
    raw_transcript = transcript_match.group(1)
    decoded_transcript = (
        raw_transcript.replace("\\u0026", "&")
                      .replace("\\u003c", "<")
                      .replace("\\u003e", ">")
                      .replace("\\u0027", "'")
                      .replace("&quot;", '"')
                      .replace("\\n", " ")
    )

    logger.info("Transcript successfully extracted and decoded.")
    return decoded_transcript.strip()

def save_transcript(slug: str, transcript: str):
    """Append the newly fetched transcript to the transcripts.csv file, ensuring no duplicates."""
    if os.path.exists(TRANSCRIPT_CSV_PATH):
        # Load existing transcripts
        transcripts_df = pd.read_csv(TRANSCRIPT_CSV_PATH)
        
        # Check if the slug already exists
        if slug in transcripts_df['slug'].values:
            logger.info(f"Transcript for {slug} already exists in {TRANSCRIPT_CSV_PATH}, skipping save.")
            return
    
    # Create a new dataframe with the current transcript
    transcript_df = pd.DataFrame([[slug, transcript]], columns=['slug', 'transcript'])
    
    # If the file doesn't exist, create it; otherwise, append to it
    transcript_df.to_csv(TRANSCRIPT_CSV_PATH, mode='a', header=not os.path.exists(TRANSCRIPT_CSV_PATH), index=False)
    logger.info(f"Transcript for {slug} saved to {TRANSCRIPT_CSV_PATH}")

def create_tfidf_matrix(documents: List[Dict[str, str]]) -> Any:
    """Create a sparse TF-IDF matrix from the combined 'description', 'slug', and 'transcript' fields."""
    # Combine slug, description, and transcript fields to improve semantic search capabilities.
    combined_texts = [f"{doc['slug']} {doc['description']} {doc['transcript']}" for doc in documents]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    logger.info(f"TF-IDF matrix created. Shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

def get_sdg_tags_for_documents(documents: List[Dict[str, str]], sdg_keywords: Dict[str, List[str]]) -> None:
    """Assign SDG tags to documents based on semantic similarity to SDG keywords."""
    # Flatten SDG keywords for vectorization
    sdg_keyword_list = [keyword for keywords in sdg_keywords.values() for keyword in keywords]
    
    # Create TF-IDF vectorizer and transform SDG keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    sdg_tfidf_matrix = vectorizer.fit_transform(sdg_keyword_list)

    for doc in documents:
        # Combine the description and transcript for SDG tagging
        combined_text = f"{doc['description']} {doc['transcript']}"
        combined_vector = vectorizer.transform([combined_text])

        # Calculate cosine similarity with SDG keywords
        cosine_similarities = cosine_similarity(combined_vector, sdg_tfidf_matrix).flatten()
        cosine_similarities = cosine_similarity(combined_vector, sdg_tfidf_matrix).flatten()
        
        # Assign SDG tags based on high similarity
        matched_tags = []
        for i in np.argsort(cosine_similarities)[::-1]:  # Sort indices in descending order
            if cosine_similarities[i] > 0.1:  # Threshold can be adjusted
                # Identify which SDG tag this keyword belongs to
                for sdg, keywords in sdg_keywords.items():
                    if i < len(keywords):
                        matched_tags.append(sdg)
                        break
            else:
                break  # Stop if the similarity is below the threshold

        # If no tags matched, find the closest one
        if not matched_tags:
            closest_index = np.argmax(cosine_similarities)  # Find index of the highest similarity
            closest_sdg = list(sdg_keywords.keys())[closest_index // len(list(sdg_keywords.values())[0])]  # Determine SDG tag
            matched_tags.append(closest_sdg)  # Assign the closest SDG tag

        # Deduplicate tags before adding them to the document
        doc['sdg_tags'] = list(set(matched_tags))  # Remove duplicates
        logger.info(f"Document '{doc['slug']}' assigned SDG tags: {doc['sdg_tags']}")  # Log the assigned tags

def save_sparse_matrix(tfidf_matrix, cache_dir: str):
    """Save sparse matrix in a numpy-compatible format."""
    tfidf_data_path = os.path.join(cache_dir, "tfidf_matrix.npz")
    np.savez_compressed(
        tfidf_data_path,
        data=tfidf_matrix.data,
        indices=tfidf_matrix.indices,
        indptr=tfidf_matrix.indptr,
        shape=tfidf_matrix.shape
    )
    logger.info(f"Sparse TF-IDF matrix components saved to {tfidf_data_path}")
    # Debug: Confirm saved keys
    with np.load(tfidf_data_path) as data:
        logger.info(f"Saved sparse matrix keys: {data.files}")

def save_tfidf_components(tfidf_matrix, vectorizer: TfidfVectorizer, documents: List[Dict[str, str]], cache_dir: str):
    """Save the TF-IDF matrix and metadata in a format compatible with numpy."""
    tfidf_metadata_path = os.path.join(cache_dir, "tfidf_metadata.npz")
    document_metadata_path = os.path.join(cache_dir, "document_metadata.npz")

    # Save sparse matrix components
    save_sparse_matrix(tfidf_matrix, cache_dir)

    # Save metadata (vocabulary and IDF values)
    np.savez_compressed(
        tfidf_metadata_path,
        vocabulary=vectorizer.vocabulary_,
        idf_values=vectorizer.idf_
    )
    logger.info(f"TF-IDF metadata saved to {tfidf_metadata_path}")

    # Save document metadata including SDG tags and transcripts
    np.savez_compressed(document_metadata_path, documents=documents)
    logger.info(f"Document metadata saved to {document_metadata_path}")

def precompute_cache():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(base_dir, "backend", "fastapi", "cache")
    csv_file_path = os.path.join(base_dir, "precompute-data", "tedx_talks.csv")
    csv_file_path = os.path.join(base_dir, "precompute-data", "tedx_talks.csv")

    # Load TEDx documents with transcripts
    documents, transcripts_df = load_tedx_documents(csv_file_path)

    # Create the TF-IDF matrix using 'slug', 'description', and 'transcript'
    tfidf_matrix, vectorizer = create_tfidf_matrix(documents)

    # Get SDG tags for each document based on semantic matching
    get_sdg_tags_for_documents(documents, sdg_keywords)

    # Save the generated components
    os.makedirs(cache_dir, exist_ok=True)
    save_tfidf_components(tfidf_matrix, vectorizer, documents, cache_dir)
    
    logger.info("Precompute cache written successfully to the cache directory and transcripts updated.")

if __name__ == "__main__":
    precompute_cache()
