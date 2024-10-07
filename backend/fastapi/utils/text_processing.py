# File: backend/fastapi/utils/text_processing.py

import math
import re
from collections import Counter
from typing import List, Dict
import numpy as np

def preprocess(text: str) -> List[str]:
    """
    Preprocesses the input text by lowercasing, removing non-alphanumeric characters,
    and tokenizing into words.

    Args:
        text (str): The input text.

    Returns:
        List[str]: List of tokens.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Computes term frequency for each token in the document.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        Dict[str, float]: Term frequency dictionary.
    """
    tf_dict = Counter(tokens)
    total_terms = len(tokens)
    for term in tf_dict:
        tf_dict[term] /= total_terms
    return tf_dict

def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """
    Computes inverse document frequency for each term in the corpus.

    Args:
        documents (List[List[str]]): List of tokenized documents.

    Returns:
        Dict[str, float]: Inverse document frequency dictionary.
    """
    N = len(documents)
    idf_dict = {}
    all_terms = set(term for doc in documents for term in doc)
    for term in all_terms:
        containing_docs = sum(1 for doc in documents if term in doc)
        idf_dict[term] = math.log(N / (1 + containing_docs))
    return idf_dict

def compute_tfidf(tf_dict: Dict[str, float], idf_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Computes TF-IDF for each term in the document.

    Args:
        tf_dict (Dict[str, float]): Term frequency dictionary.
        idf_dict (Dict[str, float]): Inverse document frequency dictionary.

    Returns:
        Dict[str, float]: TF-IDF dictionary.
    """
    tfidf = {}
    for term, tf_value in tf_dict.items():
        tfidf[term] = tf_value * idf_dict.get(term, 0.0)
    return tfidf

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)
