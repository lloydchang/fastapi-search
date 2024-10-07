# File: backend/fastapi/utils/text_processing.py

import math
import re
from collections import Counter, defaultdict

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def compute_tf(tokens):
    tf_dict = Counter(tokens)
    total_terms = len(tokens)
    for term in tf_dict:
        tf_dict[term] /= total_terms
    return tf_dict

def compute_idf(documents):
    N = len(documents)
    idf_dict = {}
    all_tokens_set = set(token for doc in documents for token in doc)
    for term in all_tokens_set:
        containing_docs = sum(1 for doc in documents if term in doc)
        idf_dict[term] = math.log(N / (1 + containing_docs))
    return idf_dict

def compute_tfidf(tf_dict, idf_dict):
    tfidf = {}
    for term, tf_value in tf_dict.items():
        tfidf[term] = tf_value * idf_dict.get(term, 0.0)
    return tfidf

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0.0) * vec2.get(term, 0.0) for term in set(vec1.keys()).union(vec2.keys()))
    magnitude1 = math.sqrt(sum(value**2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value**2 for value in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)
