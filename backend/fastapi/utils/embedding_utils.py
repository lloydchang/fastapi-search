# File: backend/fastapi/utils/embedding_utils.py

from sklearn.feature_extraction.text import TfidfVectorizer

def encode_descriptions(descriptions):
    """
    Encodes a list of descriptions using TF-IDF vectorizer.

    Args:
        descriptions (List[str]): List of TEDx talk descriptions.

    Returns:
        TfidfVectorizer, sparse matrix: The vectorizer and the TF-IDF matrix.
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        return vectorizer, tfidf_matrix
    except Exception as e:
        return None, None

def encode_sdg_keywords(sdg_keyword_list, vectorizer):
    """
    Encodes a list of SDG keywords using the provided TF-IDF vectorizer.

    Args:
        sdg_keyword_list (List[str]): List of SDG keyword strings.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer trained on the descriptions.

    Returns:
        sparse matrix: The TF-IDF matrix for SDG keywords.
    """
    try:
        sdg_tfidf_matrix = vectorizer.transform(sdg_keyword_list)
        return sdg_tfidf_matrix
    except Exception:
        return None
