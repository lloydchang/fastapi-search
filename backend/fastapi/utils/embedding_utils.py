# File: backend/fastapi/utils/embedding_utils.py

from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_matrix(documents):
    """
    Computes the TF-IDF matrix for the provided documents.

    Args:
        documents (List[str]): List of documents as strings.

    Returns:
        TfidfVectorizer: The fitted TF-IDF vectorizer.
        csr_matrix: The TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix
