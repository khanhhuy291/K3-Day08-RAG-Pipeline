"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import re
from rank_bm25 import BM25Okapi

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản loại bỏ dấu câu."""
    return re.findall(r'\w+', text.lower())


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def _ensure_bm25_index():
    global CORPUS, BM25_INDEX
    if not CORPUS or BM25_INDEX is None:
        from .task4_chunking_indexing import load_documents, chunk_documents
        docs = load_documents()
        CORPUS = chunk_documents(docs)
        if CORPUS:
            BM25_INDEX = build_bm25_index(CORPUS)
    return CORPUS, BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus, bm25 = _ensure_bm25_index()
    if not corpus or bm25 is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": round(score, 4),
                "metadata": corpus[idx]["metadata"]
            })
        if len(results) >= top_k:
            break

    # If query keywords have no exact matches, fallback to returning top chunks with score 0.0 or empty
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
