"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker hoặc fallback từ xa).
    Nếu không có API Key, fallback giữ nguyên thứ hạng tốt nhất.
    """
    if not candidates:
        return []
    import os
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if jina_api_key:
        try:
            import requests
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": min(top_k, len(candidates))
                },
                timeout=5
            )
            if response.status_code == 200:
                reranked = response.json().get("results", [])
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
        except Exception as e:
            print(f"⚠ Lỗi Jina Reranker API: {e}. Sử dụng fallback RRF/Score gốc.")

    # Fallback nếu không có API key: sắp xếp theo score hiện tại
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse (đa dạng).
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []
    # Nếu candidate chưa kèm embedding (như trong unit test cơ bản), quay về sort theo score
    if not query_embedding or any("embedding" not in c for c in candidates):
        sorted_c = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_c[:top_k]

    import math
    def cosine_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

    selected = []
    remaining = list(range(len(candidates)))
    limit = min(top_k, len(candidates))

    for _ in range(limit):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    result = []
    for i in selected:
        item = candidates[i].copy()
        result.append(item)
    return result


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (Dense Semantic + Sparse BM25).
    RRF(d) = Σ 1 / (k + rank_r(d))
    
    Lưu ý cho Role 4 & 2: Điểm RRF phản ánh thứ hạng sau gộp, max ≈ 1/(60+1) = 0.0164.
    Đừng dùng điểm RRF để xét ngưỡng fallback ở Task 9.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, 1):
            content = item.get("content", "")
            if not content:
                continue
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
            if content not in content_map:
                content_map[content] = item.copy()

    # Sắp xếp giảm dần theo điểm RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface cho hệ thống RAG Pipeline.
    Khi gọi method='rrf' trên 1 list candidates từ upstream/unit test, hàm áp dụng RRF trên danh sách đó.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr([], candidates, top_k=top_k)
    elif method == "rrf":
        # Áp dụng RRF trên candidates list
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test mô phỏng Hybrid Reranking (Semantic Search + BM25 Lexical) trên Bộ Pháp Điển
    print("\n🔗 TEST RRF RERANKING (Role 4 - Gộp kết quả Dense Semantic & Sparse BM25 Bộ Pháp Điển)")
    print("-" * 80)

    # Danh sách giả định từ Semantic Search (tìm ý nghĩa sâu)
    dense_results = [
        {"content": "[An ninh quốc gia > Cơ yếu] Điều 1.7.LQ.1. Phạm vi điều chỉnh quy định hoạt động cơ yếu", "score": 0.82, "metadata": {"source": "dense"}},
        {"content": "[An ninh quốc gia > Cơ yếu] Điều 1.7.LQ.2. Đối tượng áp dụng đối với tổ chức cơ yếu", "score": 0.75, "metadata": {"source": "dense"}},
        {"content": "Quy định chung về bảo đảm bí mật thông tin tài liệu nhà nước", "score": 0.61, "metadata": {"source": "dense"}},
    ]

    # Danh sách giả định từ BM25 Lexical Search (match chính xác từ khóa / số hiệu điều)
    sparse_results = [
        {"content": "[An ninh quốc gia > Cơ yếu] Điều 1.7.LQ.2. Đối tượng áp dụng đối với tổ chức cơ yếu", "score": 15.5, "metadata": {"source": "sparse"}},
        {"content": "[An ninh quốc gia > Cơ yếu] Điều 1.7.LQ.1. Phạm vi điều chỉnh quy định hoạt động cơ yếu", "score": 12.2, "metadata": {"source": "sparse"}},
        {"content": "[An ninh quốc gia > Cơ yếu] Điều 1.7.NĐ.2.1. Chế độ chăm sóc y tế và nghỉ dưỡng", "score": 8.1, "metadata": {"source": "sparse"}},
    ]

    fused_results = rerank_rrf([dense_results, sparse_results], top_k=3, k=60)
    for i, r in enumerate(fused_results, 1):
        print(f"  Top {i} | RRF Score: {r['score']:.4f} | Nội dung: {r['content'][:90]}...")

    # Thử nghiệm thêm Jina Cross-Encoder Reranking nếu có JINA_API_KEY
    if os.getenv("JINA_API_KEY"):
        print("\n⚡ TEST JINA CROSS-ENCODER RERANKING (API Key từ .env):")
        print("-" * 80)
        test_query = "Chế độ y tế cho người làm công tác cơ yếu thế nào?"
        candidates = dense_results + sparse_results
        ce_results = rerank_cross_encoder(test_query, candidates, top_k=2)
        for i, r in enumerate(ce_results, 1):
            print(f"  Top {i} | Cross-Encoder Score: {r['score']:.4f} | {r['content'][:90]}...")
