"""
Test cho benchmark_utils + golden_dataset.

Chạy:  pytest group_project/evaluation/test_benchmark_utils.py -v
Hoặc:  python group_project/evaluation/test_benchmark_utils.py

Không cần API key, không gọi LLM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_utils import (  # noqa: E402
    by_capability, dieu_in_chunk, load_golden_dataset, looks_like_refusal,
    refusal_rate, retrieval_scores, to_ragas, validate,
)

DS = load_golden_dataset()


# =============================================================================
# Golden dataset
# =============================================================================

def test_dataset_is_valid():
    assert validate(DS) == []


def test_dataset_size_and_split():
    assert len(DS) >= 20, "yêu cầu đề bài: tối thiểu 20 câu"
    assert sum(1 for d in DS if not d["answerable"]) >= 3, "cần đủ câu bẫy unanswerable"


def test_all_capabilities_covered():
    """Mỗi trục năng lực RAG phải có ít nhất 1 câu."""
    required = {
        "simple_factual", "numeric_precision", "enumeration", "lexical_exact",
        "paraphrase_colloquial", "multi_hop", "cross_domain", "comparative",
        "temporal_amendment", "stale_law_trap", "false_premise", "ambiguous",
        "noise_robustness", "unanswerable_out_of_scope", "unanswerable_empty_demuc",
    }
    assert required - {d["capability"] for d in DS} == set()


def test_reference_contexts_contain_their_dieu_id():
    """reference_contexts phải trích đúng điều đã khai trong ground_truth_dieu."""
    for d in DS:
        for dieu in d["ground_truth_dieu"]:
            assert any(dieu_in_chunk(dieu, c) for c in d["reference_contexts"]), \
                f"{d['id']}: reference_contexts không chứa {dieu}"


def test_reference_contexts_not_truncated():
    """
    Regression: trước đây reference_contexts bị cắt ở 1500 ký tự, làm mất khoản 4–6
    của Điều 1.11.LQ.8 (6 khoản) → chính ground truth bị thiếu, context_recall chấm sai.
    Điều luật luôn kết thúc bằng dấu câu; cắt giữa chừng thì không.
    """
    for d in DS:
        for c in d["reference_contexts"]:
            assert c.rstrip()[-1] in ".;:\"')", f"{d['id']}: context bị cắt giữa chừng: ...{c[-60:]!r}"


def test_long_enumeration_article_is_complete():
    """Điều 1.11.LQ.8 (Q19) phải có đủ 6 khoản."""
    ctx = "\n".join(next(d for d in DS if d["id"] == "Q19")["reference_contexts"])
    for khoan in ("1.", "2.", "3.", "4. Chống lại", "5. Lợi dụng", "6. Hành vi khác"):
        assert khoan in ctx, f"Q19 thiếu khoản {khoan!r}"


# =============================================================================
# dieu_in_chunk — khớp mã điều
# =============================================================================

def test_dieu_match_boundaries():
    # không được khớp nhầm mã ngắn vào mã dài
    assert not dieu_in_chunk("Điều 20.2.LQ.2", "xx Điều 20.2.LQ.25 yy")
    assert not dieu_in_chunk("Điều 20.2.NĐ.3.5", "Điều 20.2.NĐ.3.57. Tiền lương")
    # dạng thường gặp: mã điều + dấu chấm + tiêu đề
    assert dieu_in_chunk("Điều 20.2.LQ.25", "xx Điều 20.2.LQ.25. Thời gian thử việc")
    assert dieu_in_chunk("Điều 9.1.LQ.429", "...Điều 9.1.LQ.429.")
    assert dieu_in_chunk("Điều 20.2.LQ.25", "Điều 20.2.LQ.25")


def test_dieu_match_via_metadata_and_dict():
    assert dieu_in_chunk("Điều 20.2.LQ.25", {"metadata": {"dieu_id": "Điều 20.2.LQ.25"}, "content": ""})
    assert not dieu_in_chunk("Điều 20.2.LQ.25", {"metadata": {}, "content": "không có gì"})


# =============================================================================
# Retrieval metrics
# =============================================================================

SAMPLE = {
    "Q01": [{"content": "Điều 20.2.LQ.25. Thời gian thử việc", "metadata": {}}],
    "Q11": [{"content": "rác", "metadata": {}},
            {"content": "rác", "metadata": {}},
            {"content": "Điều 2.2.LQ.13. mức đóng", "metadata": {}}],
    "Q22": [{"content": "không liên quan", "metadata": {}}],
}


def test_retrieval_scores_overall():
    r = retrieval_scores(DS, SAMPLE, k=5)
    assert r["n"] == 3
    assert r["hit_rate"] == 0.667      # Q01, Q11 trúng; Q22 trượt
    assert r["recall"] == 0.5          # (1.0 + 0.5 + 0.0) / 3
    assert r["mrr"] == 0.444           # (1/1 + 1/3 + 0) / 3


def test_retrieval_scores_per_question():
    r = retrieval_scores(DS, SAMPLE, k=5)["per_question"]
    assert r["Q01"]["recall"] == 1.0 and r["Q01"]["rr"] == 1.0
    # Q11 là câu cross_domain, 2 điều gold ở 2 đề mục — chỉ lấy được 1
    assert r["Q11"]["recall"] == 0.5 and r["Q11"]["rr"] == 0.333
    assert r["Q11"]["missed"] == ["Điều 20.2.LQ.139"]
    assert r["Q22"]["recall"] == 0.0


def test_top_k_cutoff_is_applied():
    """Gold nằm ở rank 3 → k=2 phải trượt."""
    assert retrieval_scores(DS, SAMPLE, k=2)["per_question"]["Q11"]["recall"] == 0.0


def test_by_capability():
    r = retrieval_scores(DS, SAMPLE, k=5)
    assert by_capability(DS, r["per_question"]) == {
        "cross_domain": 0.5, "multi_hop": 0.0, "simple_factual": 1.0,
    }


# =============================================================================
# Refusal
# =============================================================================

def test_looks_like_refusal():
    assert looks_like_refusal("Tôi không thể xác minh thông tin này từ nguồn hiện có")
    assert not looks_like_refusal("Thời gian thử việc không quá 60 ngày.")


def test_refusal_rate_counts_both_directions():
    runs = {
        "Q26": {"answer": "Tôi không thể xác minh thông tin này.", "contexts": []},  # đúng
        "Q25": {"answer": "Hạn mức là 3 hecta.", "contexts": []},                    # bịa → sai
        "Q01": {"answer": "Không quá 60 ngày.", "contexts": []},                     # đúng
        "Q02": {"answer": "Tôi không tìm thấy thông tin.", "contexts": []},          # over-refusal
    }
    r = refusal_rate(DS, runs)
    assert r["correct_refusal_rate"] == 0.5 and r["failed_to_refuse"] == ["Q25"]
    assert r["over_refusal_rate"] == 0.5 and r["over_refused"] == ["Q02"]


# =============================================================================
# Adapters
# =============================================================================

def test_ragas_adapters_and_manual_exclusion():
    runs = {d["id"]: {"answer": "a", "contexts": ["c1", "c2"]} for d in DS}
    modern, legacy = to_ragas(DS, runs, "modern"), to_ragas(DS, runs, "legacy")

    assert list(modern) == ["user_input", "response", "retrieved_contexts",
                            "reference", "reference_contexts"]
    assert list(legacy) == ["question", "answer", "contexts", "ground_truth"]

    n_ragas = sum(1 for d in DS if d["eval_mode"] == "ragas")
    assert len(modern["user_input"]) == n_ragas == 24
    assert len(legacy["question"]) == n_ragas
    # tất cả cột phải cùng độ dài
    assert len({len(v) for v in modern.values()}) == 1


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
