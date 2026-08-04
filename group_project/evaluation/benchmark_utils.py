"""
Benchmark utilities cho golden_dataset.json (Bộ pháp điển điện tử).

Chức năng:
    1. load / validate golden dataset
    2. Adapter sang RAGAS (cả schema mới lẫn cũ) và DeepEval
    3. Retrieval metrics KHÔNG TỐN LLM CALL: hit_rate@k, recall@k, MRR
    4. Refusal rate cho các câu unanswerable
    5. Báo cáo điểm tách theo capability

Vì sao có mục 3: RAGAS/DeepEval gọi LLM rất nhiều lần (nhiều call / metric / câu hỏi).
Model ":free" của OpenRouter giới hạn 50 request/ngày cho CẢ TÀI KHOẢN. Retrieval
metrics ở đây chỉ so khớp mã điều nên chạy được vô hạn lần, miễn phí — dùng nó để
A/B config retrieval (alpha, top_k, rerank on/off) rồi mới chạy RAGAS 1 lần trên
2 config tốt nhất.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

REQUIRED_FIELDS = [
    "id", "question", "expected_answer", "expected_context", "reference_contexts",
    "ground_truth_dieu", "source_documents", "de_muc", "de_muc_id",
    "capability", "difficulty", "answerable", "eval_mode", "notes",
]

# Mã điều pháp điển, ví dụ: "Điều 20.2.LQ.25", "Điều 20.2.NĐ.3.57"
DIEU_RE = re.compile(r"Điều\s+\d+\.\d+\.[A-ZĐ]{2}\.[\d.]+")


# =============================================================================
# Load & validate
# =============================================================================

def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(dataset: list[dict]) -> list[str]:
    """Trả về list lỗi. Rỗng = hợp lệ."""
    errors = []
    seen_ids = set()

    for i, item in enumerate(dataset):
        tag = item.get("id", f"index {i}")

        for field in REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"{tag}: thiếu field '{field}'")

        if item.get("id") in seen_ids:
            errors.append(f"{tag}: id trùng lặp")
        seen_ids.add(item.get("id"))

        if item.get("answerable"):
            if not item.get("ground_truth_dieu"):
                errors.append(f"{tag}: answerable=True nhưng ground_truth_dieu rỗng")
            if not item.get("reference_contexts"):
                errors.append(f"{tag}: answerable=True nhưng reference_contexts rỗng")
        else:
            if item.get("ground_truth_dieu"):
                errors.append(f"{tag}: answerable=False nhưng vẫn có ground_truth_dieu")

        for d in item.get("ground_truth_dieu", []):
            if not DIEU_RE.fullmatch(d):
                errors.append(f"{tag}: mã điều sai định dạng: {d!r}")

        if item.get("difficulty") not in {"easy", "medium", "hard"}:
            errors.append(f"{tag}: difficulty không hợp lệ: {item.get('difficulty')!r}")
        if item.get("eval_mode") not in {"ragas", "manual"}:
            errors.append(f"{tag}: eval_mode không hợp lệ: {item.get('eval_mode')!r}")

    return errors


# =============================================================================
# Framework adapters
# =============================================================================

def to_ragas(dataset: list[dict], runs: dict[str, dict], schema: str = "modern") -> dict:
    """
    Chuyển sang định dạng RAGAS.

    Args:
        dataset: golden dataset
        runs: {question_id: {"answer": str, "contexts": list[str]}} — output pipeline của bạn
        schema: "modern" (ragas >= 0.2) hoặc "legacy" (ragas 0.1)

    Returns:
        dict các cột, đưa thẳng vào datasets.Dataset.from_dict(...)

    Lưu ý: chỉ lấy các câu eval_mode == "ragas". Câu unanswerable/ambiguous chấm riêng
    bằng refusal_rate() vì RAGAS không đo được "từ chối đúng cách".
    """
    items = [d for d in dataset if d["eval_mode"] == "ragas" and d["id"] in runs]

    if schema == "modern":
        cols = {"user_input": [], "response": [], "retrieved_contexts": [],
                "reference": [], "reference_contexts": []}
        for d in items:
            r = runs[d["id"]]
            cols["user_input"].append(d["question"])
            cols["response"].append(r["answer"])
            cols["retrieved_contexts"].append(r["contexts"])
            cols["reference"].append(d["expected_answer"])
            cols["reference_contexts"].append(d["reference_contexts"])
        return cols

    cols = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for d in items:
        r = runs[d["id"]]
        cols["question"].append(d["question"])
        cols["answer"].append(r["answer"])
        cols["contexts"].append(r["contexts"])
        cols["ground_truth"].append(d["expected_answer"])
    return cols


def to_deepeval(dataset: list[dict], runs: dict[str, dict]) -> list:
    """Chuyển sang list[LLMTestCase] của DeepEval."""
    from deepeval.test_case import LLMTestCase

    return [
        LLMTestCase(
            input=d["question"],
            actual_output=runs[d["id"]]["answer"],
            expected_output=d["expected_answer"],
            retrieval_context=runs[d["id"]]["contexts"],
        )
        for d in dataset
        if d["eval_mode"] == "ragas" and d["id"] in runs
    ]


# =============================================================================
# Retrieval metrics — KHÔNG cần LLM
# =============================================================================

def dieu_in_chunk(dieu_id: str, chunk: dict | str) -> bool:
    """
    Chunk có chứa điều luật gold không?

    Khớp theo 2 cách: mã điều xuất hiện trong metadata['dieu_id'] (tốt nhất — nhớ đưa
    mã điều vào metadata ở Task 4), hoặc xuất hiện trong text của chunk.
    """
    if isinstance(chunk, dict):
        meta = chunk.get("metadata") or {}
        if meta.get("dieu_id") == dieu_id:
            return True
        text = chunk.get("content", "")
    else:
        text = chunk

    # Ranh giới: "Điều 20.2.LQ.2" KHÔNG được khớp nhầm vào "Điều 20.2.LQ.25",
    # nhưng "Điều 20.2.LQ.25" PHẢI khớp trong "Điều 20.2.LQ.25. Thời gian thử việc"
    # (mã điều luôn đi kèm dấu chấm trước tiêu đề) → chỉ chặn chữ số đứng ngay sau,
    # hoặc dấu chấm nối tiếp bằng chữ số.
    return re.search(re.escape(dieu_id) + r"(?!\d)(?!\.\d)", text) is not None


def retrieval_scores(dataset: list[dict], retrieved: dict[str, list], k: int = 5) -> dict:
    """
    Chấm điểm retrieval thuần — 0 LLM call.

    Args:
        dataset: golden dataset
        retrieved: {question_id: list[chunk]} với chunk là dict {'content','metadata'} hoặc str
        k: cắt top-k

    Returns:
        {
          "hit_rate": tỷ lệ câu lấy được ÍT NHẤT 1 điều gold,
          "recall": tỷ lệ điều gold lấy được (trung bình theo câu),
          "mrr": mean reciprocal rank của điều gold đầu tiên,
          "n": số câu chấm được,
          "per_question": {qid: {...}},
        }
    """
    items = [d for d in dataset if d["answerable"] and d["id"] in retrieved]
    hits, recalls, rrs, per_q = 0, [], [], {}

    for d in items:
        chunks = retrieved[d["id"]][:k]
        gold = d["ground_truth_dieu"]

        found = {g for g in gold if any(dieu_in_chunk(g, c) for c in chunks)}
        recall = len(found) / len(gold)

        rr = 0.0
        for rank, c in enumerate(chunks, 1):
            if any(dieu_in_chunk(g, c) for g in gold):
                rr = 1.0 / rank
                break

        hits += 1 if found else 0
        recalls.append(recall)
        rrs.append(rr)
        per_q[d["id"]] = {
            "capability": d["capability"],
            "recall": round(recall, 3),
            "rr": round(rr, 3),
            "found": sorted(found),
            "missed": sorted(set(gold) - found),
        }

    n = len(items) or 1
    return {
        "hit_rate": round(hits / n, 3),
        "recall": round(sum(recalls) / n, 3),
        "mrr": round(sum(rrs) / n, 3),
        "n": len(items),
        "per_question": per_q,
    }


# =============================================================================
# Refusal — cho câu unanswerable / ambiguous
# =============================================================================

REFUSAL_MARKERS = [
    "không thể xác minh", "không tìm thấy", "không có thông tin",
    "không đủ thông tin", "ngoài phạm vi", "chưa có nội dung",
    "bạn vui lòng nói rõ", "bạn đang hỏi về",
]


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in REFUSAL_MARKERS)


def refusal_rate(dataset: list[dict], runs: dict[str, dict]) -> dict:
    """
    Với câu answerable=False: từ chối = ĐÚNG.
    Với câu answerable=True: từ chối = SAI (over-refusal, retriever quá chặt).
    """
    should, should_not = [], []
    for d in dataset:
        if d["id"] not in runs:
            continue
        refused = looks_like_refusal(runs[d["id"]]["answer"])
        (should_not if d["answerable"] else should).append((d["id"], refused))

    correct = [r for _, r in should]
    over = [r for _, r in should_not]
    return {
        "correct_refusal_rate": round(sum(correct) / len(correct), 3) if correct else None,
        "over_refusal_rate": round(sum(over) / len(over), 3) if over else None,
        "failed_to_refuse": [q for q, r in should if not r],
        "over_refused": [q for q, r in should_not if r],
    }


# =============================================================================
# Report
# =============================================================================

def by_capability(dataset: list[dict], per_question: dict, metric: str = "recall") -> dict:
    """Gộp điểm theo capability để biết NĂNG LỰC NÀO đang yếu."""
    buckets = defaultdict(list)
    for qid, scores in per_question.items():
        cap = scores.get("capability") or next(
            (d["capability"] for d in dataset if d["id"] == qid), "unknown")
        if metric in scores:
            buckets[cap].append(scores[metric])
    return {c: round(sum(v) / len(v), 3) for c, v in sorted(buckets.items())}


def summary(dataset: list[dict]) -> str:
    caps, diffs, dms = defaultdict(int), defaultdict(int), defaultdict(int)
    for d in dataset:
        caps[d["capability"]] += 1
        diffs[d["difficulty"]] += 1
        for m in d["de_muc"]:
            dms[m] += 1

    lines = [f"Golden dataset: {len(dataset)} câu hỏi",
             f"  answerable: {sum(1 for d in dataset if d['answerable'])} | "
             f"unanswerable: {sum(1 for d in dataset if not d['answerable'])}",
             f"  độ khó: " + ", ".join(f"{k}={diffs[k]}" for k in ("easy", "medium", "hard")),
             "", "Capability:"]
    lines += [f"  {c:28s} {n}" for c, n in sorted(caps.items())]
    lines += ["", "Đề mục:"]
    lines += [f"  {m:28s} {n}" for m, n in sorted(dms.items(), key=lambda x: -x[1])]
    return "\n".join(lines)


if __name__ == "__main__":
    ds = load_golden_dataset()
    errs = validate(ds)
    if errs:
        print(f"✗ {len(errs)} lỗi:")
        for e in errs:
            print("   -", e)
        raise SystemExit(1)
    print("✓ Golden dataset hợp lệ\n")
    print(summary(ds))
