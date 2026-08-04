"""
Đồng bộ lại golden_dataset.json với corpus.

Với mỗi mã điều trong `ground_truth_dieu`, script đọc lại điều luật từ HTML gốc và
ghi đè `reference_contexts` + `source_documents` bằng NGUYÊN VĂN corpus. Nhờ vậy
ground truth không bao giờ lệch khỏi dữ liệu thật (và không phải gõ tay).

Dùng khi:
    - thêm/sửa câu hỏi (điền `ground_truth_dieu` rồi chạy script để tự sinh context)
    - corpus được tải lại / cập nhật
    - muốn kiểm tra mọi mã điều trích dẫn còn tồn tại

Chạy:
    python group_project/evaluation/tools/refresh_contexts.py --check   # chỉ kiểm tra
    python group_project/evaluation/tools/refresh_contexts.py           # ghi đè file
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_corpus import is_empty_demuc, parse_demuc  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "group_project" / "evaluation" / "golden_dataset.json"
HIERARCHY = REPO / "data" / "bophapdien_hierarchy.json"

# Đề mục nằm trong phạm vi benchmark: tên -> de_muc_id
SCOPE = {
    "Lao động": "2efd8c6f-509f-4207-84b6-6b22ff780f2a",
    "Bảo hiểm y tế": "583b913f-4d4b-46da-911c-112342c6ea49",
    "Hôn nhân và gia đình": "4913a1cf-5f78-471c-a807-ed5f8c57aaee",
    "Cư trú": "6a501e4a-ba3f-40e2-902f-af4a52b14bb4",
    "An ninh mạng": "1fd42d83-9d78-4dd4-b6b6-73e9bd3472e1",
    "Doanh nghiệp": "319387a3-090a-47a5-82f6-a9bacbe5d341",
    "Thuế thu nhập cá nhân": "ab116df3-d3ed-4a95-95bf-0d6b8962e724",
    "Dân sự": "eb0e4753-243e-4344-90e6-70aaf5188a6d",
}

VANBAN_RE = re.compile(r"((?:Bộ luật|Luật|Nghị định|Thông tư|Pháp lệnh|Nghị quyết)\s+số\s+[\w/\-]+)")


def build_article_index() -> dict[str, dict]:
    index = {}
    for name, dmid in SCOPE.items():
        if is_empty_demuc(dmid):
            print(f"  ⚠ {name}: đề mục RỖNG, bỏ qua")
            continue
        arts = parse_demuc(dmid, name)
        for a in arts:
            index.setdefault(a["dieu_id"], a)
        print(f"  {name:25s} {len(arts):5d} điều")
    return index


def main(check_only: bool = False) -> int:
    print("Đọc corpus...")
    index = build_article_index()
    print(f"  → {len(index)} điều duy nhất\n")

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    missing, changed = [], []

    for item in dataset:
        gold = item.get("ground_truth_dieu", [])
        if not gold:
            continue

        unknown = [d for d in gold if d not in index]
        if unknown:
            missing.append((item["id"], unknown))
            continue

        contexts, sources, demucs = [], [], []
        for d in gold:
            a = index[d]
            contexts.append(f"{d}. {a['title']}\n{a['cite']}\n{a['content']}")
            m = VANBAN_RE.search(a["cite"])
            src = m.group(1) if m else a["cite"][:60]
            if src not in sources:
                sources.append(src)
            if a["demuc"] not in demucs:
                demucs.append(a["demuc"])

        if item.get("reference_contexts") != contexts:
            changed.append(item["id"])
        item["reference_contexts"] = contexts
        item["source_documents"] = sources
        item["de_muc"] = demucs
        item["de_muc_id"] = [SCOPE[n] for n in demucs]

    if missing:
        print("✗ Mã điều KHÔNG tồn tại trong corpus:")
        for qid, ids in missing:
            print(f"   {qid}: {', '.join(ids)}")
        return 1

    print(f"✓ Tất cả mã điều đều tồn tại trong corpus")
    if changed:
        print(f"  {len(changed)} câu có context khác file hiện tại: {', '.join(changed)}")
    else:
        print("  golden_dataset.json đã khớp corpus, không cần cập nhật")

    if check_only:
        return 1 if changed else 0

    if changed:
        DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ Đã ghi {DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
