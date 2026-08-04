"""
Kiểm tra tính toàn vẹn của corpus trước khi index / chấm benchmark.

Ba phép đối chiếu:
    1. data/bophapdien.json   vs  HTML gốc (demuc/*.html)   — JSON có thiếu điều không?
    2. bophapdien_hierarchy.json vs HTML                    — đề mục nào rỗng?
    3. golden_dataset.json    vs  cả hai nguồn              — gold article có tồn tại không?

Vì sao cần: bản `bophapdien.json` sinh ngày 2026-08-04 bị THIẾU 7.785/65.997 điều
(11,8%) ở 14 đề mục do quá trình convert dừng giữa chừng. Nếu index từ file đó mà
không biết, benchmark sẽ trượt vì LỖI DỮ LIỆU chứ không phải vì pipeline RAG kém.

Chạy:
    python group_project/evaluation/tools/audit_corpus.py
    python group_project/evaluation/tools/audit_corpus.py --strict   # exit 1 nếu có lệch
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_corpus import is_empty_demuc  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEMUC_DIR = REPO / "data" / "BoPhapDienDienTu" / "demuc"
JSON_PATH = REPO / "data" / "bophapdien.json"
HIERARCHY_PATH = REPO / "data" / "bophapdien_hierarchy.json"
DATASET_PATH = REPO / "group_project" / "evaluation" / "golden_dataset.json"

DIEU_PREFIX_RE = re.compile(r"(Điều\s+[\w.]+?)\.\s")


def html_article_count(de_muc_id: str) -> int:
    """Đếm điều trong HTML gốc — mỗi điều là một <p class='pDieu'>."""
    p = DEMUC_DIR / f"{de_muc_id}.html"
    if not p.exists():
        return -1
    return p.read_text(encoding="utf-8", errors="replace").count("<p class='pDieu'>")


def load_json_corpus():
    if not JSON_PATH.exists():
        return None
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def audit_json_vs_html(corpus, hierarchy) -> tuple[list, list]:
    """Trả về (mismatches, empties)."""
    json_count = Counter(a["de_muc_id"] for a in corpus["articles"])
    mismatches, empties = [], []

    for chu_de in hierarchy:
        for dm in chu_de["de_mucs"]:
            dmid, name = dm["de_muc_id"], dm["de_muc_name"]
            if is_empty_demuc(dmid):
                empties.append((name, dmid))
                continue
            html_n, json_n = html_article_count(dmid), json_count.get(dmid, 0)
            if html_n != json_n:
                mismatches.append((name, dmid, json_n, html_n))

    return mismatches, empties


def audit_benchmark(corpus) -> dict:
    """Gold article của benchmark có mặt trong bophapdien.json không?"""
    index = set()
    for a in corpus["articles"]:
        m = DIEU_PREFIX_RE.match(a["ten_dieu"])
        if m:
            index.add(m.group(1))

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    missing, affected = set(), []
    for item in dataset:
        gone = [g for g in item["ground_truth_dieu"] if g not in index]
        if gone:
            missing.update(gone)
            affected.append((item["id"], gone))
    return {"missing": sorted(missing), "affected": affected,
            "total_gold": len({g for d in dataset for g in d["ground_truth_dieu"]})}


def main(strict: bool = False) -> int:
    hierarchy = json.loads(HIERARCHY_PATH.read_text(encoding="utf-8"))
    total_demuc = sum(len(c["de_mucs"]) for c in hierarchy)
    print(f"hierarchy: {len(hierarchy)} chủ đề, {total_demuc} đề mục\n")

    corpus = load_json_corpus()
    if corpus is None:
        print(f"⚠ Không tìm thấy {JSON_PATH.name} — bỏ qua đối chiếu JSON.")
        return 0

    meta = corpus.get("meta", {})
    print(f"bophapdien.json: {len(corpus['articles'])} điều "
          f"(meta ghi {meta.get('total_articles')}, tạo {meta.get('created_at')})")

    mismatches, empties = audit_json_vs_html(corpus, hierarchy)
    html_total = sum(html_article_count(dm["de_muc_id"])
                     for c in hierarchy for dm in c["de_mucs"]
                     if not is_empty_demuc(dm["de_muc_id"]))
    print(f"HTML gốc       : {html_total} điều")
    print(f"\nĐề mục rỗng (không có nội dung): {len(empties)}/{total_demuc}")

    if mismatches:
        lost = sum(h - j for _, _, j, h in mismatches)
        print(f"\n✗ {len(mismatches)} ĐỀ MỤC LỆCH — JSON thiếu {lost} điều "
              f"({lost / html_total * 100:.1f}% corpus)\n")
        print(f"  {'ĐỀ MỤC':44s} {'JSON':>7s} {'HTML':>7s} {'THIẾU':>7s}")
        print("  " + "-" * 68)
        for name, _, j, h in sorted(mismatches, key=lambda r: r[3] - r[2], reverse=True):
            print(f"  {name[:43]:44s} {j:7d} {h:7d} {h - j:7d}")
        print("\n  → JSON được sinh bằng cách crawl website, quá trình bị dừng giữa chừng.")
        print("    HTML trong data/BoPhapDienDienTu/demuc/ mới là bản ĐẦY ĐỦ.")
        print("    Cần chạy lại converter, hoặc index thẳng từ HTML.")
    else:
        print("\n✓ JSON khớp HTML ở mọi đề mục")

    bm = audit_benchmark(corpus)
    print(f"\nBenchmark: {bm['total_gold']} gold article")
    if bm["missing"]:
        print(f"✗ {len(bm['missing'])} gold article KHÔNG có trong bophapdien.json:")
        for d in bm["missing"]:
            print(f"    {d}")
        print("  Câu hỏi bị ảnh hưởng nếu index từ JSON:")
        for qid, gone in bm["affected"]:
            print(f"    {qid}: thiếu {', '.join(gone)}")
        print("  → Các câu này sẽ trượt vì LỖI DỮ LIỆU, không phải vì pipeline kém.")
    else:
        print("✓ Mọi gold article đều có trong bophapdien.json")

    failed = bool(mismatches or bm["missing"])
    return 1 if (strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main(strict="--strict" in sys.argv))
