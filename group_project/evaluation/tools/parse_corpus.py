"""
Parse file HTML đề mục của Bộ pháp điển thành list điều luật có cấu trúc.

Dùng để sinh golden_dataset.json (xem regenerate_golden_dataset.py) và để kiểm chứng
mọi mã điều trích dẫn trong benchmark là có thật.

Cấu trúc HTML nguồn (data/BoPhapDienDienTu/demuc/<de_muc_id>.html):
    <p class='pDieu'>    Điều 20.2.LQ.25. Thời gian thử việc      ← mã điều + tiêu đề
    <p class='pGhiChu'>  (Điều 25 Bộ luật số 45/2019/QH14, ...)    ← nguồn + ngày hiệu lực
    <p class='pNoiDung'> 1. ... 2. ...                             ← nội dung
    <p class='pChiDan'>  (Điều này có nội dung liên quan đến ...)  ← tham chiếu chéo

Chạy:
    python group_project/evaluation/tools/parse_corpus.py <de_muc_id> [keyword] [limit]
"""

import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEMUC_DIR = REPO / "data" / "BoPhapDienDienTu" / "demuc"

TAG_RE = re.compile(r"<[^>]+>")
# "Điều 20.2.LQ.25. Thời gian thử việc" -> ("Điều 20.2.LQ.25", "Thời gian thử việc").
# Non-greedy + \D ở nhóm 2 để không cắt nhầm giữa các số của mã điều.
TITLE_RE = re.compile(r"(Điều\s+[\w.]+?)\.\s+(\D.*)", re.S)


def clean(fragment: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", fragment, flags=re.S | re.I)
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def parse_demuc(de_muc_id: str, de_muc_name: str = "") -> list[dict]:
    """Trả về list {demuc, demuc_id, dieu_id, title, cite, content, refs}."""
    raw = (DEMUC_DIR / f"{de_muc_id}.html").read_text(encoding="utf-8", errors="replace")
    articles = []

    for block in re.split(r"<p class='pDieu'>", raw)[1:]:
        m = re.match(r"(.*?)</p>", block, flags=re.S)
        if not m:
            continue
        title_raw, rest = clean(m.group(1)), block[m.end():]

        cite_m = re.search(r"<p class='pGhiChu'>(.*?)</p>", rest, flags=re.S)
        content_m = re.search(
            r"<p class='pNoiDung'>(.*?)"
            r"(?=<p class='pChiDan'>|<p class='pChuong'>|<p class='pMuc'>|$)",
            rest, flags=re.S)
        refs_m = re.search(r"<p class='pChiDan'>(.*?)</p>", rest, flags=re.S)

        tm = TITLE_RE.match(title_raw)
        articles.append({
            "demuc": de_muc_name,
            "demuc_id": de_muc_id,
            "dieu_id": tm.group(1) if tm else title_raw[:40],
            "title": (tm.group(2) if tm else title_raw).strip(),
            "cite": clean(cite_m.group(1)) if cite_m else "",
            "content": clean(content_m.group(1)) if content_m else "",
            "refs": clean(refs_m.group(1)) if refs_m else "",
        })
    return articles


def is_empty_demuc(de_muc_id: str) -> bool:
    """118/320 đề mục chỉ có <div class='_content'></div> — không có nội dung."""
    p = DEMUC_DIR / f"{de_muc_id}.html"
    return not p.exists() or p.stat().st_size < 2000


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    de_muc_id = sys.argv[1]
    keyword = sys.argv[2].lower() if len(sys.argv) > 2 else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if is_empty_demuc(de_muc_id):
        print(f"⚠ Đề mục {de_muc_id} RỖNG — không có nội dung trong corpus.")
        raise SystemExit(0)

    arts = parse_demuc(de_muc_id)
    print(f"Tổng số điều: {len(arts)}\n")

    shown = 0
    for a in arts:
        if keyword and keyword not in (a["dieu_id"] + a["title"] + a["content"]).lower():
            continue
        print("=" * 90)
        print(f"{a['dieu_id']}. {a['title']}")
        print(f"  NGUỒN: {a['cite'][:200]}")
        print(f"  NỘI DUNG: {a['content'][:1200]}")
        if a["refs"]:
            print(f"  LIÊN QUAN: {a['refs'][:300]}")
        shown += 1
        if shown >= limit:
            break
