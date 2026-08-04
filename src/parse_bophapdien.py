"""
Script trích xuất và lọc dữ liệu từ Bộ Pháp Điển Điện Tử sang định dạng JSON.

Nguồn dữ liệu:
  - Metadata: BoPhapDienDienTu/jsonData.js (var jdChuDe, var jdDeMuc, var jdAllTree)
  - Nội dung: BoPhapDienDienTu/demuc/*.html (320 file HTML chứa các Điều luật)

Đầu ra JSON:
  - data/standardized/legal/bophapdien.json (Toàn bộ 39,500+ Điều luật dạng phẳng)
  - data/standardized/legal/bophapdien_hierarchy.json (Cấu trúc phân cấp Chủ đề -> Đề mục)
  - data/standardized/legal/by_demuc/ (320 file JSON riêng cho từng Đề mục)
"""

import json
import os
import re
import time
from pathlib import Path
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
BPD_DIR = PROJECT_ROOT / "BoPhapDienDienTu"
DEMUC_DIR = BPD_DIR / "demuc"
JSON_DATA_PATH = BPD_DIR / "jsonData.js"

OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized" / "legal"
OUTPUT_BY_DEMUC_DIR = OUTPUT_DIR / "by_demuc"


def load_metadata():
    """Đọc dữ liệu metadata từ file jsonData.js."""
    print("1. Đang đọc metadata từ jsonData.js...")
    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        l1 = f.readline().strip()
        l2 = f.readline().strip()
        l3 = f.readline().strip()

    chu_de_list = json.loads(l1[l1.find("[") : l1.rfind("]") + 1])
    de_muc_list = json.loads(l2[l2.find("[") : l2.rfind("]") + 1])
    all_tree_list = json.loads(l3[l3.find("[") : l3.rfind("]") + 1])

    chu_de_map = {item["Value"]: item["Text"] for item in chu_de_list}
    de_muc_map = {item["Value"]: item for item in de_muc_list}
    tree_map = {item["MAPC"]: item for item in all_tree_list if "MAPC" in item}

    print(
        f"✓ Đã tải {len(chu_de_map)} Chủ đề, {len(de_muc_map)} Đề mục, {len(tree_map)} nút cây cấu trúc."
    )
    return chu_de_list, de_muc_list, chu_de_map, de_muc_map, tree_map


def clean_text(text: str) -> str:
    """Làm sạch khoảng trắng thừa trong chuỗi text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_demuc_html(file_path: Path, chu_de_name: str, chu_de_id: str, de_muc_name: str, de_muc_id: str, tree_map: dict):
    """Trích xuất danh sách các Điều từ 1 file HTML đề mục."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    content_div = soup.find("div", class_="_content")
    if not content_div:
        return []

    articles = []
    current_header = []
    current_dieu = None

    for p in content_div.find_all(["p"], recursive=False):
        p_class = p.get("class", [])
        if "pChuong" in p_class:
            txt = clean_text(p.get_text())
            if txt:
                current_header.append(txt)
        elif "pDieu" in p_class:
            if current_dieu:
                articles.append(current_dieu)

            anchor = p.find("a")
            mapc = anchor.get("name", "") if anchor else ""
            ten_dieu = clean_text(p.get_text())

            tree_info = tree_map.get(mapc, {})
            node_id = tree_info.get("ID", "")

            current_dieu = {
                "id": node_id,
                "mapc": mapc,
                "ten_dieu": ten_dieu,
                "chu_de": chu_de_name,
                "chu_de_id": chu_de_id,
                "de_muc": de_muc_name,
                "de_muc_id": de_muc_id,
                "phan_chuong_muc": " > ".join(current_header[-3:]),
                "ghi_chu": "",
                "link_vbpl": "",
                "noi_dung": "",
                "chi_dan": [],
            }
        elif current_dieu:
            if "pGhiChu" in p_class:
                current_dieu["ghi_chu"] = clean_text(p.get_text())
                a_tag = p.find("a")
                if a_tag and a_tag.get("href"):
                    current_dieu["link_vbpl"] = a_tag.get("href")
            elif "pNoiDung" in p_class:
                txt = clean_text(p.get_text())
                if txt:
                    if current_dieu["noi_dung"]:
                        current_dieu["noi_dung"] += "\n" + txt
                    else:
                        current_dieu["noi_dung"] = txt
            elif "pChiDan" in p_class:
                txt = clean_text(p.get_text())
                if txt:
                    current_dieu["chi_dan"].append(txt)

    if current_dieu:
        articles.append(current_dieu)

    return articles


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BY_DEMUC_DIR.mkdir(parents=True, exist_ok=True)

    chu_de_list, de_muc_list, chu_de_map, de_muc_map, tree_map = load_metadata()

    html_files = list(DEMUC_DIR.glob("*.html"))
    print(f"2. Đang phân tích và trích xuất {len(html_files)} file Đề mục HTML...")

    all_articles = []
    demuc_summary = []

    for idx, file_path in enumerate(html_files, 1):
        demuc_id = file_path.stem
        de_muc_info = de_muc_map.get(demuc_id, {})
        de_muc_name = de_muc_info.get("Text", "")
        chu_de_id = de_muc_info.get("ChuDe", "")
        chu_de_name = chu_de_map.get(chu_de_id, "")

        articles = parse_demuc_html(
            file_path=file_path,
            chu_de_name=chu_de_name,
            chu_de_id=chu_de_id,
            de_muc_name=de_muc_name,
            de_muc_id=demuc_id,
            tree_map=tree_map,
        )

        all_articles.extend(articles)

        # Lưu file JSON riêng cho từng Đề mục
        demuc_json_path = OUTPUT_BY_DEMUC_DIR / f"{demuc_id}.json"
        with open(demuc_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "de_muc_id": demuc_id,
                    "de_muc_name": de_muc_name,
                    "chu_de_id": chu_de_id,
                    "chu_de_name": chu_de_name,
                    "total_articles": len(articles),
                    "articles": articles,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        demuc_summary.append(
            {
                "de_muc_id": demuc_id,
                "de_muc_name": de_muc_name,
                "chu_de_name": chu_de_name,
                "total_articles": len(articles),
                "json_file": str(demuc_json_path.relative_to(PROJECT_ROOT)),
            }
        )

        if idx % 50 == 0 or idx == len(html_files):
            print(f"   - Đã xử lý {idx}/{len(html_files)} Đề mục ({len(all_articles)} Điều).")

    # 3. Xuất file bophapdien.json master chứa tất cả Điều luật
    master_json_path = OUTPUT_DIR / "bophapdien.json"
    print(f"3. Đang ghi file master JSON: {master_json_path}...")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "total_chu_de": len(chu_de_list),
                    "total_de_muc": len(de_muc_list),
                    "total_articles": len(all_articles),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "articles": all_articles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4. Xuất file hierarchy.json (Chủ đề -> Đề mục)
    hierarchy_json_path = OUTPUT_DIR / "bophapdien_hierarchy.json"
    hierarchy_data = []
    for chu_de in chu_de_list:
        cd_id = chu_de["Value"]
        cd_text = chu_de["Text"]
        demucs_in_cd = [dm for dm in de_muc_list if dm.get("ChuDe") == cd_id]
        hierarchy_data.append(
            {
                "chu_de_id": cd_id,
                "chu_de_name": cd_text,
                "stt": chu_de.get("STT"),
                "de_mucs": [
                    {
                        "de_muc_id": dm["Value"],
                        "de_muc_name": dm["Text"],
                        "stt": dm.get("STT"),
                    }
                    for dm in demucs_in_cd
                ],
            }
        )

    with open(hierarchy_json_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)

    t1 = time.time()
    print("=" * 60)
    print(f"✓ THÀNH CÔNG! Đã lọc toàn bộ dữ liệu Bộ Pháp Điển sang JSON.")
    print(f"  - Tổng số Chủ đề: {len(chu_de_list)}")
    print(f"  - Tổng số Đề mục: {len(de_muc_list)}")
    print(f"  - Tổng số Điều luật đã lọc: {len(all_articles)}")
    print(f"  - Thời gian xử lý: {t1 - t0:.2f} giây")
    print(f"  - File Master JSON: {master_json_path}")
    print(f"  - File Cấu trúc Phân cấp: {hierarchy_json_path}")
    print(f"  - Thư mục JSON theo Đề mục: {OUTPUT_BY_DEMUC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
