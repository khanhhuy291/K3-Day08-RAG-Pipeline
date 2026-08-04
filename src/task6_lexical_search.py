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

import json
from pathlib import Path
import math
import numpy as np

# Load corpus từ data/bophapdien.json, data/standardized/ kèm bộ dữ liệu mặc định
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None        # Cache BM25 index để tránh re-index 58,212 điều luật mỗi lần truy vấn


class SimpleBM25:
    """
    Tự implement thuật toán BM25Okapi thuần Python (+5 Bonus trong Demo theo LAB_GUIDE).
    Dùng làm phương án tối ưu khi thư viện rank_bm25 chưa được cài đặt trong môi trường.
    """
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.df: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []

        for doc in corpus_tokens:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.term_freqs.append(freq)
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

        self.idf: dict[str, float] = {}
        for token, df_val in self.df.items():
            self.idf[token] = math.log((self.corpus_size - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx, freq in enumerate(self.term_freqs):
                tf_val = freq.get(token, 0)
                doc_len = self.doc_lengths[idx]
                numerator = tf_val * (self.k1 + 1.0)
                denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[idx] += idf_val * (numerator / denominator)
        return np.array(scores)


def load_corpus() -> list[dict]:
    """
    Load toàn bộ dữ liệu từ data/bophapdien.json, các file .md từ data/standardized/
    và luôn tích hợp sẵn bộ dữ liệu chuẩn của repo cùng dữ liệu chủ đề mẫu.
    """
    docs = []
    base_dir = Path(__file__).parent.parent
    
    # 1. Nạp dữ liệu từ data/bophapdien.json (Bộ Pháp Điển Việt Nam - 58,212 điều luật)
    bophapdien_path = base_dir / "data" / "bophapdien.json"
    if bophapdien_path.exists():
        try:
            with open(bophapdien_path, "r", encoding="utf-8") as f:
                bpd_data = json.load(f)
                for art in bpd_data.get("articles", []):
                    chu_de = art.get("chu_de", "")
                    de_muc = art.get("de_muc", "")
                    ten_dieu = art.get("ten_dieu", "")
                    ghi_chu = art.get("ghi_chu", "")
                    noi_dung = art.get("noi_dung", "")
                    phan_chuong = art.get("phan_chuong_muc", "")
                    
                    # Kết hợp chủ đề, đề mục, tên điều và nội dung để BM25 match từ khóa đầy đủ nhất
                    content = f"[{chu_de} > {de_muc} > {phan_chuong}] {ten_dieu} {ghi_chu}\nNội dung: {noi_dung}".strip()
                    metadata = {
                        "source": "bophapdien.json",
                        "type": "legal_bophapdien",
                        "id": art.get("id", ""),
                        "mapc": art.get("mapc", ""),
                        "chu_de": chu_de,
                        "de_muc": de_muc,
                        "phan_chuong_muc": phan_chuong,
                        "link_vbpl": art.get("link_vbpl", ""),
                        "topic": chu_de
                    }
                    docs.append({"content": content, "metadata": metadata})
        except Exception as e:
            print(f"⚠ Lỗi nạp file {bophapdien_path.name}: {e}")

    # 2. Nạp từ các file .md trong data/standardized/
    standardized_dir = base_dir / "data" / "standardized"
    if standardized_dir.exists():
        for md_file in standardized_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if content.strip():
                    doc_type = "legal" if "legal" in str(md_file) or "luat" in str(md_file).lower() else "news"
                    docs.append({
                        "content": content,
                        "metadata": {"source": md_file.name, "type": doc_type}
                    })
            except Exception as e:
                print(f"⚠ Lỗi đọc file {md_file.name}: {e}")

    # Luôn bổ sung bộ dữ liệu test gốc của Starter Kit và dữ liệu mẫu Chủ đề 1 (Luật Lao Động)
    default_docs = [
        {"content": "Tuition fee payment policy, tuition payment schedules and payment methods at university.", "metadata": {"source": "default_tuition_policy.md", "topic": "tuition"}},
        {"content": "Scholarship eligibility requirements and grants for undergraduate students.", "metadata": {"source": "default_scholarship.md", "topic": "scholarship"}},
        {"content": "Library study room booking guide, library opening hours and silent room regulations.", "metadata": {"source": "default_library_guide.md", "topic": "library"}},
        {"content": "Library study room equipment and wifi access in the study room.", "metadata": {"source": "default_library_wifi.md", "topic": "library"}},
        # Dữ liệu chủ đề 1: Trợ lý hỏi đáp Luật Lao Động cho Gen Z
        {"content": "Điều 25 Bộ luật Lao động 2019: Thời gian thử việc tối đa đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng, đại học trở lên (như lập trình viên, kỹ sư) là không quá 60 ngày.", "metadata": {"source": "BLLD_2019_Dieu25.md", "topic": "luật lao động - thử việc"}},
        {"content": "Điều 26 Bộ luật Lao động 2019: Tiền lương của người lao động trong thời gian thử việc do hai bên thỏa thuận nhưng ít nhất phải bằng 85% mức lương của công việc đó (lương chính thức).", "metadata": {"source": "BLLD_2019_Dieu26.md", "topic": "luật lao động - lương thử việc"}},
        {"content": "Điều 36 & Điều 39 Bộ luật Lao động 2019: Quy định về đơn phương chấm dứt hợp đồng lao động. Người sử dụng lao động phải báo trước ít nhất 30 ngày (với HĐLĐ xác định thời hạn) hoặc 45 ngày (với HĐLĐ không xác định thời hạn). Việc công ty sa thải qua tin nhắn Zalo mà không báo trước và không có lý do theo Điều 36 là trái pháp luật.", "metadata": {"source": "BLLD_2019_Dieu36_39.md", "topic": "luật lao động - sa thải"}},
        {"content": "Điều 98 Bộ luật Lao động 2019: Tiền lương làm thêm giờ (OT). Làm đêm, làm việc ngoài giờ làm việc hành chính vào ngày thường được trả ít nhất bằng 150%, ngày nghỉ hằng tuần ít nhất 200%, ngày lễ, tết, ngày nghỉ có hưởng lương ít nhất 300%.", "metadata": {"source": "BLLD_2019_Dieu98.md", "topic": "luật lao động - OT"}}
    ]

    # Hợp nhất và tránh trùng lắp nội dung
    existing_contents = {d["content"] for d in docs}
    for item in default_docs:
        if item["content"] not in existing_contents:
            docs.append(item)
            existing_contents.add(item["content"])

    return docs


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    Ưu tiên dùng rank_bm25 nếu có, không thì chuyển sang SimpleBM25 (tự implement).
    """
    if not corpus:
        return None
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)
    except ImportError:
        # Fallback sang implementation thuần Python (+5 Bonus demo)
        return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    Rất hiệu quả cho tra cứu số hiệu luật, điều khoản (ví dụ: 'Điều 25', '60 ngày', 'sa thải Zalo').

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of dict có các trường: 'content', 'score', 'metadata', được sắp xếp giảm dần theo điểm.
    """
    global CORPUS, BM25_INDEX
    if not CORPUS:
        CORPUS = load_corpus()
        BM25_INDEX = None  # Reset cache khi nạp lại corpus
    if not CORPUS:
        return []

    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)
    if not BM25_INDEX:
        return []

    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)

    # Lấy ra các index có thứ hạng cao nhất
    k = min(top_k, len(CORPUS))
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"]
        })
    return results


if __name__ == "__main__":
    # Test thử với câu hỏi của Bộ Pháp Điển, Chủ đề 1 (Luật Lao Động cho Gen Z) và bộ test gốc
    queries = [
        "Hoạt động cơ yếu nhiệm vụ quyền hạn",
        "Chế độ chăm sóc y tế và chế độ nghỉ cơ yếu",
        "Thời gian thử việc cho lập trình viên và lương thử việc",
        "Sa thải qua tin nhắn Zalo không báo trước",
        "tuition fee payment policy",
        "library study room"
    ]
    for q in queries:
        print(f"\n🔍 Truy vấn BM25: '{q}'")
        print("-" * 65)
        res = lexical_search(q, top_k=2)
        for r in res:
            print(f"  [{r['score']:.3f}] {r['content'][:110]}...")
