"""
Task 4 — Chunking & Indexing Dữ Liệu Bộ Pháp Điển Vào Vector Store.

Chiến lược Chunking (Legal RAG Context-Preserving Strategy):
    1. Context Prefix Injection: Chèn thông tin Chủ đề, Đề mục, Cấu trúc Chương/Mục và Tên Điều vào đầu mỗi Chunk.
    2. Article Preservation (Bảo toàn Điều luật):
       - Các Điều luật có độ dài <= 1500 ký tự sẽ được GIỮ NGUYÊN 1 Chunk trọn vẹn.
    3. Clause-Level Splitting (Tách theo Khoản/Mục cho Điều dài):
       - Các Điều luật > 1500 ký tự sẽ được tách theo Khoản (1., 2., 3...) sử dụng RecursiveCharacterTextSplitter
         với chunk_size=1000, chunk_overlap=150 (đã trừ hao độ dài header để chunk cuối không vượt ngân sách).
    4. Metadata Enrichment: Đính kèm các thông tin chu_de_id, de_muc_id, mapc, link_vbpl vào metadata để hỗ trợ Metadata Filtering.

Embedding Model:
    - intfloat/multilingual-e5-small (384 dim), local, multilingual và phù hợp retrieval.

Vector Store:
    - ChromaDB persistent store (collection: 'bophapdien_docs').
"""

import json
import re
import time
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=150, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_text(self, text: str) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text]
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start = end - self.chunk_overlap
            return chunks


PROJECT_ROOT = Path(__file__).parent.parent
BOPHAPDIEN_JSON_PATH = PROJECT_ROOT / "data" / "standardized" / "legal" / "bophapdien.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
PROGRESS_FILE = PROJECT_ROOT / "chroma_db" / ".index_progress.json"

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_SINGLE_ARTICLE_LENGTH = 1500  # Điều <= 1500 ký tự sẽ giữ nguyên 1 chunk
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "legal_context_injection"

# Regex tách khoản linh hoạt: bắt "\n1. ", "\n12. ", "\n1) ", v.v. (không giới hạn 1-5)
CLAUSE_SEPARATOR_REGEX = r"\n\s*\d{1,2}[\.\)]\s"

# Model local 384 chiều, hỗ trợ tiếng Việt; E5 yêu cầu prefix query:/passage:.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Vector Store
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "bophapdien_docs_e5"
BATCH_SIZE = 256

# =============================================================================
# IMPLEMENTATION
# =============================================================================

_MODEL_CACHE = None


def get_embedding_model():
    """Lazy-load multilingual E5 model để tránh tải khi chỉ import module/test."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer

        print(f"⏳ Loading local embedding model '{EMBEDDING_MODEL}'...")
        _MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Local model loaded successfully (dim={EMBEDDING_DIM}).")
    return _MODEL_CACHE


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed text bằng multilingual E5 local, với prefix retrieval đúng chuẩn."""
    if not texts:
        return []
    prefix = "query: " if is_query else "passage: "
    embeddings = get_embedding_model().encode(
        [f"{prefix}{text}" for text in texts],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def get_collection():
    """Lấy hoặc tạo ChromaDB collection 'bophapdien_docs'."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_bophapdien_articles() -> list[dict]:
    """Đọc danh sách Điều luật từ bophapdien.json."""
    if not BOPHAPDIEN_JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {BOPHAPDIEN_JSON_PATH}")

    print(f"1. Đang đọc dữ liệu từ {BOPHAPDIEN_JSON_PATH.name}...")
    with open(BOPHAPDIEN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    print(f"✓ Đã nạp thành công {len(articles)} Điều luật từ Bộ Pháp Điển.")
    return articles


def load_documents() -> list[dict]:
    """Compatibility interface cho test Task 4.

    Trả về các Điều luật dưới dạng document có trường ``content``; các trường
    gốc vẫn được giữ để ``chunk_articles`` sử dụng trong pipeline thực tế.
    """
    if not BOPHAPDIEN_JSON_PATH.exists():
        return []

    documents = []
    for article in load_bophapdien_articles():
        document = dict(article)
        document["content"] = (article.get("noi_dung", "") or "").strip()
        documents.append(document)
    return documents


def _sanitize_metadata(metadata: dict) -> dict:
    """
    Chroma chỉ chấp nhận metadata kiểu str/int/float/bool (không chấp nhận None).
    Chuẩn hóa mọi giá trị None -> "" và ép các kiểu lạ về str để tránh lỗi upsert.
    """
    clean = {}
    for k, v in metadata.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _make_chunk_id(mapc: str, article_index: int, sub_chunk_index: int) -> str:
    """
    Sinh ID duy nhất và ổn định cho mỗi chunk.

    ``mapc`` không phải khóa duy nhất trong Bộ Pháp Điển, nên phải kèm vị trí
    Điều luật trong file nguồn. Nhờ vậy Chroma không nhận hai chunk khác nhau
    có cùng ID trong cùng một batch.
    """
    prefix = mapc or "chunk"
    return f"{prefix}_{article_index}_{sub_chunk_index}"


def chunk_articles(articles: list[dict]) -> list[dict]:
    """
    Chunking dữ liệu Bộ Pháp Điển theo chiến lược Legal Context Injection & Preservation.
    """
    print(f"2. Đang tiến hành Chunking {len(articles)} Điều luật...")

    chunks = []

    for article_index, article in enumerate(articles):
        chu_de = article.get("chu_de", "") or ""
        de_muc = article.get("de_muc", "") or ""
        phan_chuong_muc = article.get("phan_chuong_muc", "") or ""
        ten_dieu = article.get("ten_dieu", "") or ""
        ghi_chu = article.get("ghi_chu", "") or ""
        noi_dung = (article.get("noi_dung", "") or "").strip()

        if not noi_dung:
            continue

        # Header ngữ cảnh pháp lý chèn vào đầu mỗi Chunk
        chapter_context = f" | [{phan_chuong_muc}]" if phan_chuong_muc else ""
        note_context = f"[{ghi_chu}]\n" if ghi_chu else ""
        context_header = (
            f"[Chủ đề: {chu_de}] | [Đề mục: {de_muc}]{chapter_context}\n"
            f"[{ten_dieu}]\n"
            f"{note_context}\n"
        )

        base_metadata = _sanitize_metadata({
            "source_id": article.get("id", ""),
            "mapc": article.get("mapc", ""),
            "ten_dieu": ten_dieu,
            "chu_de": chu_de,
            "chu_de_id": article.get("chu_de_id", ""),
            "de_muc": de_muc,
            "de_muc_id": article.get("de_muc_id", ""),
            "link_vbpl": article.get("link_vbpl", ""),
        })

        mapc = base_metadata.get("mapc", "")

        # Trường hợp 1: Điều <= 1500 ký tự -> Bảo toàn 1 Chunk trọn vẹn
        if len(noi_dung) <= MAX_SINGLE_ARTICLE_LENGTH:
            full_text = context_header + noi_dung
            chunks.append({
                "id": _make_chunk_id(mapc, article_index, 0),
                "content": full_text,
                "metadata": {**base_metadata, "sub_chunk_index": 0, "is_split": False}
            })
        # Trường hợp 2: Điều > 1500 ký tự -> Tách theo Khoản con
        else:
            # Trừ hao độ dài header (+ 20 ký tự buffer) khỏi ngân sách chunk_size,
            # để (header + sub_chunk) không vượt quá CHUNK_SIZE dự kiến ban đầu.
            effective_chunk_size = max(CHUNK_SIZE - len(context_header) - 20, 300)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=effective_chunk_size,
                chunk_overlap=CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    CLAUSE_SEPARATOR_REGEX,  # tách theo khoản bất kỳ số nào (regex)
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )

            # Nếu splitter không hỗ trợ separator dạng regex (fallback tự viết ở trên),
            # dùng regex thủ công để tách khoản trước, rồi mới split_text từng khoản.
            try:
                splits = splitter.split_text(noi_dung)
            except Exception:
                # Fallback: tách thủ công theo regex khoản, rồi split từng đoạn quá dài
                parts = re.split(f"({CLAUSE_SEPARATOR_REGEX})", noi_dung)
                merged = []
                buf = ""
                for part in parts:
                    if re.fullmatch(CLAUSE_SEPARATOR_REGEX, part or ""):
                        buf += part
                    else:
                        buf += part or ""
                        merged.append(buf)
                        buf = ""
                if buf:
                    merged.append(buf)

                splits = []
                for seg in merged:
                    if len(seg) <= effective_chunk_size:
                        splits.append(seg)
                    else:
                        sub_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=effective_chunk_size,
                            chunk_overlap=CHUNK_OVERLAP,
                            separators=["\n", ". ", " ", ""],
                        )
                        splits.extend(sub_splitter.split_text(seg))

            for i, split_text in enumerate(splits):
                chunk_text = context_header + split_text
                chunks.append({
                    "id": _make_chunk_id(mapc, article_index, i),
                    "content": chunk_text,
                    "metadata": {**base_metadata, "sub_chunk_index": i, "is_split": True}
                })

    print(f"✓ Hoàn tất Chunking: Tạo tổng cộng {len(chunks)} Chunks từ {len(articles)} Điều luật.")
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Compatibility interface cho test Task 4 và pipeline generic."""
    return chunk_articles(documents)


def _load_progress() -> int:
    """Đọc offset đã index thành công lần trước (để resume nếu bị gián đoạn)."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("last_completed_index", 0)
        except Exception:
            return 0
    return 0


def _save_progress(index: int):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_completed_index": index}, f)


def index_chunks_to_chroma(chunks: list[dict], limit: int = None, resume: bool = True):
    """
    Embed và Index chunks vào ChromaDB theo batch, có hỗ trợ resume nếu bị gián đoạn giữa chừng.
    """
    if limit:
        chunks = chunks[:limit]
        print(f"⚠️ Đang chạy demo index với {limit} chunks...")

    total_chunks = len(chunks)
    start_offset = _load_progress() if resume else 0
    if start_offset >= total_chunks:
        start_offset = 0  # progress cũ không còn hợp lệ với tập dữ liệu hiện tại

    if start_offset > 0:
        print(f"↻ Tiếp tục Indexing từ chunk {start_offset}/{total_chunks} (resume từ lần chạy trước)...")

    print(f"3. Đang Embed & Index {total_chunks - start_offset} chunks còn lại vào ChromaDB (Batch Size = {BATCH_SIZE})...")

    collection = get_collection()

    t0 = time.time()

    for start_idx in range(start_offset, total_chunks, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)
        batch = chunks[start_idx:end_idx]

        try:
            texts = [c["content"] for c in batch]
            embeddings = embed_texts(texts)

            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            _save_progress(end_idx)

        except Exception as e:
            print(f"✗ Lỗi khi xử lý batch [{start_idx}:{end_idx}]: {e}")
            print(f"  Tiến độ đã lưu tới chunk {start_idx}. Chạy lại script để resume từ đây.")
            raise

        if (start_idx // BATCH_SIZE + 1) % 10 == 0 or end_idx == total_chunks:
            elapsed = time.time() - t0
            processed = end_idx - start_offset
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"   - Indexing: {end_idx}/{total_chunks} chunks ({end_idx/total_chunks*100:.1f}%) - Speed: {speed:.1f} chunks/s")

    print(f"✓ Hoàn tất Indexing vào Collection '{COLLECTION_NAME}' trong {time.time()-t0:.2f}s!")

    # Xóa file progress khi hoàn tất toàn bộ (chạy full, không phải demo limit)
    if not limit and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def run_pipeline(limit: int = None, resume: bool = True):
    """Hàm main thực thi toàn bộ pipeline."""
    print("=" * 60)
    print("Task 4: Legal Chunking & Indexing Pipeline (Bộ Pháp Điển)")
    print(f"  Method: {CHUNKING_METHOD}")
    print(f"  Preserve Single Article Length <= {MAX_SINGLE_ARTICLE_LENGTH}")
    print(f"  Embedding Model: {EMBEDDING_MODEL} (Dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} (Collection={COLLECTION_NAME})")
    print("=" * 60)

    articles = load_bophapdien_articles()
    chunks = chunk_articles(articles)
    index_chunks_to_chroma(chunks, limit=limit, resume=resume)


if __name__ == "__main__":
    # Đổi limit=None để index toàn bộ dataset 58k+ Điều luật
    run_pipeline()
