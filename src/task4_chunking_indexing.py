"""
Task 4 — Chunking & Indexing Dữ Liệu Bộ Pháp Điển Vào Vector Store.

Chiến lược Chunking (Legal RAG Context-Preserving Strategy):
    1. Context Prefix Injection: Chèn thông tin Chủ đề, Đề mục, Cấu trúc Chương/Mục và Tên Điều vào đầu mỗi Chunk.
    2. Article Preservation (Bảo toàn Điều luật):
       - Các Điều luật có độ dài <= 1500 ký tự sẽ được GIỮ NGUYÊN 1 Chunk trọn vẹn.
    3. Clause-Level Splitting (Tách theo Khoản/Mục cho Điều dài):
       - Các Điều luật > 1500 ký tự sẽ được tách theo Khoản (1., 2., 3...) sử dụng RecursiveCharacterTextSplitter
         với chunk_size=1000, chunk_overlap=150.
    4. Metadata Enrichment: Đính kèm các thông tin chu_de_id, de_muc_id, mapc, link_vbpl vào metadata để hỗ trợ Metadata Filtering.

Embedding Model:
    - BAAI/bge-m3 (1024 dim) — SOTA Multilingual Model mạnh nhất cho Tiếng Việt và Văn Bản Pháp Luật.

Vector Store:
    - ChromaDB persistent store (collection: 'bophapdien_docs').
"""

import json
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

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_SINGLE_ARTICLE_LENGTH = 1500  # Điều <= 1500 ký tự sẽ giữ nguyên 1 chunk
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "legal_context_injection"

# Embedding Model: BAAI/bge-m3 (1024 dims)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector Store
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "bophapdien_docs"
BATCH_SIZE = 256  # Kích thước batch khi embed & upsert

# =============================================================================
# IMPLEMENTATION
# =============================================================================

_MODEL_CACHE = None


def get_embedding_model():
    """Lazy load embedding model BAAI/bge-m3."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer
        print(f"⏳ Loading Embedding Model '{EMBEDDING_MODEL}'...")
        _MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Model '{EMBEDDING_MODEL}' loaded successfully (dim={EMBEDDING_DIM}).")
    return _MODEL_CACHE


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


def chunk_articles(articles: list[dict]) -> list[dict]:
    """
    Chunking dữ liệu Bộ Pháp Điển theo chiến lược Legal Context Injection & Preservation.
    """
    print(f"2. Đang tiến hành Chunking {len(articles)} Điều luật...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ", "\n", ". ", " "]
    )

    chunks = []

    for article in articles:
        chu_de = article.get("chu_de", "")
        de_muc = article.get("de_muc", "")
        phan_chuong_muc = article.get("phan_chuong_muc", "")
        ten_dieu = article.get("ten_dieu", "")
        ghi_chu = article.get("ghi_chu", "")
        noi_dung = article.get("noi_dung", "").strip()

        if not noi_dung:
            continue

        # Header ngữ cảnh pháp lý chèn vào đầu mỗi Chunk
        context_header = (
            f"[Chủ đề: {chu_de}] | [Đề mục: {de_muc}]"
            f"{' | [' + phan_chuong_muc + ']' if phan_chuong_muc else ''}\n"
            f"[{ten_dieu}]\n"
            f"{'[' + ghi_chu + ']\n' if ghi_chu else ''}\n"
        )

        base_metadata = {
            "source_id": article.get("id", ""),
            "mapc": article.get("mapc", ""),
            "ten_dieu": ten_dieu,
            "chu_de": chu_de,
            "chu_de_id": article.get("chu_de_id", ""),
            "de_muc": de_muc,
            "de_muc_id": article.get("de_muc_id", ""),
            "link_vbpl": article.get("link_vbpl", ""),
        }

        # Trường hợp 1: Điều <= 1500 ký tự -> Bảo toàn 1 Chunk trọn vẹn
        if len(noi_dung) <= MAX_SINGLE_ARTICLE_LENGTH:
            full_text = context_header + noi_dung
            chunks.append({
                "content": full_text,
                "metadata": {**base_metadata, "sub_chunk_index": 0, "is_split": False}
            })
        # Trường hợp 2: Điều > 1500 ký tự -> Tách theo Khoản con
        else:
            splits = splitter.split_text(noi_dung)
            for i, split_text in enumerate(splits):
                chunk_text = context_header + split_text
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**base_metadata, "sub_chunk_index": i, "is_split": True}
                })

    print(f"✓ Hoàn tất Chunking: Tạo tổng cộng {len(chunks)} Chunks từ {len(articles)} Điều luật.")
    return chunks


def index_chunks_to_chroma(chunks: list[dict], limit: int = None):
    """
    Embed và Index chunks vào ChromaDB theo batch.
    """
    if limit:
        chunks = chunks[:limit]
        print(f"⚠️ Đang chạy demo index với {limit} chunks...")

    total_chunks = len(chunks)
    print(f"3. Đang Embed & Index {total_chunks} chunks vào ChromaDB (Batch Size = {BATCH_SIZE})...")

    model = get_embedding_model()
    collection = get_collection()

    t0 = time.time()

    for start_idx in range(0, total_chunks, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)
        batch = chunks[start_idx:end_idx]

        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        ids = [
            f"{c['metadata']['mapc']}_{c['metadata']['sub_chunk_index']}"
            if c['metadata']['mapc']
            else f"chunk_{start_idx + i}"
            for i, c in enumerate(batch)
        ]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (start_idx // BATCH_SIZE + 1) % 10 == 0 or end_idx == total_chunks:
            elapsed = time.time() - t0
            speed = end_idx / elapsed if elapsed > 0 else 0
            print(f"   - Indexing: {end_idx}/{total_chunks} chunks ({end_idx/total_chunks*100:.1f}%) - Speed: {speed:.1f} chunks/s")

    print(f"✓ Hoàn tất Indexing vào Collection '{COLLECTION_NAME}' trong {time.time()-t0:.2f}s!")


def run_pipeline(limit: int = None):
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
    index_chunks_to_chroma(chunks, limit=limit)


if __name__ == "__main__":
    # Đổi limit=None để index toàn bộ dataset 58k+ Điều luật
    run_pipeline()
