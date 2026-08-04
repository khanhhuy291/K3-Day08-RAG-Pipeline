"""Task 5 — Dense semantic search with optional HyDE query expansion."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import embed_texts, get_collection


load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")

# HyDE improves recall for short natural-language questions: the LLM first writes
# a plausible answer passage, which is then embedded in place of the raw query.
HYDE_ENABLED = True
HYDE_TEMPERATURE = 0.2
HYDE_MAX_TOKENS = 400
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
HYDE_MODEL = os.getenv("HYDE_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-4o-mini"

HYDE_SYSTEM_PROMPT = """Bạn hỗ trợ truy xuất văn bản pháp luật Việt Nam.
Hãy viết một đoạn tài liệu giả định, ngắn và giàu từ khóa, có thể trả lời câu hỏi.
Không khẳng định đây là thông tin đúng; không thêm lời chào, lưu ý, hay trích dẫn.
Ưu tiên nêu điều kiện, đối tượng, thủ tục và thuật ngữ pháp lý có thể xuất hiện
trong văn bản nguồn. Trả lời bằng tiếng Việt."""


def _generate_hypothetical_document(query: str) -> str | None:
    """Generate a HyDE passage, or return ``None`` when the LLM is unavailable."""
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Câu hỏi: {query}"},
            ],
            temperature=HYDE_TEMPERATURE,
            max_tokens=HYDE_MAX_TOKENS,
        )
        passage = (response.choices[0].message.content or "").strip()
        return passage or None
    except Exception as error:
        # Retrieval must remain usable offline and must not fail because HyDE fails.
        print(f"  ⚠ HyDE unavailable; searching with the original query: {error}")
        return None


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = HYDE_ENABLED) -> list[dict]:
    """Search ChromaDB using local multilingual E5 embeddings and cosine similarity.

    When ``use_hyde`` is true, embed a hypothetical answer generated from the
    query. The original query is used automatically if no API key is configured
    or the LLM request fails.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    collection_count = collection.count()
    if collection_count == 0:
        return []

    search_text = query.strip()
    if use_hyde:
        hypothetical_doc = _generate_hypothetical_document(search_text)
        if hypothetical_doc:
            search_text = hypothetical_doc

    query_embedding = embed_texts([search_text], is_query=True)[0]
    result_count = min(top_k, collection_count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for document, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        # Chroma returns cosine distance; convert it to a [0, 1] similarity.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append({
            "content": document,
            "score": round(score, 4),
            "metadata": metadata or {},
        })

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """Convenience entry point for explicitly running semantic search with HyDE."""
    return semantic_search(query, top_k=top_k, use_hyde=True)


if __name__ == "__main__":
    for result in hyde_search("Điều kiện để được miễn hoặc giảm học phí là gì?", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:160]}...")
