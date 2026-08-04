"""
RAG Chatbot — University Services (Group Project)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG — phải đặt ĐẦU TIÊN trước mọi lệnh st.*
# =============================================================================

st.set_page_config(
    page_title="UniAssist — University RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Dark Mode + Glassmorphism + Premium UI
# =============================================================================

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base & Root ──────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 40%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit elements ─────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #111128 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar text */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(99, 102, 241, 0.25);
}

/* ── Main content area ───────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 5rem 2rem;
    max-width: 1100px;
}

/* ── Custom header banner ────────────────────────── */
.uni-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 50%, rgba(34,211,238,0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 1rem;
}
.uni-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.uni-header .sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}
.pipeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.7rem;
    color: #22d3ee;
    font-weight: 500;
    white-space: nowrap;
}

/* ── Chat message containers ─────────────────────── */
.chat-row {
    display: flex;
    margin-bottom: 1.25rem;
    gap: 0.75rem;
    animation: fadeInUp 0.3s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Avatar circles */
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 0.15rem;
}
.avatar-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 0 12px rgba(99,102,241,0.4);
}
.avatar-ai {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(34,211,238,0.4);
    box-shadow: 0 0 12px rgba(34,211,238,0.2);
}

/* Message bubbles */
.bubble {
    padding: 0.9rem 1.2rem;
    border-radius: 14px;
    max-width: 78%;
    line-height: 1.65;
    font-size: 0.91rem;
}
.bubble-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.35);
    color: #e2e8f0;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.bubble-ai {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(34,211,238,0.2);
    color: #cbd5e1;
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(8px);
}
.chat-row-user {
    flex-direction: row-reverse;
}

/* Citation highlight inside AI bubble */
.bubble-ai code {
    background: rgba(34,211,238,0.12);
    color: #22d3ee;
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    font-size: 0.8rem;
}
.citation-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    border-radius: 6px;
    padding: 0.05rem 0.45rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0 2px;
}

/* ── Source cards ─────────────────────────────────── */
.sources-section {
    margin-top: 0.75rem;
    border-top: 1px solid rgba(99,102,241,0.15);
    padding-top: 0.7rem;
}
.source-card {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.source-card:hover {
    border-color: rgba(99,102,241,0.45);
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.source-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
}
.source-type {
    font-size: 0.68rem;
    color: #64748b;
    background: rgba(99,102,241,0.1);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
}
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.score-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.score-num {
    font-size: 0.7rem;
    color: #22d3ee;
    font-weight: 600;
    min-width: 34px;
    text-align: right;
}
.source-snippet {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.5;
    margin-top: 0.3rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Suggestion buttons ─────────────────────────── */
.stButton > button {
    background: rgba(15,23,42,0.7) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    text-align: left !important;
    transition: all 0.2s !important;
    padding: 0.5rem 0.9rem !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.6) !important;
    color: #c7d2fe !important;
    transform: translateX(2px);
}

/* Clear button */
.clear-btn > button {
    background: rgba(239,68,68,0.08) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #fca5a5 !important;
    font-size: 0.75rem !important;
}
.clear-btn > button:hover {
    background: rgba(239,68,68,0.18) !important;
}

/* ── Slider ───────────────────────────────────────── */
[data-testid="stSlider"] {
    padding: 0.25rem 0;
}
[data-testid="stSlider"] .stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #22d3ee) !important;
}

/* ── Chat input ───────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.85) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInputTextArea"] {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

/* ── Spinner / loading ────────────────────────────── */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.4) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* ── Info/warning boxes ───────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.status-hybrid {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
}
.status-pageindex {
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.3);
    color: #22d3ee;
}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

/* ── Markdown text in chat ────────────────────────── */
.bubble-ai p, .bubble-ai li { color: #cbd5e1; }
.bubble-ai strong { color: #e2e8f0; }
.bubble-ai h1, .bubble-ai h2, .bubble-ai h3 { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# =============================================================================
# HELPER: Render một message
# =============================================================================

def render_message(role: str, content: str, sources: list = None):
    """Render một chat bubble hoàn chỉnh với sources (nếu có)."""
    is_user = (role == "user")
    row_class  = "chat-row-user" if is_user else ""
    avatar_cls = "avatar-user" if is_user else "avatar-ai"
    bubble_cls = "bubble-user" if is_user else "bubble-ai"
    icon       = "👤" if is_user else "🎓"

    avatar_html = f'<div class="avatar {avatar_cls}">{icon}</div>'

    # Build source cards HTML (chỉ cho AI)
    sources_html = ""
    if not is_user and sources:
        cards_html = ""
        for i, src in enumerate(sources, 1):
            meta  = src.get("metadata", {})
            name  = meta.get("source", meta.get("ten_dieu", f"Nguồn {i}"))[:48]
            dtype = meta.get("type", src.get("source", "document"))
            score = float(src.get("score", 0))
            bar_w = min(100, int(score * 100))
            snip  = src.get("content", "")[:140].replace("<", "&lt;").replace(">", "&gt;")
            badge_color = "#22d3ee" if dtype == "pageindex" else "#a5b4fc"
            cards_html += f"""
            <div class="source-card">
              <div class="source-header">
                <span class="source-name">[{i}] {name}</span>
                <span class="source-type" style="color:{badge_color}">{dtype}</span>
              </div>
              <div class="score-bar-wrap">
                <div class="score-bar-bg">
                  <div class="score-bar-fill" style="width:{bar_w}%"></div>
                </div>
                <span class="score-num">{score:.3f}</span>
              </div>
              <div class="source-snippet">{snip}…</div>
            </div>"""

        sources_html = f"""
        <div class="sources-section">
          <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;font-weight:600;">
            📚 {len(sources)} nguồn tham khảo
          </div>
          {cards_html}
        </div>"""

    # Escape & render content (markdown-like within HTML)
    safe_content = content.replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(f"""
    <div class="chat-row {row_class}">
      {'' if is_user else avatar_html}
      <div class="bubble {bubble_cls}">
        {safe_content}
        {sources_html}
      </div>
      {avatar_html if is_user else ''}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # Logo & branding
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.2rem 0;">
      <div style="font-size:2.8rem;line-height:1;">🎓</div>
      <div style="font-size:1.1rem;font-weight:700;
                  background:linear-gradient(90deg,#a5b4fc,#22d3ee);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  margin-top:0.4rem;">UniAssist</div>
      <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem;">
        RAG Chatbot · University Services
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
                    border-radius:10px;padding:0.6rem;text-align:center;">
          <div style="font-size:1.4rem;font-weight:700;color:#a5b4fc;">
            {st.session_state.total_queries}
          </div>
          <div style="font-size:0.65rem;color:#64748b;">Câu hỏi</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.25);
                    border-radius:10px;padding:0.6rem;text-align:center;">
          <div style="font-size:1.4rem;font-weight:700;color:#22d3ee;">
            {len(st.session_state.messages) // 2}
          </div>
          <div style="font-size:0.65rem;color:#64748b;">Hội thoại</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Settings
    st.markdown('<p style="font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:0.5rem;">⚙️ CÀI ĐẶT</p>', unsafe_allow_html=True)
    top_k = st.slider("Số chunks retrieval (top_k)", min_value=2, max_value=10, value=5, step=1)
    score_threshold = st.slider("Ngưỡng fallback (threshold)", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

    st.divider()

    # Suggestions
    st.markdown('<p style="font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:0.6rem;">💡 GỢI Ý CÂU HỎI</p>', unsafe_allow_html=True)
    suggestions = [
        "💰 Học phí tại RMIT Vietnam là bao nhiêu?",
        "📚 Làm sao đặt phòng học nhóm ở thư viện?",
        "🏆 Điều kiện xin học bổng Achievement?",
        "🏠 Dịch vụ hỗ trợ chỗ ở cho sinh viên?",
        "📝 Cách đăng ký học phần qua myRMIT?",
        "🏋️ Thông tin khu thể thao RMIT?",
        "💼 RMIT Career Fair 2026 khi nào?",
        "🏥 Dịch vụ y tế và tư vấn tâm lý?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:15]}"):
            st.session_state.pending_query = s.split(" ", 1)[1]  # bỏ emoji prefix

    st.divider()

    # Pipeline info
    st.markdown("""
    <p style="font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:0.5rem;">🔧 PIPELINE</p>
    <div style="font-size:0.72rem;color:#64748b;line-height:1.9;">
      <span style="color:#a5b4fc;">①</span> Semantic Search (BAAI/bge-m3)<br>
      <span style="color:#a5b4fc;">②</span> BM25 Lexical Search<br>
      <span style="color:#a5b4fc;">③</span> RRF Reranking (k=60)<br>
      <span style="color:#22d3ee;">④</span> PageIndex Fallback<br>
      <span style="color:#a5b4fc;">⑤</span> LLM Generation + Citation
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Clear history
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# MAIN CHAT AREA — Header
# =============================================================================

st.markdown("""
<div class="uni-header">
  <div style="font-size:2.2rem;">🎓</div>
  <div style="flex:1;">
    <h1>UniAssist — University RAG Chatbot</h1>
    <div class="sub">Hỏi đáp thông minh về học phí · Học bổng · Thư viện · Ký túc xá · Đăng ký học phần</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:flex-end;">
    <span class="pipeline-badge">⚡ Hybrid Retrieval</span>
    <span class="pipeline-badge">🔀 RRF Reranking</span>
    <span class="pipeline-badge">🤖 LLM + Citation</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

# Welcome message nếu chưa có hội thoại
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;opacity:0.7;">
      <div style="font-size:3.5rem;margin-bottom:1rem;">💬</div>
      <div style="font-size:1.05rem;font-weight:600;color:#94a3b8;margin-bottom:0.5rem;">
        Bắt đầu hội thoại!
      </div>
      <div style="font-size:0.82rem;color:#475569;max-width:420px;margin:0 auto;line-height:1.7;">
        Hỏi bất kỳ câu hỏi nào về dịch vụ và chính sách đại học.<br>
        Hệ thống sẽ tìm kiếm trong cơ sở dữ liệu và trả lời có trích dẫn nguồn.
      </div>
    </div>
    """, unsafe_allow_html=True)

# Render lịch sử
for msg in st.session_state.messages:
    render_message(
        role=msg["role"],
        content=msg["content"],
        sources=msg.get("sources", []) if msg["role"] == "assistant" else None,
    )


# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về dịch vụ/chính sách đại học...")
query = user_input or st.session_state.pending_query

if query:
    # Reset pending
    st.session_state.pending_query = None
    st.session_state.total_queries += 1

    # Thêm message user vào history
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # Gọi RAG pipeline
    with st.spinner("🔍 Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
        try:
            from src.task10_generation import generate_with_citation
            response = generate_with_citation(query, top_k=top_k)
            answer  = response.get("answer", "Không thể tạo câu trả lời.")
            sources = response.get("sources", [])
            ret_src = response.get("retrieval_source", "hybrid")

        except NotImplementedError:
            answer  = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py`!"
            sources = []
            ret_src = "none"
        except Exception as e:
            answer  = f"❌ **Lỗi khi chạy RAG Pipeline:**\n```\n{e}\n```\nKiểm tra lại `.env` và các module trong `src/`."
            sources = []
            ret_src = "none"

    # Thêm response vào history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": ret_src,
    })

    # Render AI response
    render_message("assistant", answer, sources)

    # Retrieval source badge
    if ret_src in ("hybrid", "pageindex"):
        badge_class = "status-hybrid" if ret_src == "hybrid" else "status-pageindex"
        badge_icon  = "⚡ Hybrid Search" if ret_src == "hybrid" else "🔗 PageIndex Fallback"
        st.markdown(f"""
        <div style="text-align:right;margin-top:-0.5rem;margin-bottom:0.5rem;">
          <span class="status-badge {badge_class}">{badge_icon}</span>
        </div>
        """, unsafe_allow_html=True)

    st.rerun()
