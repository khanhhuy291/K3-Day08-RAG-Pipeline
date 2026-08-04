# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học bằng kỹ thuật Retrieval-Augmented Generation (RAG), đồng thời xây dựng pipeline đánh giá chất lượng hệ thống.

---

# Yêu cầu 1: University Services RAG Chatbot

## Tính năng

- Chatbot hỏi đáp về chính sách và dịch vụ đại học.
- Giao diện Web bằng Streamlit.
- Trả lời có Citation.
- Hỗ trợ hội thoại nhiều lượt (Conversation Memory).
- Hiển thị các tài liệu nguồn được sử dụng.
- Hỗ trợ Hybrid Retrieval (BM25 + Dense Retrieval).
- Có Reranking nhằm cải thiện chất lượng truy xuất.

---

# Yêu cầu 2: Evaluation Pipeline

Framework sử dụng:

- **RAGAS**

Các metric đánh giá:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

Thực hiện:

- Xây dựng Golden Dataset gồm tối thiểu 15 câu hỏi.
- Đánh giá trên toàn bộ dataset.
- So sánh hai cấu hình:
  - Dense Retrieval
  - Hybrid Retrieval + Reranker
- Phân tích các trường hợp có điểm thấp và đề xuất cải thiện.

---

# Kiến Trúc Hệ Thống

```text
                        +-----------------------+
                        |    Streamlit UI       |
                        +-----------+-----------+
                                    |
                                    |
                          User Question
                                    |
                                    v
                     +----------------------------+
                     | Conversation Memory        |
                     +-------------+--------------+
                                   |
                                   v
                        +---------------------+
                        | Retrieval Pipeline  |
                        +---------------------+
                          |              |
                     Dense Search     BM25 Search
                          |              |
                           \            /
                            \          /
                             Hybrid Fusion
                                   |
                                   v
                            Cross Encoder
                              Reranker
                                   |
                                   v
                          Top-k Relevant Chunks
                                   |
                                   v
                        Large Language Model
                                   |
                                   v
                      Answer + Citation + Sources
```

---

# Công Nghệ Sử Dụng

- Streamlit
- LangChain
- HuggingFace
- BAAI/bge-m3
- Qdrant
- BM25
- Cross-Encoder Reranker
- RAGAS

---

# Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|------------|------|----------|------------|
| Đinh Xuân Huy | 2A202601894 | Thiết kế kiến trúc hệ thống, Dense Retrieval, Hybrid Search, Vector Database, tích hợp RAG Pipeline | ✅ |
| Nguyễn Bá Khánh Huy | 2A202601591 | Xây dựng Data Processing, Chunking, Embedding và quản lý dữ liệu | ✅ |
| Ngô Quang Dũng | 2A202601819 | Phát triển giao diện Streamlit và Conversation Memory | ✅ |
| Phạm Tuấn Việt | 2A202601987 | Citation, Source Display và tích hợp LLM Generation | ✅ |
| Phạm Tiến Anh | 2A202601549 | Xây dựng Evaluation Pipeline (RAGAS), Golden Dataset và báo cáo kết quả | ✅ |
| Đỗ Đức Trường | 2A202601499 | Kiểm thử hệ thống, so sánh A/B, tối ưu Retrieval và hỗ trợ hoàn thiện README | ✅ |

---

# Cấu Trúc Thư Mục

```text
group_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── embedding/
│
├── retrieval/
│
├── generation/
│
├── ui/
│
├── vectorstore/
│
├── evaluation/
│   ├── golden_dataset.json
│   ├── eval_pipeline.py
│   └── results.md
│
└── utils/
```

---

# Hướng Dẫn Chạy

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

---

# Deliverables

- ✅ Chatbot hoạt động hoàn chỉnh
- ✅ Citation và Source Documents
- ✅ Conversation Memory
- ✅ Golden Dataset (15+ câu hỏi)
- ✅ Evaluation Pipeline
- ✅ So sánh A/B giữa hai cấu hình Retrieval
- ✅ Báo cáo kết quả Evaluation
- ✅ README hoàn chỉnh

---

# Đóng Góp

Mỗi thành viên chịu trách nhiệm phát triển và kiểm thử phần được phân công, đồng thời phối hợp tích hợp thành hệ thống RAG hoàn chỉnh.
