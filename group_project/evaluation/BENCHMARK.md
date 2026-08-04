# Benchmark RAG — Trợ lý pháp lý (Bộ pháp điển điện tử)

Bộ 28 câu hỏi đánh giá pipeline RAG theo tiêu chí RAGAS, xây dựng trên corpus
`data/BoPhapDienDienTu/`.

| File | Nội dung |
|---|---|
| `golden_dataset.json` | 28 câu hỏi + ground truth |
| `benchmark_utils.py` | Loader, validator, adapter RAGAS/DeepEval, retrieval metrics |
| `test_benchmark_utils.py` | 15 test, không cần API key |
| `tools/parse_corpus.py` | Parse HTML đề mục → điều luật có cấu trúc |
| `tools/refresh_contexts.py` | Sinh lại `reference_contexts` từ corpus |
| `tools/audit_corpus.py` | **Đối chiếu `bophapdien.json` với HTML gốc** |

Mọi `expected_answer` và `reference_contexts` đều **trích nguyên văn từ corpus**
(sinh bằng script parse HTML, không gõ tay), và cả 21 mã điều được trích dẫn đã được
kiểm chứng là tồn tại thật trong `data/BoPhapDienDienTu/demuc/`.

```bash
python group_project/evaluation/benchmark_utils.py             # validate + thống kê
python -m pytest group_project/evaluation/test_benchmark_utils.py -q
python group_project/evaluation/tools/refresh_contexts.py --check   # đối chiếu corpus
```

### Thêm câu hỏi mới

Không gõ tay `reference_contexts`. Thêm entry vào `golden_dataset.json` với
`ground_truth_dieu` đã điền, để `reference_contexts: []`, rồi chạy:

```bash
python group_project/evaluation/tools/refresh_contexts.py
```

Script sẽ đọc corpus, điền nguyên văn nội dung điều luật, `source_documents`, `de_muc`,
`de_muc_id`, và **báo lỗi nếu mã điều không tồn tại**. Tra mã điều bằng:

```bash
python group_project/evaluation/tools/parse_corpus.py <de_muc_id> "từ khóa"
```

---

## 1. Phạm vi corpus

Benchmark bám vào **8 đề mục** (3.115 điều):

| Đề mục | Mã trong mã điều | Số điều | `de_muc_id` |
|---|---|---|---|
| Lao động | 20.2 | 477 | `2efd8c6f-509f-4207-84b6-6b22ff780f2a` |
| Bảo hiểm y tế | 2.2 | 427 | `583b913f-4d4b-46da-911c-112342c6ea49` |
| Hôn nhân và gia đình | 8.4 | 213 | `4913a1cf-5f78-471c-a807-ed5f8c57aaee` |
| Cư trú | 39.3 | 105 | `6a501e4a-ba3f-40e2-902f-af4a52b14bb4` |
| An ninh mạng | 1.11 | 87 | `1fd42d83-9d78-4dd4-b6b6-73e9bd3472e1` |
| Doanh nghiệp | 12.1 | 930 | `319387a3-090a-47a5-82f6-a9bacbe5d341` |
| Thuế thu nhập cá nhân | 33.1 ⚠ | 136 | `ab116df3-d3ed-4a95-95bf-0d6b8962e724` |
| Dân sự | 9.1 | 740 | `eb0e4753-243e-4344-90e6-70aaf5188a6d` |

> ⚠ **Bẫy đánh số ở Thuế TNCN:** tiêu đề trang ghi *"Đề mục 33.10"* (đề mục thứ 10 của
> chủ đề 33) nhưng **mã điều trong nội dung lại là `Điều 33.1.LQ.19`** — cả 136 điều đều
> vậy. Hai hệ đánh số này khác nhau ở những đề mục có STT ≥ 10. `ground_truth_dieu` dùng
> **mã điều trong nội dung** (dạng xuất hiện thật trong chunk). Nếu Task 3–4 sinh
> `dieu_id` từ tiêu đề trang thì sẽ lệch và mọi câu Thuế TNCN bị chấm 0.

> **Yêu cầu bắt buộc:** 8 đề mục trên phải nằm trong index thì điểm mới có nghĩa.
> Index rộng hơn thì vẫn chạy được (khó hơn — nhiều nhiễu hơn), index hẹp hơn thì
> điểm không so sánh được.

### ⚠⚠ `data/bophapdien.json` THIẾU 11,8% corpus — kiểm tra trước khi index

```
python group_project/evaluation/tools/audit_corpus.py
```

| Nguồn | Số điều |
|---|---|
| HTML gốc `demuc/*.html` | **65.997** |
| `bophapdien.json` (tạo 2026-08-04 09:57) | 58.212 |
| **Thiếu** | **7.785 (11,8%)** ở **14 đề mục** |

| Đề mục | JSON | HTML | Thiếu |
|---|---:|---:|---:|
| Xử lý vi phạm hành chính | 557 | 4.322 | **3.765** |
| Hàng hải Việt Nam | 702 | 1.677 | 975 |
| An toàn thực phẩm | 98 | 927 | 829 |
| Thống kê | 110 | 596 | 486 |
| Khí tượng thủy văn | 108 | 506 | 398 |
| Bảo vệ và kiểm dịch thực vật | 26 | 331 | 305 |
| **Lao động** | 208 | 477 | **269** |
| An toàn thông tin mạng | 64 | 293 | 229 |
| Giám định tư pháp | 252 | 443 | 191 |
| Quản lý ngoại thương | 1.134 | 1.296 | 162 |
| Dân số | 26 | 104 | 78 |
| Cảnh sát biển Việt Nam | 128 | 163 | 35 |
| Tiếp nhận, xử lý phản ánh, kiến nghị | 4 | 39 | 35 |
| Một số hoạt động kinh doanh đặc thù | 1.029 | 1.057 | 28 |

Đặc điểm: mỗi đề mục bị cắt **sạch từ một điểm trở đi**, không có lỗ hổng ở giữa —
Lao động có đủ `Điều 20.2.LQ.1` … `LQ.98` rồi dừng hẳn (Bộ luật Lao động 2019 có 220
điều). HTML tại điểm cắt hoàn toàn bình thường, nên lỗi nằm ở **quá trình convert bị
dừng giữa chừng**, không phải do dữ liệu nguồn hỏng.

`bophapdien.json` được sinh bằng cách **crawl website** chứ không phải parse HTML local
— trường `id` (UUID) không tồn tại trong HTML, còn `mapc` và `link_vbpl` thì có. Vì vậy
**không thể tái tạo file này từ HTML**; phải chạy lại converter.

**Ảnh hưởng trực tiếp lên benchmark:** thiếu `Điều 20.2.LQ.113` (nghỉ hằng năm) và
`Điều 20.2.LQ.139` (nghỉ thai sản) → **Q07, Q08, Q10, Q11 sẽ trượt vì LỖI DỮ LIỆU, không
phải vì pipeline RAG kém**. Cho đến khi converter chạy lại: **index từ HTML**, hoặc
chấp nhận 4 câu này không có ý nghĩa.

Về nội dung thì hai nguồn **đồng nhất**: đối chiếu 1.448 điều có ở cả hai bên → 1.410
trùng khớp tuyệt đối, 38 khác biệt đều vô hại (JSON sạch hơn: bỏ tên file phụ lục đính
kèm và khoảng trắng thừa). Vấn đề duy nhất là **thiếu điều**, không phải sai nội dung.

### ⚠ 118/320 đề mục trong corpus là file RỖNG

Kiểm tra thực tế trên `data/BoPhapDienDienTu/demuc/`:

```
tổng số file      320
rỗng (<2KB)       118   ← chỉ có <div class='_content'></div>
có nội dung       202
```

Trong số rỗng có nhiều đề mục quan trọng: **Đất đai, Căn cước, Bảo hiểm xã hội,
Xây dựng, Thuế tiêu thụ đặc biệt, Trật tự an toàn giao thông đường bộ**.

Hệ quả cho cả nhóm:

1. `hierarchy.json` liệt kê 320 đề mục nhưng **chỉ 202 có nội dung** — đừng dựa vào
   hierarchy để khẳng định một chủ đề "có trong dữ liệu".
2. Người dùng hỏi về Đất đai là câu hỏi hợp lệ nhưng hệ thống **không có evidence** →
   bắt buộc phải từ chối. Đây chính là câu **Q25**.
3. Nếu bản tải về được cho là đầy đủ thì cần kiểm tra lại khâu tải; nếu là do
   phía Bộ pháp điển chưa pháp điển hóa thì cần ghi rõ giới hạn này trong báo cáo.

---

## 2. Ma trận năng lực

28 câu phủ **15 trục năng lực**. Cột cuối trả lời câu hỏi quan trọng nhất:
*câu này fail thì sửa ở đâu?*

| Trục | Câu | Đo cái gì | Fail → sửa Task |
|---|---|---|---|
| `simple_factual` | Q01, Q07, Q15, Q20, Q21, Q23 | Tra cứu 1 điều, baseline | 4 (index/embedding) |
| `numeric_precision` | Q02, Q04, Q12 | Giữ đúng con số, tỷ lệ, ngoại lệ | 4 (chunk cắt số), 10 (prompt) |
| `enumeration` | Q03, Q08, Q19 | Lấy **trọn** danh sách dài | 4 (chunk_size), 10 (top_k) |
| `lexical_exact` | Q13, Q14 | Tra theo số hiệu văn bản / mã điều | 6 (BM25), 4 (metadata) |
| `paraphrase_colloquial` | Q09, Q10 | Ngôn ngữ đời thường → thuật ngữ luật | 5 (embedding đa ngữ) |
| `multi_hop` | Q06, Q22 | Cần ≥2 điều mới trả lời đủ | 9 (top_k), 7 (rerank) |
| `cross_domain` | Q11 | Evidence nằm ở 2 đề mục khác nhau | 9 (hybrid fusion) |
| `comparative` | Q05, Q24 | Đối chiếu, phân biệt near-duplicate | 7 (rerank/MMR) |
| `temporal_amendment` | Q17 | Nhận ra điều khoản **đã bị sửa đổi** | 10 (prompt trích dẫn) |
| `stale_law_trap` | Q18 | Nhận ra văn bản **đã hết hiệu lực** | 10, 3 (giữ ngày hiệu lực) |
| `false_premise` | Q16 | Bác bỏ tiền đề sai, không nịnh người dùng | 10 (system prompt) |
| `ambiguous` | Q27 | Hỏi lại thay vì đoán bừa | 10 |
| `noise_robustness` | Q28 | Query rác → fallback | **9 (threshold)** |
| `unanswerable_out_of_scope` | Q26 | Ngoài miền → từ chối | 9, 10 |
| `unanswerable_empty_demuc` | Q25 | Đúng chủ đề nhưng corpus rỗng → từ chối | 9, 10 |

Phân bố: easy 9 / medium 11 / hard 8 — answerable 24 / unanswerable 4.

---

## 3. Bốn câu bẫy quan trọng nhất

Đây là phần khiến bộ benchmark này khác một bộ hỏi-đáp thông thường.

### Q28 — query rác, bắt đúng con bug đã được cảnh báo trong repo

Docstring `src/task9_retrieval_pipeline.py` cảnh báo: điểm RRF luôn ≈ `1/(k+1)` ≈
0.0164 **bất kể liên quan hay không**. Nếu lấy điểm RRF đem so `SCORE_THRESHOLD`
thì fallback **không bao giờ** kích hoạt.

Q28 (`"asdkjh123 zxcvbnm qwerty ???"`) là phép thử trực tiếp: nếu nó trả về một chunk
pháp luật bất kỳ thay vì từ chối → đang so sai thang điểm, phải chuyển sang dùng
**điểm cosine gốc của `semantic_search`**.

Dùng Q26 + Q28 (chắc chắn lạc đề) đối chiếu với Q01 + Q07 (chắc chắn liên quan) để
**calibrate `SCORE_THRESHOLD`**: đo cosine của hai nhóm rồi chọn ngưỡng nằm giữa.

### Q25 — chủ đề có thật, dữ liệu rỗng

Hỏi hạn mức giao đất nông nghiệp. `hierarchy.json` **có** đề mục "Đất đai" nên câu
hỏi trông hoàn toàn hợp lệ, nhưng file đề mục rỗng. Bịa ra con số = hallucination
nặng nhất trong ngữ cảnh pháp lý.

### Q17 — con số trong corpus đã bị sửa đổi

Corpus ghi giảm trừ gia cảnh **9 triệu đồng/tháng**, và ngay trong text có ghi chú
*"Khoản 1 Điều này được sửa đổi bổ sung theo khoản 4 Điều 6 Luật số 71/2014/QH13"*.
Mức áp dụng thực tế hiện nay là 11 triệu (Nghị quyết 954/2020/UBTVQH14) — **không có
trong corpus**.

- Đạt: nêu con số theo corpus **và** cảnh báo điều khoản đã bị sửa đổi.
- Trượt: khẳng định 9 triệu là mức hiện hành.

### Q18 — văn bản đã hết hiệu lực vẫn nằm trong corpus

Đề mục Cư trú trong corpus là **Luật Cư trú 2006** (81/2006/QH11), nói về sổ hộ khẩu.
Luật này đã được thay bằng Luật Cư trú 2020 (68/2020/QH14), sổ hộ khẩu giấy bỏ từ
01/01/2023.

- Đạt: trả lời theo corpus nhưng nêu rõ nguồn + ngày hiệu lực + khuyến cáo đối chiếu.
- Trượt: trình bày như luật hiện hành.

> Q17 và Q18 cho thấy với trợ lý pháp lý, **trích dẫn kèm ngày hiệu lực quan trọng
> ngang với nội dung**. Nên đưa `ngay_hieu_luc` và `so_hieu_van_ban` vào metadata từ
> Task 3–4 để LLM có cái mà trích.

---

## 4. Schema

```jsonc
{
  "id": "Q11",
  "question": "...",
  "expected_answer": "...",           // reference / ground_truth cho RAGAS
  "expected_context": "...",          // mô tả cho người đọc
  "reference_contexts": ["..."],      // trích NGUYÊN VĂN corpus → context_recall
  "ground_truth_dieu": ["Điều 2.2.LQ.13", "Điều 20.2.LQ.139"],  // chấm retrieval, 0 LLM call
  "source_documents": ["Luật số 25/2008/QH12"],
  "de_muc": ["Bảo hiểm y tế", "Lao động"],
  "de_muc_id": ["583b913f-...", "2efd8c6f-..."],
  "capability": "cross_domain",
  "difficulty": "hard",
  "answerable": true,
  "eval_mode": "ragas",               // "ragas" = chấm tự động | "manual" = chấm tay
  "notes": "Bằng chứng nằm ở HAI đề mục khác nhau..."
}
```

Ba field đầu **giữ nguyên tên cũ** để không phá `eval_pipeline.py` của nhóm — phần
còn lại là bổ sung.

`eval_mode: "manual"` (Q25–Q28) bị **loại khỏi RAGAS** vì RAGAS không đo được "từ chối
đúng cách" — `faithfulness` của một câu từ chối là vô nghĩa. Chấm 4 câu này bằng
`refusal_rate()`.

---

## 5. Cách chạy

### Bước 1 — Retrieval metrics trước (miễn phí, 0 LLM call)

```python
from benchmark_utils import load_golden_dataset, retrieval_scores, by_capability
from src.task9_retrieval_pipeline import retrieve

ds = load_golden_dataset()
retrieved = {d["id"]: retrieve(d["question"], top_k=5) for d in ds if d["answerable"]}

r = retrieval_scores(ds, retrieved, k=5)
print(r["hit_rate"], r["recall"], r["mrr"])
print(by_capability(ds, r["per_question"]))   # năng lực nào đang yếu
```

`retrieval_scores` chấm bằng cách so khớp mã điều trong `ground_truth_dieu` — **không
gọi LLM**, chạy bao nhiêu lần cũng được. Dùng nó để quét A/B (alpha, top_k, rerank
on/off, chunk_size) thoải mái, rồi mới chạy RAGAS **một lần** trên 2 config tốt nhất.

Lý do: RAGAS/DeepEval gọi LLM nhiều lần **cho mỗi metric, cho mỗi câu hỏi**. Model
`:free` của OpenRouter giới hạn **50 request/ngày cho cả tài khoản** — đổi model hay
tạo key mới đều không reset. 28 câu × 4 metric sẽ cháy quota giữa chừng.

> Để `retrieval_scores` chấm chính xác, **đưa mã điều vào chunk**: hoặc giữ dòng
> `"Điều 20.2.LQ.25. Thời gian thử việc"` trong text, hoặc set
> `metadata["dieu_id"]` ở Task 4. Không có mã điều thì mọi câu đều bị chấm 0.

### Bước 2 — RAGAS trên config đã chọn

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from benchmark_utils import to_ragas

runs = {d["id"]: {"answer": ..., "contexts": [...]} for d in ds}   # từ generate_with_citation
data = Dataset.from_dict(to_ragas(ds, runs, schema="legacy"))      # "modern" nếu ragas>=0.2
result = evaluate(data, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
```

Hết quota giữa chừng thì chạy subset 5 câu — nên chọn **Q01, Q09, Q11, Q17, Q19**
(mỗi câu một trục khác nhau: baseline, ngữ nghĩa, cross-domain, thời hiệu, danh sách dài).

### Bước 3 — Câu unanswerable

```python
from benchmark_utils import refusal_rate
print(refusal_rate(ds, runs))
```

- `correct_refusal_rate` — Q25–Q28 có từ chối đúng không (càng cao càng tốt)
- `over_refusal_rate` — có từ chối oan câu trả lời được không (càng thấp càng tốt)

Hai chỉ số này phải đọc **cùng nhau**: threshold nâng cao thì `correct_refusal_rate`
đẹp lên nhưng `over_refusal_rate` xấu đi. Điểm cân bằng chính là thứ cần tìm khi
calibrate.

---

## 6. Gợi ý A/B

Yêu cầu đề bài là ≥2 config. Ba cặp đáng chạy nhất:

| So sánh | Câu phân hóa mạnh nhất | Giả thuyết |
|---|---|---|
| hybrid+rerank vs dense-only | Q13, Q14 (`lexical_exact`) | dense thua nặng ở số hiệu văn bản |
| top_k=3 vs top_k=8 | Q19, Q03 (`enumeration`), Q06, Q22 (`multi_hop`) | top_k nhỏ mất khoản cuối |
| có vs không PageIndex fallback | Q25, Q26, Q28 (`unanswerable`) | không fallback → bịa |

Báo cáo nên tách điểm **theo capability** (`by_capability`), không chỉ điểm trung bình
— trung bình che mất chuyện "hệ thống chạy tốt mọi thứ trừ tra cứu theo số hiệu".

---

## 7. Ngưỡng gợi ý

| Chỉ số | Tối thiểu | Tốt |
|---|---|---|
| Retrieval hit_rate | 0.70 | 0.85 |
| Retrieval recall | 0.60 | 0.80 |
| MRR | 0.55 | 0.75 |
| Faithfulness | 0.70 | 0.85 |
| Answer relevance | 0.70 | 0.85 |
| Context recall | 0.65 | 0.80 |
| Context precision | 0.60 | 0.75 |
| correct_refusal_rate | 0.75 | 1.00 |
| over_refusal_rate | ≤0.15 | ≤0.05 |

Đây là ngưỡng tham chiếu để định hướng, **không phải chuẩn ngành** — corpus tiếng Việt
văn phong pháp lý khó hơn benchmark tiếng Anh thông thường. Điều đáng báo cáo là
**chênh lệch giữa các config** và **trục năng lực nào yếu**, chứ không phải con số tuyệt đối.
