---
title: Legal QA VN
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ⚖️ Vietnamese Legal QA: Trợ lý tra cứu Bộ luật Lao động 2019

RAG hỏi đáp pháp luật lao động tiếng Việt: trích dẫn đúng Điều, từ chối câu hỏi ngoài dữ liệu,
mọi quyết định thiết kế đều dựa trên thí nghiệm có số liệu.

**Demo:** <link Space>

## Kiến trúc
DOCX (OCR) → làm sạch an toàn (không sửa chữ số) → tách Điều (ràng buộc số tăng dần) → chunk theo Điều/Khoản
→ bge-m3 → dense search (numpy) → LLM (Groq gpt-oss-120b, dự phòng gpt-oss-20b) → câu trả lời + nguồn trích dẫn
+ cảnh báo khi trích dẫn nằm ngoài ngữ cảnh.

## Thí nghiệm (POC trên Colab T4)
**Chunking và embedding** (Recall@5 / MRR@5)
| Cấu hình | BM25 | Dense |
|---|---|---|
| Chunk theo Điều | 0.765 / 0.640 | **0.959 / 0.865** |
| Chunk cố định 1000 ký tự | 0.786 / 0.682 | 0.939 / 0.832 |

| Embedding | Recall@5 | MRR@5 |
|---|---|---|
| **BAAI/bge-m3** | **0.959** | **0.865** |
| multilingual-e5-base | 0.929 | 0.816 |
| vietnamese-bi-encoder | 0.827 | 0.726 |

**Retrieval** (106 câu)
| Mode | Recall@5 | MRR@5 | Recall casual | Latency |
|---|---|---|---|---|
| BM25 | 0.764 | 0.650 | 0.64 | 2ms |
| **Dense** | **0.991** | **0.922** | **0.98** | 26ms |
| Hybrid (RRF) | 0.877 | 0.790 | 0.80 | |
| Dense + bge-reranker-v2-m3 | 0.953 | 0.919 | 0.92 | 2.6s |

→ Chọn dense: hybrid bị BM25 kéo xuống; reranker không tăng recall mà chậm hơn ~100 lần.

**Guardrail:** ngưỡng điểm reranker lọt 37.5% câu ngoài phạm vi (toàn câu pháp lý ngoài lĩnh vực:
tạm trú, hình sự, giao thông) → chuyển sang để LLM tự từ chối theo prompt: từ chối đúng 8/8 câu.

**Generation** (40 câu, giám khảo là model khác model trả lời): correctness 0.95 · faithfulness 1.00 ·
trích dẫn đúng Điều 0.95 · từ chối nhầm 1/40.

Hạn chế: bộ eval nhỏ, phần lớn câu do LLM sinh; các con số có khoảng tin cậy rộng.

## Chạy local
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements-dev.txt
    cp .env.example .env                                     # điền GROQ_API_KEY
    pytest
    python scripts/build_index.py --skip-embed               # kiểm tra parse
    python scripts/build_index.py --strict                   # build index
    python scripts/eval_retrieval.py
    python app.py

## OCR (dự phòng)
    python scripts/ocr_pdf.py path/to/scan.pdf

## Lưu ý
Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.
