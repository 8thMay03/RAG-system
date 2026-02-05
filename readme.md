# 🚀 RAG System - Advanced Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Một hệ thống **Retrieval-Augmented Generation (RAG)** tiên tiến với Hybrid Retrieval, Query Processing, và Advanced Prompting để xây dựng chatbot thông minh dựa trên tài liệu của bạn.

---

## 📌 Tổng quan

Hệ thống RAG này cho phép bạn:

- 📚 **Tải và index** các tài liệu (PDF, TXT, DOCX)
- 🔍 **Tìm kiếm thông minh** với Hybrid Retrieval (BM25 + FAISS)
- 🎯 **Rerank** kết quả với CrossEncoder để tăng độ chính xác
- 💬 **Trả lời câu hỏi** dựa trên ngữ cảnh từ tài liệu
- 📊 **Đánh giá** hệ thống với RAGAS metrics
- ⚡ **API** sẵn sàng với FastAPI
- 🎨 **UI** đơn giản và thân thiện

### ✨ Tính năng nổi bật

- 🔄 **Hybrid Retrieval**: Kết hợp BM25 (lexical) và FAISS (semantic) với RRF fusion
- 🎯 **Weighted Fusion**: Điều chỉnh trọng số giữa BM25 và FAISS theo loại query
- 🔍 **Query Expansion**: Tự động mở rộng query để tăng recall
- 🧠 **Query Classification**: Phân loại query và adaptive retrieval
- 📝 **Advanced Prompts**: Few-shot examples, Chain-of-Thought, Citation
- 🔄 **Reranking**: CrossEncoder reranking để cải thiện precision
- 📊 **Evaluation**: Tích hợp RAGAS để đánh giá chất lượng

---

## 🏗️ Kiến trúc

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Query Expander   │ ◄─── Expand query với synonyms
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Query Classifier │ ◄─── Phân loại query (factual/conceptual/complex)
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Hybrid Retriever              │
│  ┌──────────┐   ┌──────────┐  │
│  │  BM25     │ + │  FAISS   │  │ ◄─── Weighted RRF Fusion
│  │ (Lexical) │   │(Semantic)│  │
│  └──────────┘   └──────────┘  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────┐
│   Reranker      │ ◄─── CrossEncoder reranking
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   LLM (Gemini)  │ ◄─── Generate answer với advanced prompts
└──────┬──────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

---

## 🚀 Tính năng chính

### Core Features

- ✅ **Multi-format Support**: PDF, TXT, DOCX
- ✅ **Hybrid Retrieval**: BM25 + FAISS với RRF fusion
- ✅ **Semantic Search**: FAISS vector search với embeddings
- ✅ **Lexical Search**: BM25 keyword matching
- ✅ **Reranking**: CrossEncoder để cải thiện kết quả
- ✅ **FastAPI API**: RESTful API sẵn sàng sử dụng
- ✅ **Web UI**: Giao diện web đơn giản

### Advanced Features

- 🎯 **Weighted Hybrid Retrieval**: Điều chỉnh trọng số BM25/FAISS
- 🔍 **Query Expansion**: Tự động mở rộng query
- 🧠 **Query Classification**: Adaptive retrieval theo query type
- 📝 **Parent-Child Chunking**: Giữ context tốt hơn
- 💡 **Advanced Prompts**: Few-shot, Chain-of-Thought, Citation
- 📊 **Evaluation System**: RAGAS metrics integration

---

## 📂 Cấu trúc Dự án

```
RAG-system/
├── docs/                          # Thư mục chứa documents
│   └── uploaded_docs/            # Documents đã upload
│
├── db/                            # Vector stores
│   ├── faiss_index/              # FAISS index
│   └── bm25_index/               # BM25 index
│
├── src/
│   ├── app.py                     # FastAPI application
│   │
│   ├── chains/                    # RAG chains và prompts
│   │   ├── RAG.py                 # RAG class cơ bản
│   │   ├── ImprovedRAG.py        # RAG với improvements ⭐
│   │   ├── QueryExpander.py      # Query expansion
│   │   ├── QueryClassifier.py    # Query classification
│   │   ├── Reranker.py           # CrossEncoder reranker
│   │   ├── AdvancedPrompts.py    # Advanced prompts
│   │   └── prompts.py            # Basic prompts
│   │
│   ├── retrievers/                # Retrievers
│   │   ├── HybridRetriever.py    # Basic hybrid retriever
│   │   ├── WeightedHybridRetriever.py  # Weighted hybrid ⭐
│   │   ├── FaissRetriever.py     # FAISS retriever
│   │   └── Bm25Retriever.py      # BM25 retriever
│   │
│   ├── stores/                    # Vector stores
│   │   ├── FaissStore.py         # FAISS store
│   │   └── Bm25Store.py          # BM25 store
│   │
│   ├── splitters/                 # Text splitters
│   │   ├── TextSplitter.py       # Basic splitter
│   │   └── ParentChildTextSplitter.py  # Parent-child ⭐
│   │
│   ├── llms/                      # LLM wrappers
│   │   └── llm.py                # Gemini, Mistral wrappers
│   │
│   ├── eval/                      # Evaluation
│   │   ├── evaluate.py           # RAGAS evaluation
│   │   └── example_evaluation.py # Example usage
│   │
│   ├── examples/                  # Examples
│   │   └── improved_rag_example.py
│   │
│   ├── functions/                 # Utilities
│   │   └── utils.py              # Helper functions
│   │
│   └── UI/                        # Web UI
│       └── index.html            # Frontend
│
├── requirements.txt               # Dependencies
├── readme.md                      # This file
├── RAG_IMPROVEMENTS.md            # Chi tiết improvements
├── IMPLEMENTATION_GUIDE.md        # Hướng dẫn sử dụng
├── IMPROVEMENTS_SUMMARY.md        # Tóm tắt improvements
└── REVIEW.md                      # Đánh giá hệ thống
```

---

## 🛠️ Cài đặt

### Yêu cầu

- Python 3.8+
- CUDA (optional, cho GPU acceleration)

### Bước 1: Clone repository

```bash
git clone https://github.com/8thMay03/RAG-system.git
cd RAG-system
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: Nếu có GPU, có thể cài `faiss-gpu` thay vì `faiss-cpu`:

```bash
pip install faiss-gpu
```

### Bước 3: Cấu hình API Key

Tạo file `.env` trong thư mục gốc:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Lấy API key tại: https://makersuite.google.com/app/apikey

---

## 🚀 Quick Start

### Cách 1: Sử dụng Improved RAG (Khuyến nghị)

```python
from src.chains.ImprovedRAG import ImprovedRAG
import torch

# Khởi tạo với các tính năng nâng cao
rag = ImprovedRAG(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_query_expansion=True,      # Bật query expansion
    use_query_classification=True,  # Bật adaptive retrieval
    use_parent_child=False         # Tùy chọn
)

# Thêm documents
rag.add_document("docs/your_document.pdf")

# Hỏi câu hỏi
answer = rag.ask("Câu hỏi của bạn là gì?")
print(answer)
```

### Cách 2: Sử dụng RAG cơ bản

```python
from src.chains.RAG import RAG
import torch

rag = RAG(device='cuda' if torch.cuda.is_available() else 'cpu')
rag.add_document("docs/document.pdf")
answer = rag.ask("Câu hỏi của bạn")
```

### Cách 3: Sử dụng API

**Khởi động server:**

```bash
python -m src.app
```

Server sẽ chạy tại: `http://127.0.0.1:8000`

**API Endpoints:**

- `GET /hello` - Health check
- `POST /upload` - Upload document
- `POST /chat` - Chat với RAG system

**Ví dụ với curl:**

```bash
# Upload document
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@docs/your_document.pdf"

# Chat
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Câu hỏi của bạn"}'
```

### Cách 4: Sử dụng Web UI

1. Khởi động API server: `python -m src.app`
2. Mở file `src/UI/index.html` trong browser
3. Upload document và bắt đầu chat!

---

## 📊 Đánh giá Hệ thống

Hệ thống tích hợp RAGAS để đánh giá chất lượng:

```python
from src.chains.ImprovedRAG import ImprovedRAG
from src.eval.evaluate import evaluate_rag, print_evaluation_summary
import torch

rag = ImprovedRAG(device='cuda')

questions = [
    "Câu hỏi 1?",
    "Câu hỏi 2?",
]

ground_truths = [
    "Câu trả lời đúng 1",
    "Câu trả lời đúng 2",
]

# Đánh giá
results = evaluate_rag(
    rag=rag,
    questions=questions,
    ground_truths=ground_truths
)

# In kết quả
print_evaluation_summary(results)
results.to_csv("evaluation_results.csv")
```

**Metrics được đánh giá:**
- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall
- Answer Correctness

---

## 🎯 Advanced Usage

### Weighted Hybrid Retrieval

```python
from src.retrievers.WeightedHybridRetriever import WeightedHybridRetriever

# Tạo retriever với custom weights
retriever = WeightedHybridRetriever(
    bm25_retriever,
    faiss_retriever,
    bm25_weight=0.6,  # Tăng weight cho BM25
    faiss_weight=0.4,
    k=20
)

# Điều chỉnh weights động
retriever.set_weights(0.7, 0.3)
```

### Query Expansion

```python
from src.chains.QueryExpander import QueryExpander
from src.llms.llm import GeminiFlash

llm = GeminiFlash().get_model()
expander = QueryExpander(llm)

# Expand query
expanded = expander.expand("Python là gì?")

# Generate multiple queries
queries = expander.generate_multiple_queries("RAG", n=3)
```

### Query Classification

```python
from src.chains.QueryClassifier import QueryClassifier

classifier = QueryClassifier(llm)

# Classify và lấy parameters
query_type, params = classifier.classify_and_get_params("Python là gì?")
# Returns: ("conceptual", {"bm25_weight": 0.3, "faiss_weight": 0.7, ...})
```

## 🛠️ Technologies

- **Python 3.13+**
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search
- **BM25** - Lexical search
- **Google Gemini API** - LLM
- **FastAPI** - API framework
- **RAGAS** - Evaluation framework
- **Sentence Transformers** - Embeddings
- **CrossEncoder** - Reranking

---

## 🔄 Workflow

```
1. Upload Document
   ↓
2. Text Extraction & Chunking
   ↓
3. Generate Embeddings (FAISS)
   ↓
4. Index với BM25
   ↓
5. User Query
   ↓
6. Query Expansion (optional)
   ↓
7. Query Classification (optional)
   ↓
8. Hybrid Retrieval (BM25 + FAISS)
   ↓
9. Reranking với CrossEncoder
   ↓
10. LLM Generation với Advanced Prompts
   ↓
11. Answer
```

---

## 🐛 Troubleshooting

### Lỗi thường gặp

**1. CUDA out of memory**
- Giảm `chunk_size` trong TextSplitter
- Sử dụng `faiss-cpu` thay vì `faiss-gpu`

**2. API key không hoạt động**
- Kiểm tra file `.env` có đúng format không
- Đảm bảo API key hợp lệ

**3. Query expansion chậm**
- Tắt `use_query_expansion=False` để tăng tốc
- Hoặc cache expanded queries

Xem thêm: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#troubleshooting)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [RAGAS](https://github.com/explodinggradients/ragas) - Evaluation framework
- [Google Gemini](https://ai.google.dev/) - LLM API

---

## 📞 Contact

- GitHub: [8thMay03/RAG-system](https://github.com/8thMay03/RAG-system)
- Issues: [GitHub Issues](https://github.com/8thMay03/RAG-system/issues)

---

**⭐ Nếu project này hữu ích, hãy star repo để ủng hộ!**
