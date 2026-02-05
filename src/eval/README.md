# RAG System Evaluation

Module này cung cấp các công cụ để đánh giá hệ thống RAG sử dụng RAGAS (Retrieval-Augmented Generation Assessment).

## Cài đặt

Đảm bảo đã cài đặt các dependencies:
```bash
pip install ragas datasets pandas sentence-transformers langchain-google-genai
# (Khuyến nghị) nếu cần dùng Gemini qua LangChain trong ragas:
# pip install "ragas>=0.1.9"
```

> Nếu dùng Gemini cho chấm điểm, cần biến môi trường `GOOGLE_API_KEY`.
> Nếu không có, ragas sẽ fallback sang mặc định (OpenAI), cần `OPENAI_API_KEY`.

## Các Metrics được sử dụng

1. **Answer Relevancy**: Đánh giá độ liên quan của câu trả lời với câu hỏi
2. **Faithfulness**: Đánh giá độ trung thực - câu trả lời có dựa trên context được retrieve không
3. **Context Precision**: Đánh giá độ chính xác của context được retrieve
4. **Context Recall**: Đánh giá độ recall - có retrieve đủ context cần thiết không
5. **Answer Correctness**: Đánh giá độ chính xác của câu trả lời so với ground truth (cần ground_truth)

## Cách sử dụng

### 1. Đánh giá với danh sách questions

```python
from src.chains.RAG import RAG
from src.eval.evaluate import evaluate_rag, print_evaluation_summary, _default_ragas_llm
import torch

# Khởi tạo RAG
rag = RAG(device='cuda' if torch.cuda.is_available() else 'cpu')

# Danh sách câu hỏi
questions = [
    "Câu hỏi 1?",
    "Câu hỏi 2?",
    "Câu hỏi 3?",
]

# Optional: Ground truths (câu trả lời đúng)
ground_truths = [
    "Câu trả lời đúng 1",
    "Câu trả lời đúng 2",
    "Câu trả lời đúng 3",
]

# Chạy đánh giá
results_df = evaluate_rag(
    rag=rag,
    questions=questions,
    ground_truths=ground_truths,    # Optional
    llm=_default_ragas_llm(),       # Dùng Gemini nếu có GOOGLE_API_KEY; nếu không, ragas dùng mặc định (OpenAI)
)

# In kết quả
print(results_df)
print_evaluation_summary(results_df)

# Lưu kết quả
results_df.to_csv("evaluation_results.csv", index=False)
```

### 2. Đánh giá từ file CSV/JSON

Tạo file CSV hoặc JSON với format:
- **CSV**: Cột `question` (bắt buộc), `ground_truth` (optional)
- **JSON**: Tương tự

```python
from src.eval.evaluate import evaluate_from_file

results_df = evaluate_from_file(
    rag=rag,
    file_path="test_dataset.csv",
    question_col="question",
    ground_truth_col="ground_truth"  # Optional
)
```

### 3. Sử dụng metrics tùy chỉnh

```python
from ragas.metrics import answer_relevancy, faithfulness

results_df = evaluate_rag(
    rag=rag,
    questions=questions,
    metrics=[answer_relevancy, faithfulness]  # Chỉ đánh giá 2 metrics này
)
```

## Kết quả

Kết quả trả về là một pandas DataFrame chứa:
- `question`: Câu hỏi
- `answer`: Câu trả lời từ RAG system
- `contexts`: List các context được retrieve
- `ground_truth`: Ground truth (nếu có)
- Các cột metrics: `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`, `answer_correctness`

## Ví dụ

Xem file `example_evaluation.py` để biết ví dụ chi tiết.

