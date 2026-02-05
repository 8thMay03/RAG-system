from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset
from typing import List, Dict, Optional
import pandas as pd
import torch
import os
from dotenv import load_dotenv
import asyncio
import time

# Import RunConfig với fallback
try:
    from ragas import RunConfig
except ImportError:
    try:
        from ragas.run_config import RunConfig
    except ImportError:
        RunConfig = None
        print("⚠️ Không thể import RunConfig. Sẽ dùng cấu hình mặc định.")

from src.chains.RAG import RAG
from src.functions.utils import combine_all_docs

load_dotenv()

# Ưu tiên GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# Chuẩn bị RAG instance
rag = RAG(device=DEVICE)


def _setup_ragas_with_gemini():
    """
    Cấu hình RAGAS để sử dụng Google Gemini thay vì OpenAI.
    Trả về tuple (llm, embeddings) hoặc (None, None) nếu không thể cấu hình.
    """
    if not GOOGLE_API_KEY:
        print("⚠️ Không tìm thấy GOOGLE_API_KEY. RAGAS sẽ cần OPENAI_API_KEY.")
        return None, None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        
        # Thử import wrapper cho ragas >= 0.2.0
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.2,
                    google_api_key=GOOGLE_API_KEY,
                )
            )
            embeddings = LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GOOGLE_API_KEY,
                )
            )
            print("✅ Đã cấu hình RAGAS với Google Gemini (ragas >= 0.2.0)")
            return llm, embeddings
        except ImportError:
            pass
        
        # Thử import wrapper cho ragas 0.1.x
        try:
            from ragas.llms import LangchainLLM
            from ragas.embeddings import LangchainEmbeddings
            
            llm = LangchainLLM(
                ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.2,
                    google_api_key=GOOGLE_API_KEY,
                )
            )
            embeddings = LangchainEmbeddings(
                GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GOOGLE_API_KEY,
                )
            )
            print("✅ Đã cấu hình RAGAS với Google Gemini (ragas 0.1.x)")
            return llm, embeddings
        except ImportError:
            pass
        
        print("⚠️ Không thể import LLM wrapper từ ragas. Hãy nâng cấp: pip install 'ragas>=0.2.0'")
        return None, None
        
    except ImportError as e:
        print(f"⚠️ Thiếu dependency: {e}")
        print("Hãy cài đặt: pip install langchain-google-genai")
        return None, None


def get_rag_outputs(rag: RAG, question: str) -> Dict[str, any]:
    """
    Lấy answer và contexts từ RAG system cho một câu hỏi.
    
    Args:
        rag: Instance của RAG class
        question: Câu hỏi cần đánh giá
        
    Returns:
        Dict chứa 'answer' và 'contexts'
    """
    # Lấy documents từ hybrid retriever
    docs = rag.hybrid_retriever.invoke(question)
    
    # Rerank documents
    reranked_docs = rag.reranker.rerank(question, docs)
    
    # Lấy contexts (list các page_content)
    contexts = [doc.page_content for doc in reranked_docs]
    
    # Lấy answer từ chain
    answer = rag.ask(question)
    
    # Nếu answer là string, giữ nguyên; nếu là object có content, lấy content
    if hasattr(answer, 'content'):
        answer = answer.content
    elif isinstance(answer, dict) and 'content' in answer:
        answer = answer['content']
    elif not isinstance(answer, str):
        answer = str(answer)
    
    return {
        "answer": answer,
        "contexts": contexts
    }


def evaluate_rag(
    rag: RAG,
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List] = None,
) -> pd.DataFrame:
    """
    Đánh giá hệ thống RAG sử dụng RAGAS.
    
    Args:
        rag: Instance của RAG class
        questions: Danh sách câu hỏi để đánh giá
        ground_truths: Danh sách câu trả lời đúng (optional, cho answer_correctness)
        metrics: Danh sách metrics để đánh giá. Nếu None, sử dụng tất cả metrics mặc định
        
    Returns:
        DataFrame chứa kết quả đánh giá
    """
    if metrics is None:
        metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
        # Thêm answer_correctness nếu có ground_truths
        if ground_truths is not None:
            metrics.append(answer_correctness)
    
    print(f"Đang đánh giá {len(questions)} câu hỏi...")
    
    # Lấy answers và contexts từ RAG system
    results = []
    for i, question in enumerate(questions):
        print(f"Đang xử lý câu hỏi {i+1}/{len(questions)}: {question[:50]}...")
        try:
            output = get_rag_outputs(rag, question)
            results.append({
                "question": question,
                "answer": output["answer"],
                "contexts": output["contexts"]
            })
        except Exception as e:
            print(f"Lỗi khi xử lý câu hỏi {i+1}: {e}")
            results.append({
                "question": question,
                "answer": "",
                "contexts": []
            })
        
        # Delay 2 giây giữa các câu hỏi để tránh rate limit
        if i < len(questions) - 1:
            time.sleep(2)
    
    # Tạo dataset cho RAGAS
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results]
    }
    
    # Thêm ground_truths nếu có
    if ground_truths is not None:
        if len(ground_truths) != len(questions):
            raise ValueError(f"Số lượng ground_truths ({len(ground_truths)}) phải bằng số lượng questions ({len(questions)})")
        data["ground_truth"] = ground_truths
    
    dataset = Dataset.from_dict(data)
    
    # Chuẩn bị LLM và Embeddings cho RAGAS
    ragas_llm, ragas_embeddings = _setup_ragas_with_gemini()

    # Chạy đánh giá
    print("\nĐang chạy đánh giá với RAGAS...")
    
    # Cấu hình để tránh timeout và rate limit
    run_config = None
    if RunConfig is not None:
        try:
            run_config = RunConfig(
                timeout=300,       # Tăng timeout lên 5 phút
                max_retries=3,     # Retry 3 lần nếu lỗi
                max_wait=120,      # Chờ tối đa 2 phút giữa các retry
                max_workers=1,     # Chạy tuần tự để tránh rate limit
            )
            print("📋 Cấu hình: timeout=300s, max_workers=1 (tuần tự), max_retries=3")
        except Exception as e:
            print(f"⚠️ Không thể tạo RunConfig: {e}")
            run_config = None
    
    # Tạo kwargs cho evaluate
    eval_kwargs = {
        "dataset": dataset,
        "metrics": metrics,
    }
    
    if ragas_llm is not None and ragas_embeddings is not None:
        eval_kwargs["llm"] = ragas_llm
        eval_kwargs["embeddings"] = ragas_embeddings
    else:
        print("⚠️ Sử dụng cấu hình mặc định của RAGAS (cần OPENAI_API_KEY)")
    
    if run_config is not None:
        eval_kwargs["run_config"] = run_config
    
    result = evaluate(**eval_kwargs)
    
    # Convert sang DataFrame
    df = result.to_pandas()
    
    return df


def evaluate_from_file(
    rag: RAG,
    file_path: str,
    question_col: str = "question",
    ground_truth_col: Optional[str] = None,
    metrics: Optional[List] = None
) -> pd.DataFrame:
    """
    Đánh giá RAG system từ file CSV hoặc JSON.
    
    Args:
        rag: Instance của RAG class
        file_path: Đường dẫn đến file chứa questions (và optional ground_truths)
        question_col: Tên cột chứa questions
        ground_truth_col: Tên cột chứa ground_truths (optional)
        metrics: Danh sách metrics để đánh giá
        
    Returns:
        DataFrame chứa kết quả đánh giá
    """
    # Đọc file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("File phải là CSV hoặc JSON")
    
    questions = df[question_col].tolist()
    ground_truths = None
    if ground_truth_col and ground_truth_col in df.columns:
        ground_truths = df[ground_truth_col].tolist()
    
    return evaluate_rag(rag, questions, ground_truths, metrics)


def print_evaluation_summary(df: pd.DataFrame):
    """
    In tóm tắt kết quả đánh giá.
    
    Args:
        df: DataFrame kết quả từ evaluate_rag
    """
    print("\n" + "="*60)
    print("TÓM TẮT KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)
    
    # Danh sách các cột không phải metric
    non_metric_cols = ['question', 'answer', 'contexts', 'ground_truth', 'user_input', 'response', 'retrieved_contexts', 'reference']
    
    # Chỉ lấy các cột numeric (là metric scores)
    metric_cols = []
    for col in df.columns:
        if col.lower() not in [c.lower() for c in non_metric_cols]:
            # Kiểm tra xem cột có phải numeric không
            if pd.api.types.is_numeric_dtype(df[col]):
                metric_cols.append(col)
    
    if not metric_cols:
        print("\n⚠️ Không tìm thấy cột metric nào trong kết quả.")
        print(f"Các cột có trong DataFrame: {list(df.columns)}")
        print("\n" + "="*60)
        return
    
    for metric in metric_cols:
        mean_score = df[metric].mean()
        std_score = df[metric].std()
        min_score = df[metric].min()
        max_score = df[metric].max()
        
        print(f"\n{metric.upper()}:")
        print(f"  Trung bình: {mean_score:.4f}")
        print(f"  Độ lệch chuẩn: {std_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")
    
    print("\n" + "="*60)
