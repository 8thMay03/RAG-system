"""
Ví dụ sử dụng evaluation system để đánh giá RAG system.

Cách sử dụng:
1. Đảm bảo đã có documents trong database (sử dụng RAG.add_document())
2. Đặt biến môi trường GOOGLE_API_KEY trong file .env
3. Chạy script này với dataset của bạn
"""

from src.chains.RAG import RAG
from src.eval.evaluate import evaluate_rag, evaluate_from_file, print_evaluation_summary
import torch

# Khởi tạo RAG system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rag = RAG(device=device)

# Ví dụ 1: Đánh giá với danh sách questions
questions = [
    "Luật giao thông đường bộ quy định gì về tốc độ?",
    "Người điều khiển xe phải mang theo những giấy tờ gì?",
    "Khi nào được phép vượt xe?",
]

# Optional: Ground truths (câu trả lời đúng)
ground_truths = [
    "Người lái xe phải chấp hành quy định về tốc độ, khoảng cách an toàn tối thiểu với xe phía trước cùng làn đường.",
    "Chứng nhận đăng ký xe, giấy phép lái xe, chứng nhận kiểm định, chứng nhận bảo hiểm.",
    "Khi không có chướng ngại vật phía trước, không có xe chạy ngược chiều, xe phía trước không có tín hiệu vượt xe khác.",
]

# Chạy đánh giá
print("Bắt đầu đánh giá RAG system...")
results_df = evaluate_rag(
    rag=rag,
    questions=questions,
    ground_truths=ground_truths  # Optional
)

# In kết quả
print("\nCác cột trong kết quả:", results_df.columns.tolist())
print("\nKết quả chi tiết:")
print(results_df)

# In tóm tắt
print_evaluation_summary(results_df)

# Lưu kết quả ra file
results_df.to_csv("evaluation_results.csv", index=False)
print("\nKết quả đã được lưu vào evaluation_results.csv")

# Ví dụ 2: Đánh giá từ file CSV/JSON
# results_df = evaluate_from_file(
#     rag=rag,
#     file_path="test_dataset.csv",
#     question_col="question",
#     ground_truth_col="ground_truth"  # Optional
# )
