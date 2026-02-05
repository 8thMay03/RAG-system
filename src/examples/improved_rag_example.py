"""
Example sử dụng Improved RAG với các kỹ thuật cải tiến.
"""
import torch
import os
from dotenv import load_dotenv
from src.chains.ImprovedRAG import ImprovedRAG

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Khởi tạo Improved RAG
    print("🚀 Khởi tạo Improved RAG...")
    rag = ImprovedRAG(
        device=DEVICE,
        use_query_expansion=True,      # Bật query expansion
        use_query_classification=True,  # Bật query classification
        use_parent_child=False         # Tắt parent-child (có thể bật nếu muốn)
    )
    
    # Thêm documents (nếu chưa có)
    # rag.add_document("docs/your_document.pdf")
    
    # Test với các loại queries khác nhau
    queries = [
        # Factual query
        "Python được tạo ra vào năm nào?",
        
        # Conceptual query
        "RAG là gì và nó hoạt động như thế nào?",
        
        # Complex query
        "So sánh ưu nhược điểm của RAG và fine-tuning",
    ]
    
    print("\n" + "="*60)
    print("TESTING IMPROVED RAG")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 60)
        
        answer = rag.ask(query)
        print(f"💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
