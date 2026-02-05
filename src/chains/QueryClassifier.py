from typing import Dict
from langchain_core.language_models import BaseLanguageModel


class QueryClassifier:
    """
    Phân loại query để điều chỉnh retrieval strategy.
    """
    
    QUERY_TYPES = {
        "factual": {
            "description": "Câu hỏi về sự kiện, số liệu cụ thể",
            "bm25_weight": 0.6,
            "faiss_weight": 0.4,
            "k": 10,
            "rerank_top_k": 3
        },
        "conceptual": {
            "description": "Câu hỏi về khái niệm, ý tưởng, giải thích",
            "bm25_weight": 0.3,
            "faiss_weight": 0.7,
            "k": 30,
            "rerank_top_k": 7
        },
        "complex": {
            "description": "Câu hỏi phức tạp cần nhiều thông tin",
            "bm25_weight": 0.5,
            "faiss_weight": 0.5,
            "k": 40,
            "rerank_top_k": 10,
            "use_multi_query": True
        },
        "comparison": {
            "description": "Câu hỏi so sánh",
            "bm25_weight": 0.4,
            "faiss_weight": 0.6,
            "k": 25,
            "rerank_top_k": 8
        }
    }
    
    def __init__(self, llm: BaseLanguageModel):
        """
        Args:
            llm: Language model để classify queries
        """
        self.llm = llm
    
    def classify(self, query: str) -> str:
        """
        Phân loại query vào một trong các loại.
        
        Args:
            query: Câu hỏi cần phân loại
            
        Returns:
            str: Loại query (factual, conceptual, complex, comparison)
        """
        prompt = f"""Phân loại câu hỏi sau vào một trong các loại:
- factual: Câu hỏi về sự kiện, số liệu cụ thể (ví dụ: "Năm nào Python được tạo ra?")
- conceptual: Câu hỏi về khái niệm, ý tưởng, giải thích (ví dụ: "RAG là gì?")
- complex: Câu hỏi phức tạp cần nhiều thông tin (ví dụ: "So sánh RAG và fine-tuning")
- comparison: Câu hỏi so sánh (ví dụ: "Sự khác biệt giữa X và Y?")

Câu hỏi: {query}

Chỉ trả về một từ: factual, conceptual, complex, hoặc comparison:"""
        
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, 'content'):
                result = result.content
            
            query_type = str(result).strip().lower().split()[0]
            
            # Validate
            if query_type in self.QUERY_TYPES:
                return query_type
            else:
                # Default fallback
                return "factual"
        except Exception as e:
            print(f"Error classifying query: {e}")
            return "factual"
    
    def get_retrieval_params(self, query_type: str) -> Dict:
        """
        Lấy parameters cho retrieval dựa trên query type.
        
        Args:
            query_type: Loại query
            
        Returns:
            Dict: Parameters cho retrieval
        """
        return self.QUERY_TYPES.get(query_type, self.QUERY_TYPES["factual"]).copy()
    
    def classify_and_get_params(self, query: str) -> tuple[str, Dict]:
        """
        Phân loại query và lấy parameters trong một lần.
        
        Args:
            query: Câu hỏi
            
        Returns:
            tuple: (query_type, params)
        """
        query_type = self.classify(query)
        params = self.get_retrieval_params(query_type)
        return query_type, params
