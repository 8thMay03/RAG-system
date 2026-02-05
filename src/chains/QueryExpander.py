from typing import List
from langchain_core.language_models import BaseLanguageModel


class QueryExpander:
    """
    Query Expansion và Rewriting để cải thiện retrieval.
    """
    
    def __init__(self, llm: BaseLanguageModel):
        """
        Args:
            llm: Language model để expand queries
        """
        self.llm = llm
    
    def expand(self, query: str) -> str:
        """
        Mở rộng query bằng cách thêm synonyms và related terms.
        
        Args:
            query: Query gốc
            
        Returns:
            str: Query đã được expand
        """
        prompt = f"""Hãy mở rộng câu hỏi sau bằng cách thêm các từ khóa liên quan và synonyms.
Giữ nguyên ý nghĩa gốc và ngữ cảnh.

Câu hỏi gốc: {query}

Câu hỏi mở rộng (chỉ trả về câu hỏi, không giải thích):"""
        
        try:
            expanded = self.llm.invoke(prompt)
            # Extract content nếu là object
            if hasattr(expanded, 'content'):
                expanded = expanded.content
            return str(expanded).strip()
        except Exception as e:
            # Fallback: trả về query gốc nếu có lỗi
            print(f"Error expanding query: {e}")
            return query
    
    def generate_multiple_queries(self, query: str, n: int = 3) -> List[str]:
        """
        Tạo nhiều queries từ 1 query gốc để tăng recall.
        
        Args:
            query: Query gốc
            n: Số lượng queries muốn tạo
            
        Returns:
            List[str]: Danh sách queries
        """
        prompt = f"""Tạo {n} câu hỏi khác nhau nhưng cùng mục đích với câu hỏi sau.
Mỗi câu hỏi nên nhấn mạnh một khía cạnh khác nhau.

Câu hỏi gốc: {query}

Danh sách {n} câu hỏi (mỗi câu trên một dòng, không đánh số):"""
        
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, 'content'):
                result = result.content
            
            # Parse queries từ result
            queries = [q.strip() for q in str(result).split('\n') if q.strip()]
            # Thêm query gốc vào đầu
            queries = [query] + queries[:n-1]
            return queries[:n]
        except Exception as e:
            print(f"Error generating multiple queries: {e}")
            return [query]
    
    def rewrite(self, query: str) -> str:
        """
        Rewrite query để rõ ràng và cụ thể hơn.
        
        Args:
            query: Query gốc
            
        Returns:
            str: Query đã được rewrite
        """
        prompt = f"""Hãy viết lại câu hỏi sau để rõ ràng và cụ thể hơn, nhưng giữ nguyên ý nghĩa.
Nếu câu hỏi đã rõ ràng, giữ nguyên.

Câu hỏi gốc: {query}

Câu hỏi đã viết lại (chỉ trả về câu hỏi, không giải thích):"""
        
        try:
            rewritten = self.llm.invoke(prompt)
            if hasattr(rewritten, 'content'):
                rewritten = rewritten.content
            return str(rewritten).strip()
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query
