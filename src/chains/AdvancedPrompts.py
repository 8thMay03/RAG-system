from langchain_core.prompts import ChatPromptTemplate


class AdvancedQAPrompt:
    """
    Advanced Prompt với Few-shot examples và Chain-of-Thought.
    """
    
    def __init__(self, use_cot=True, use_citation=True):
        """
        Args:
            use_cot: Sử dụng Chain-of-Thought reasoning
            use_citation: Yêu cầu cite sources
        """
        self.use_cot = use_cot
        self.use_citation = use_citation
        self.prompt = self._build_prompt()
    
    def _build_prompt(self):
        """Xây dựng prompt template."""
        
        system_prompt = """Bạn là một trợ lý AI chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

QUY TẮC QUAN TRỌNG:
1. CHỈ sử dụng thông tin trong ngữ cảnh được cung cấp
2. Nếu không có thông tin trong ngữ cảnh, hãy nói rõ: "Thông tin này không có trong tài liệu được cung cấp."
3. Trả lời chi tiết và đầy đủ, không tóm tắt
4. Sử dụng ngôn ngữ tự nhiên, dễ hiểu"""
        
        if self.use_citation:
            system_prompt += "\n5. Khi có thể, hãy trích dẫn nguồn (ví dụ: 'Theo tài liệu...', 'Trong phần...')"
        
        if self.use_cot:
            system_prompt += "\n6. Suy luận từng bước: giải thích cách bạn đi đến câu trả lời"
        
        system_prompt += """

VÍ DỤ:
Ngữ cảnh: Python là một ngôn ngữ lập trình phổ biến được tạo ra bởi Guido van Rossum vào năm 1991. Python nổi tiếng với cú pháp đơn giản và dễ đọc.

Câu hỏi: Python là gì và ai tạo ra nó?
Trả lời: Python là một ngôn ngữ lập trình phổ biến. Theo tài liệu, Python được tạo ra bởi Guido van Rossum vào năm 1991. Python nổi tiếng với cú pháp đơn giản và dễ đọc, giúp các lập trình viên viết code một cách hiệu quả hơn.

---
Ngữ cảnh: RAG (Retrieval-Augmented Generation) là một kỹ thuật kết hợp retrieval và generation.

Câu hỏi: Khi nào RAG được phát minh?
Trả lời: Thông tin này không có trong tài liệu được cung cấp.

---
Ngữ cảnh: {context}"""
        
        human_template = "Câu hỏi: {question}\n\nHãy trả lời chi tiết:"
        
        if self.use_cot:
            human_template += "\n\n(Suy luận từng bước nếu cần)"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template)
        ])
    
    def get_prompt(self):
        """Trả về prompt template."""
        return self.prompt
