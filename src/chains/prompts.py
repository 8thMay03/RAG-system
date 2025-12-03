from langchain_core.prompts import ChatPromptTemplate


class QA_prompt:
    def __init__(self):
        self.system_prompt = (
            """
                Hãy sử dụng đúng ngữ cảnh được cung cấp để trả lời câu hỏi.
                Bạn hãy trả lời chi tiết và đầy đủ, không giới hạn số câu.
                Không được tóm tắt. Viết đầy đủ toàn bộ kết quả. 
                Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói "Thông tin này không có trong tài liệu được cung cấp.".
                Ngữ cảnh: {context}
            """
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

    def get_prompt(self):
        return self.prompt