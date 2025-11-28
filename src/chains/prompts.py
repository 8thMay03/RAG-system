from langchain_core.prompts import ChatPromptTemplate


class QA_prompt:
    def __init__(self):
        self.system_prompt = (
            """
                Hãy sử dụng đúng ngữ cảnh được cung cấp để trả lời câu hỏi.
                Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói "Thông tin này không có trong tài liệu được cung cấp.".
                Trả lời ngắn gọn tối đa ba câu. 
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