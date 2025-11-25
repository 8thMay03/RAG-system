from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import torch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

device = ('cuda' if torch.cuda.is_available() else 'cpu')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': device})

db = FAISS.load_local("../db/faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

model = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

system_prompt = (
    """
        Hãy sử dụng đúng ngữ cảnh được cung cấp để trả lời câu hỏi.
        Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói "Thông tin này không có trong tài liệu được cung cấp.".
        Trả lời ngắn gọn tối đa ba câu. 
        Ngữ cảnh: {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | prompt
    | model
)

if __name__ == '__main__':
    answer = chain.invoke("Nguyễn Quốc Khánh là ai?")
    print(answer)
