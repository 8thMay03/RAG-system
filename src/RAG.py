from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from functions.utils import *
from splitters.TextSplitter import TextSplitter
from VectorStores.FaissStore import FaissStore

class RAG:
    def __init__(self, device='cuda'):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': device})
        init_doc = load_file(r'D:\GithubRepositories\RAG-system\docs\doc.txt')

        # Split init_doc into chunks
        self.splitter = TextSplitter()
        chunks = self.splitter.split(init_doc)

        # Add to faiss_store
        self.faiss_store = FaissStore(self.embeddings)
        self.faiss_store.add(chunks)

        self.retriever = self.faiss_store.get_retriever(k=3)

        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

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
                ("human", "{question}"),
            ]
        )
        self.chain = (
            RunnableLambda(lambda x : {
                "docs": self.retriever.invoke(x),
                "question": x
            })
            | RunnableLambda(lambda x :{
                "context": combine_all_docs(x["docs"]),
                "question": x["question"]
            })
            | prompt
            | self.llm
        )
    
    def add_document(self, path):
        docs = load_file(path)
        chunks = self.splitter.split(docs)
        self.faiss_store.add(chunks)
        return "Success!"
    
    def ask(self, question):
        return self.chain.invoke(question)
