from langchain_core.documents import Document
import torch
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from utils import *

class RAG:
    def __init__(self, device):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': device})
        init_doc = load_file(r'D:\GithubRepositories\RAG-system\docs\doc.txt')
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = self.splitter.split_documents(init_doc)
        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.db.save_local("../db/faiss_index")
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
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
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
    
    def add_document(self, path):
        docs = load_file(path)
        chunks = self.splitter.split_documents(docs)
        self.db.add_documents(chunks)
        return "Success!"
    
    def ask(self, question):
        return self.chain.invoke({"input" : question})["answer"]
