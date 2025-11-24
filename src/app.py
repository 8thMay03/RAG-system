from fastapi import FastAPI, UploadFile, File
from langchain_core.documents import Document
import torch
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from utils import *
from RAG import *
import uvicorn
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

chatbot = RAG(device=DEVICE)

@app.get("/hello")
def hello():
    return {"Hellp world!"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    tmp_path = f"../docs/uploaded_docs/{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chatbot.add_document(tmp_path)
    return "Upload success!"

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def query(question: ChatRequest):
    answer = chatbot.ask(question.query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)