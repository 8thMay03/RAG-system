import shutil
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from pydantic import BaseModel
import torch

from src.chains.RAG import *

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


# Test
@app.get("/hello")
async def hello():
    return {"Hellp world!"}


# Upload documents
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp_path = f"docs\\uploaded_docs\\{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chatbot.add_document(tmp_path)
    return "Upload success!"


class ChatRequest(BaseModel):
    query: str


# Chat
@app.post("/chat")
async def query(chat_request: ChatRequest):
    answer = chatbot.ask(chat_request.query)
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000)
