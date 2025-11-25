import shutil

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from pydantic import BaseModel

from RAG import *

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
def hello():
    return {"Hellp world!"}

# Upload documents
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    tmp_path = f"../docs/uploaded_docs/{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chatbot.add_document(tmp_path)
    return "Upload success!"

class ChatRequest(BaseModel):
    query: str

# Chat
@app.post("/chat")
def query(question: ChatRequest):
    answer = chatbot.ask(question.query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)