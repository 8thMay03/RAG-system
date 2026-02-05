import shutil
import os
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from pydantic import BaseModel
import torch

from src.chains.RAG import *

app = FastAPI(
    title="RAG System API",
    description="API cho hệ thống RAG với Hybrid Retrieval",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Thay bằng specific origins trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# Đảm bảo thư mục upload tồn tại
UPLOAD_DIR = Path("docs/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

chatbot = RAG(device=DEVICE)


# Test
@app.get("/hello")
async def hello():
    return {"message": "Hello world!"}


# Upload documents
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng file không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size (nếu có)
        if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File quá lớn. Kích thước tối đa: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
        
        # Tạo đường dẫn an toàn với pathlib
        safe_filename = Path(file.filename).name  # Chỉ lấy tên file, loại bỏ path
        tmp_path = UPLOAD_DIR / safe_filename
        
        # Lưu file
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Thêm document vào RAG system
        result = chatbot.add_document(str(tmp_path))
        
        return {
            "status": "success",
            "message": "Upload và index thành công!",
            "filename": file.filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi upload file: {str(e)}"
        )


class ChatRequest(BaseModel):
    query: str


# Chat
@app.post("/chat")
async def query(chat_request: ChatRequest):
    try:
        if not chat_request.query or not chat_request.query.strip():
            raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
        
        answer = chatbot.ask(chat_request.query)
        
        # Xử lý answer nếu là object
        if hasattr(answer, 'content'):
            answer = answer.content
        elif isinstance(answer, dict) and 'content' in answer:
            answer = answer['content']
        
        return {"answer": str(answer)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000)
