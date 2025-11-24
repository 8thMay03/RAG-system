
---

# RAG Chatbot using LangChain, FAISS, and Gemini API

## 📌 Overview

This project builds a **Retrieval-Augmented Generation (RAG) Chatbot** using:

* **LangChain** for constructing the retrieval + generation pipeline
* **FAISS** as the vector store for similarity search
* **Google Gemini API** for generating natural and accurate responses

The chatbot can answer questions based on **your own documents** through a retrieval process combined with language model generation.

---

## 🚀 Key Features

* 🔍 **Semantic search** powered by FAISS
* 📄 **Supports multiple document types**: PDF, text, docx
* ✂️ **Automatic chunking + embedding generation**
* 🧠 **Full RAG pipeline**: Retriever → LLM → Answer
* 💬 **Context-aware conversation support** (history-aware retriever)
* ⚡ **Fast deployment** with FastAPI
* 📦 **Local FAISS storage** to reduce costs

---

## Demo

![Demo](./demo.gif)

## 📂 Project Structure

```
project/
├── docs                     # Document storage
├── db                       # FAISS vector storage
├── src/
│   ├── app.py               # FastAPI server
│   ├── RAG.py               # RAG class
│   ├── utils.py             # Helper functions: chunking, loaders
│   └── index.html           # UI
├── requirements.txt
├── README.md
└── .env                     # API keys
```

## 🛠️ Technologies Used

* Python 3.13
* LangChain 1.x
* FAISS
* Google Gemini API
* FastAPI
* uvicorn
* HuggingFace

---

## 🔧 Installation

### 1️⃣ Clone the project

```
git clone https://github.com/8thMay03/RAG-system.git
cd RAG-system
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Create the `.env` file

```env
GOOGLE_API_KEY=your_api_key_here
```

## ▶️ Run the FastAPI server

```
python app.py
```

The API runs at:

```
http://127.0.0.1:8000/
```

## ▶️ Run the UI

Open the `index.html` file.

---

## 🔗 How the RAG pipeline works

1. The user sends a question
2. The system generates an embedding from the query
3. FAISS retrieves the most relevant document chunks
4. LangChain combines the context with the query
5. Gemini API generates an answer based on the documents

---

## 📜 License

MIT License.

---
