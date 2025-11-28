from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': DEVICE})

class FaissStore:
    def __init__(self, index_path="../db/faiss_index"):
        self.index_path = index_path
        self.embedding_model = embedding_model
        if os.path.exists(index_path):
            self.db = FAISS.load_local(index_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            self.db = None
        self.retriever = self.get_retriever()

    def add_documents(self, documents):
        if self.db is None:
            self.db = FAISS.from_documents(documents, self.embedding_model)
        else:
            self.db.add_documents(documents)
        self.db.save_local(self.index_path)
        self.retriever = self.get_retriever()

    def get_retriever(self, k=3):
        return self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
