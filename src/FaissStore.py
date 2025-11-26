from langchain_community.vectorstores import FAISS
import os

class FaissStore:
    def __init__(self, embedding_model, index_path="../db/faiss_index"):
        self.index_path = index_path
        self.embeddings = embedding_model
        if os.path.exists(index_path):
            self.db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.db = None

    def add(self, documents):
        if self.db is None:
            self.db = FAISS.from_documents(documents, self.embeddings)
        else:
            self.db.add_documents(documents)
        self.db.save_local(self.index_path)

    def get_retriever(self, k=3):
        return self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
