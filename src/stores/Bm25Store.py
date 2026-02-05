from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import json
import os

class Bm25Store:
    def __init__(self, index_path='db/bm25_index/index.json'):
        self.index_path = index_path
        self.documents = self.build_documents()
        self.retriever = self.get_retriever()


    def build_documents(self):
        # Read the index file to build the BM25 documents
        self.documents = []
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            for doc in docs:
                self.documents.append(Document(page_content = doc["page_content"], metadata = doc["metadata"]))

        return self.documents


    def add_documents(self, new_docs):
        # Add new_docs
        self.documents.extend(new_docs)

        # Save ALL documents to json file (not just new_docs)
        json_all_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(json_all_docs, f, ensure_ascii=False, indent=2)

        self.retriever = self.get_retriever()


    def get_retriever(self):
        if not self.documents:
            return None
        return BM25Retriever.from_documents(self.documents)