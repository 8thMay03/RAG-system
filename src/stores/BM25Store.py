from filetype import document_match
from langchain_community.retrievers import BM25Retriever
import json
import os

class BM25Store:
    def __init__(self, index_path):
        self.index_path = index_path
        self.retriever = self.get_retriever()
        if os.path.exists(index_path):
            self.documents = self.build_documents()
        else:
            self.documents = []


    def build_documents(self):
        # Read the index file to build the BM25 documents
        with open(self.index_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        # Add doc to document store of BM25
        for doc in docs:
            self.documents.append(doc)
        self.retriever = self.get_retriever()
        return self.documents


    def add_documents(self, docs):
        with open(self.index_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        self.documents.extend(docs)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        self.retriever = self.get_retriever()


    def get_retriever(self):
        return BM25Retriever.from_documents(self.documents)