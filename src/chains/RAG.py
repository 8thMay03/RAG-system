from src.functions.utils import *
from src.splitters.TextSplitter import TextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from src.chains.prompts import QA_prompt
from src.stores.FaissStore import FaissStore
from src.stores.Bm25Store import Bm25Store
from src.retrievers.HybridRetriever import HybridRetriever
from src.retrievers.FaissRetriever import FaissRetriever
from src.retrievers.Bm25Retriever import Bm25Retriever
from src.splitters.TextSplitter import TextSplitter


class RAG:
    def __init__(self, device='cuda'):
        # Splitter
        self.splitter = TextSplitter()

        # Faiss store
        self.faiss_store = FaissStore()
        self.faiss_retriever = FaissRetriever(self.faiss_store)

        # BM25 store
        self.bm25_store = Bm25Store()
        self.bm25_retriever = Bm25Retriever(self.bm25_store)

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.faiss_retriever)

        # LLM model
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        # Prompt template
        qa_prompt = QA_prompt()
        prompt = qa_prompt.get_prompt()

        # QA chain
        self.chain = (
            RunnableLambda(lambda x : {
                "docs": self.hybrid_retriever.invoke(x),
                "question": x
            })
            | RunnableLambda(lambda x :{
                "context": combine_all_docs(x["docs"]),
                "question": x["question"]
            })
            | prompt
            | self.llm
        )
    
    def add_document(self, path):
        docs = load_file(path)
        chunked_docs = self.splitter.split(docs)
        self.faiss_store.add_documents(chunked_docs)
        self.bm25_store.add_documents(chunked_docs)
        return "Success!"
    
    def ask(self, question):
        return self.chain.invoke(question)
