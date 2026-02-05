"""
Improved RAG với các kỹ thuật tiên tiến:
- Weighted Hybrid Retrieval
- Query Expansion & Classification
- Advanced Prompts
- Parent-Child Chunking (optional)
"""
from src.functions.utils import *
from langchain_core.runnables import RunnableLambda
from src.stores.FaissStore import FaissStore
from src.stores.Bm25Store import Bm25Store
from src.retrievers.FaissRetriever import FaissRetriever
from src.retrievers.Bm25Retriever import Bm25Retriever
from src.retrievers.WeightedHybridRetriever import WeightedHybridRetriever
from src.splitters.TextSplitter import TextSplitter
from src.splitters.ParentChildTextSplitter import ParentChildTextSplitter
from src.llms.llm import GeminiFlash
from src.chains.Reranker import CrossEncoderReranker
from src.chains.QueryExpander import QueryExpander
from src.chains.QueryClassifier import QueryClassifier
from src.chains.AdvancedPrompts import AdvancedQAPrompt


class ImprovedRAG:
    """
    RAG System với các cải tiến:
    - Query Expansion
    - Query Classification & Adaptive Retrieval
    - Weighted Hybrid Fusion
    - Advanced Prompts
    """
    
    def __init__(self, device='cuda', use_query_expansion=True, 
                 use_query_classification=True, use_parent_child=False):
        """
        Args:
            device: Device để chạy models ('cuda' hoặc 'cpu')
            use_query_expansion: Sử dụng query expansion
            use_query_classification: Sử dụng query classification để adaptive retrieval
            use_parent_child: Sử dụng parent-child chunking (experimental)
        """
        # LLM cho query processing
        self.llm = GeminiFlash().get_model()
        
        # Splitter
        if use_parent_child:
            self.splitter = ParentChildTextSplitter()
        else:
            self.splitter = TextSplitter()
        
        # Stores
        self.faiss_store = FaissStore()
        self.bm25_store = Bm25Store()
        
        # Base retrievers
        self.faiss_retriever = FaissRetriever(self.faiss_store)
        self.bm25_retriever = Bm25Retriever(self.bm25_store)
        
        # Hybrid retriever với default weights
        self.hybrid_retriever = WeightedHybridRetriever(
            self.bm25_retriever,
            self.faiss_retriever,
            bm25_weight=0.4,
            faiss_weight=0.6,
            k=20
        )
        
        # Reranker
        self.reranker = CrossEncoderReranker()
        
        # Query processing
        self.use_query_expansion = use_query_expansion
        self.use_query_classification = use_query_classification
        
        if use_query_expansion:
            self.query_expander = QueryExpander(self.llm)
        
        if use_query_classification:
            self.query_classifier = QueryClassifier(self.llm)
        
        # Advanced prompt
        self.qa_prompt = AdvancedQAPrompt(use_cot=True, use_citation=True)
        prompt = self.qa_prompt.get_prompt()
        
        # QA chain
        self.chain = (
            RunnableLambda(self._process_query)
            | RunnableLambda(lambda x: {
                "docs": self.hybrid_retriever.invoke(x["query"]),
                "question": x["original_query"]
            })
            | RunnableLambda(lambda x: {
                "docs": self.reranker.rerank(x["question"], x["docs"]),
                "question": x["question"]
            })
            | RunnableLambda(lambda x: {
                "context": combine_all_docs(x["docs"]),
                "question": x["question"]
            })
            | prompt
            | self.llm
        )
    
    def _process_query(self, query: str) -> dict:
        """
        Xử lý query: expansion và classification.
        
        Args:
            query: Query gốc
            
        Returns:
            dict: {"query": processed_query, "original_query": query}
        """
        original_query = query
        processed_query = query
        
        # Query expansion
        if self.use_query_expansion:
            processed_query = self.query_expander.expand(query)
        
        # Query classification và adaptive retrieval
        if self.use_query_classification:
            query_type, params = self.query_classifier.classify_and_get_params(query)
            
            # Điều chỉnh retrieval parameters
            self.hybrid_retriever.set_weights(
                params['bm25_weight'],
                params['faiss_weight']
            )
            self.hybrid_retriever.k = params['k']
            self.reranker.top_k = params['rerank_top_k']
            
            print(f"Query type: {query_type}, Params: {params}")
        
        return {
            "query": processed_query,
            "original_query": original_query
        }
    
    def add_document(self, path: str):
        """
        Thêm document vào hệ thống.
        
        Args:
            path: Đường dẫn đến file
        """
        docs = load_file(path)
        chunked_docs = self.splitter.split(docs)
        
        # Thêm metadata về source file
        for doc in chunked_docs:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = path
        
        self.faiss_store.add_documents(chunked_docs)
        self.bm25_store.add_documents(chunked_docs)
        return "Success!"
    
    def ask(self, question: str):
        """
        Trả lời câu hỏi.
        
        Args:
            question: Câu hỏi
            
        Returns:
            str: Câu trả lời
        """
        result = self.chain.invoke(question)
        
        # Extract content nếu là object
        if hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict) and 'content' in result:
            return result['content']
        else:
            return str(result)
