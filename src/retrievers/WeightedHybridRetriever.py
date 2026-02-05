from collections import defaultdict
from typing import List
from langchain_core.documents import Document


class WeightedHybridRetriever:
    """
    Hybrid Retriever với Weighted RRF Fusion.
    Cho phép điều chỉnh trọng số giữa BM25 và FAISS retrieval.
    """
    
    def __init__(self, bm25_retriever, faiss_retriever, 
                 bm25_weight=0.4, faiss_weight=0.6, k=20, rrf_k=60):
        """
        Args:
            bm25_retriever: BM25 retriever instance
            faiss_retriever: FAISS retriever instance
            bm25_weight: Trọng số cho BM25 (default: 0.4)
            faiss_weight: Trọng số cho FAISS (default: 0.6)
            k: Số documents trả về (default: 20)
            rrf_k: Constant cho RRF formula (default: 60)
        """
        self.bm25_retriever = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.k = k
        self.rrf_k = rrf_k
        
        # Normalize weights
        total_weight = bm25_weight + faiss_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.faiss_weight = faiss_weight / total_weight

    def _get_doc_id(self, doc: Document) -> str:
        """Tạo unique ID cho document."""
        doc_id = doc.metadata.get("id", None)
        if doc_id is None:
            # Fallback: sử dụng hash của content
            doc_id = str(hash(doc.page_content[:100]))
        return str(doc_id)

    def weighted_rrf_fusion(self, bm25_docs: List[Document], 
                           faiss_docs: List[Document]) -> List[Document]:
        """
        Merge kết quả từ BM25 và FAISS với weighted RRF.
        
        Args:
            bm25_docs: Documents từ BM25 retriever
            faiss_docs: Documents từ FAISS retriever
            
        Returns:
            List[Document]: Top-k documents sau khi fusion
        """
        scores = defaultdict(float)
        doc_store = {}

        # BM25 với weight
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_id(doc)
            doc_store[doc_id] = doc
            # Weighted RRF score
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[doc_id] += self.bm25_weight * rrf_score

        # FAISS với weight
        for rank, doc in enumerate(faiss_docs):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_store:
                doc_store[doc_id] = doc
            # Weighted RRF score
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[doc_id] += self.faiss_weight * rrf_score

        # Sort theo score giảm dần
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        # Trả về top-k documents
        return [doc_store[i] for i in sorted_ids[:self.k]]

    def invoke(self, query: str) -> List[Document]:
        """
        Retrieve documents cho query.
        
        Args:
            query: Câu hỏi cần retrieve
            
        Returns:
            List[Document]: Top-k documents
        """
        # Retrieve từ cả hai retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)
        
        # Fusion với weighted RRF
        return self.weighted_rrf_fusion(bm25_docs, faiss_docs)
    
    def set_weights(self, bm25_weight: float, faiss_weight: float):
        """
        Điều chỉnh trọng số động.
        
        Args:
            bm25_weight: Trọng số mới cho BM25
            faiss_weight: Trọng số mới cho FAISS
        """
        total_weight = bm25_weight + faiss_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.faiss_weight = faiss_weight / total_weight
