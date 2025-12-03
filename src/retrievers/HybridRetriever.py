from collections import defaultdict


class HybridRetriever:
    def __init__(self, bm25_retriever, faiss_retriever, k=20, rrf_k=60):
        self.bm25_retriever = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.k = k
        self.rrf_k = rrf_k


    def rrf_fusion(self, results_list):
        """
        results_list: list[list[Document]]
            Ex: [bm25_docs, faiss_docs]
        """

        scores = defaultdict(float)
        doc_store = {}

        for results in results_list:
            for rank, doc in enumerate(results):
                # tạo id duy nhất cho doc (metadata hoặc page_content)
                doc_id = doc.metadata.get("id", None)

                if doc_id is None:
                    # fallback nếu user chưa gán id
                    doc_id = doc.page_content[:50]

                # Lưu doc để trả về sau
                doc_store[doc_id] = doc

                # Tính điểm RRF
                scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        # Sort theo score giảm dần
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        # Trả về top-k document
        return [doc_store[i] for i in sorted_ids[:self.k]]

    def invoke(self, query):
        bm25_docs = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)
        return self.rrf_fusion([bm25_docs, faiss_docs])