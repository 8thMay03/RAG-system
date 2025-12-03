class FaissRetriever:
    def __init__(self, faiss_store, k=20, search_type="similarity"):
        self.faiss_store = faiss_store
        self.k = k
        self.search_type = search_type


    def invoke(self, query):
        # If faiss_store is None then we cant find any documents
        if self.faiss_store is None:
            return []
        retriever = self.faiss_store.get_retriever()
        # If retriever is None then we cant find any documents
        if retriever is None:
            return []
        return retriever.invoke(query)