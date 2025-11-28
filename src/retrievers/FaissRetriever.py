class FaissRetriever:
    def __init__(self, faiss_store, k=3, search_type="similarity"):
        self.faiss_store = faiss_store
        self.k = k
        self.search_type = search_type

    def invoke(self, query):
        if self.faiss_store is None:
            return []
        retriever = self.faiss_store.db.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k},
        )
        return retriever.invoke(query)