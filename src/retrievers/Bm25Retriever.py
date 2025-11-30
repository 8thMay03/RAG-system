class Bm25Retriever:
    def __init__(self, bm25_store, k=3):
        self.bm25_store = bm25_store
        self.k = k


    def invoke(self, query):
        if self.bm25_store is None:
            return []
        retriever = self.bm25_store.get_retriever()
        if retriever is None:
            return []
        retriever.k = self.k
        return retriever.invoke(query)
