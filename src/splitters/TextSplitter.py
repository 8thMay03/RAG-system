from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

    def split(self, doc):
        return self.splitter.split_documents(doc)