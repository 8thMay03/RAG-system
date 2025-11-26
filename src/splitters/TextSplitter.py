from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def split(self, doc):
        return self.splitter.split_documents(doc)