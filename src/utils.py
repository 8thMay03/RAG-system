from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader 
from langchain_core.documents import Document

def load_file(path):
    if path.endswith('.pdf'): # PDF file
        pdf_loader = PyPDFLoader(path)
        pdf_docs = pdf_loader.load()
        return pdf_docs
    
    elif path.endswith('.docx'): # DOCX file
        docx_loader = Docx2txtLoader(path)
        docx_docs = docx_loader.load()
        return docx_docs
    
    elif path.endswith('.txt'): # TXT file
        text_loader = TextLoader(path, encoding='utf8')
        text_docs = text_loader.load()
        return text_docs
    
    else:
        raise ValueError("Unsupported file format")
    