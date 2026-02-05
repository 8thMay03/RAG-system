from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import uuid


class ParentChildTextSplitter:
    """
    Parent-Child Chunking Strategy.
    - Chia thành parent chunks lớn (giữ context)
    - Chia parent thành child chunks nhỏ (cho embedding)
    - Khi retrieve child, có thể lấy thêm parent context
    """
    
    def __init__(self, parent_chunk_size=2000, child_chunk_size=400, 
                 parent_overlap=200, child_overlap=50):
        """
        Args:
            parent_chunk_size: Kích thước parent chunk
            child_chunk_size: Kích thước child chunk
            parent_overlap: Overlap giữa các parent chunks
            child_overlap: Overlap giữa các child chunks
        """
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n", ". ", " ", ""]
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents thành parent-child chunks.
        
        Args:
            documents: List documents gốc
            
        Returns:
            List[Document]: List child chunks với metadata về parent
        """
        all_chunks = []
        
        # Bước 1: Tạo parent chunks
        parent_chunks = self.parent_splitter.split_documents(documents)
        
        # Bước 2: Tạo child chunks từ mỗi parent
        for parent_idx, parent in enumerate(parent_chunks):
            parent_id = str(uuid.uuid4())
            
            # Tạo child chunks từ parent
            children = self.child_splitter.split_documents([parent])
            
            for child_idx, child in enumerate(children):
                # Thêm metadata để link với parent
                child.metadata = child.metadata.copy()
                child.metadata['parent_id'] = parent_id
                child.metadata['parent_index'] = parent_idx
                child.metadata['child_index'] = child_idx
                child.metadata['total_children'] = len(children)
                # Lưu parent content preview (500 chars đầu)
                child.metadata['parent_content_preview'] = parent.page_content[:500]
                
                all_chunks.append(child)
        
        return all_chunks
    
    def get_parent_context(self, child_chunk: Document, parent_chunks: List[Document]) -> str:
        """
        Lấy parent context cho một child chunk.
        
        Args:
            child_chunk: Child chunk cần lấy context
            parent_chunks: List tất cả parent chunks
            
        Returns:
            str: Parent content
        """
        parent_id = child_chunk.metadata.get('parent_id')
        if not parent_id:
            return ""
        
        # Tìm parent chunk
        for parent in parent_chunks:
            if parent.metadata.get('parent_id') == parent_id:
                return parent.page_content
        
        return ""
