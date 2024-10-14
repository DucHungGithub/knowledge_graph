from typing import List
from langchain_text_splitters import TextSplitter

import pandas as pd
from langchain_core.documents import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader




from utils.token import documentsToDataframe

def split_document_chunk(input_dir: str, chunk_overlap: int = 100, chunk_size: int = 1200) -> pd.DataFrame:
    loader = DirectoryLoader(input_dir, show_progress=True)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    list_documents =  splitter.split_documents(documents)
    
    return documentsToDataframe(list_documents)

def load_docs_in_chunks(input_dir: str, chunk_overlap: int = 100, chunk_size: int = 1200) -> List[str]:
    loader = DirectoryLoader(input_dir, show_progress=True)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    list_text = [doc.page_content for doc in splitter.split_documents(documents=documents)]
     
    return list_text