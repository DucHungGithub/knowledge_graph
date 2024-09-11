from typing import Any, Dict, List
import uuid

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_text_splitters.base import TextSplitter

from utils.uuid import gen_uuid
import config as defs


async def run_split_documents_text_units(
    docs: List[Document],
    args: Dict[str, Any]
) -> List[Document]:
    chunk_overlap = args.get("chunk_overlap", 100)
    chunk_size = args.get("chunk_size", 1200)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    splitter = TokenTextSplitter(encoding_name=encoding_name, chunk_overlap=chunk_overlap, chunk_size=chunk_size)
    return await split_documents_text_units(splitter=splitter, docs=docs, args=args)


async def split_documents_text_units(
    splitter: TextSplitter,
    docs: List[Document],
    args: Dict[str, Any]
) -> List[Document]:
    
    docs_texts = splitter.split_documents(documents=docs)
    for tu in docs_texts:
        tu.id = gen_uuid()
    
    return docs_texts