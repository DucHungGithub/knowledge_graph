import logging
from typing import Any, Dict, List
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain.document_loaders import DirectoryLoader

import config as defs
from utils.uuid import gen_uuid

logger = logging.getLogger(__name__)

async def run_load_documents(
    args: Dict[str, Any]
) -> List[Document]:
    input_path = args.get("INPUT_FOLDER_PATH", defs.INPUT_FOLDER_PATH)
    if input_path is None:
        logger.exception(f"Can not find the path to load", exc_info=True)
        raise LookupError("Not found path")
    loader = DirectoryLoader(input_path, show_progress=True)
    
    return await load_documents(loader=loader, args=args)

async def load_documents(
    loader: BaseLoader,
    args: Dict[str, Any],
) -> List[Document]:
    documents = loader.load()
    for doc in documents:
        doc.id = gen_uuid()
    return documents
    
    