import os
import logging
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain.document_loaders import DirectoryLoader
import config as defs
from utils.uuid import gen_uuid

logger = logging.getLogger(__name__)

# Function to save checkpoint
def save_checkpoint(checkpoint_file: str, processed_files: List[str]):
    with open(checkpoint_file, 'w') as f:
        f.write('\n'.join(processed_files))

# Function to load checkpoint
def load_checkpoint(checkpoint_file: str) -> List[str]:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return f.read().splitlines()
    return []

async def run_load_documents(
    args: Dict[str, Any]
) -> List[Document]:
    input_path = args.get("INPUT_FOLDER_PATH", defs.INPUT_FOLDER_PATH)
    checkpoint_file = args.get("CHECKPOINT_FILE", "checkpoint.txt")
    
    if input_path is None:
        logger.exception(f"Cannot find the path to load", exc_info=True)
        raise LookupError("Not found path")

    # Load the checkpoint (files already processed)
    processed_files = load_checkpoint(checkpoint_file)

    loader = DirectoryLoader(input_path, show_progress=True)

    # Load and filter unprocessed documents
    all_documents = await load_documents(loader=loader, args=args)
    unprocessed_documents = [
        doc for doc in all_documents if doc.metadata['source'] not in processed_files
    ]

    # Process the unprocessed documents
    for doc in unprocessed_documents:
        doc.id = gen_uuid()

    # Save the checkpoint (processed files)
    processed_files.extend([doc.metadata['source'] for doc in unprocessed_documents])
    save_checkpoint(checkpoint_file, processed_files)

    return unprocessed_documents

async def load_documents(
    loader: BaseLoader,
    args: Dict[str, Any],
) -> List[Document]:
    documents = loader.load()  # Load all documents
    return documents
