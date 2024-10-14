from typing import Awaitable, Callable, List, Dict, Any

from langchain_core.documents import Document
from pydantic import BaseModel


class EntityExtractionResult(BaseModel):
    entities: List[Dict[str, Any]]
    rels: List[Dict[str, Any]]
    graphml_graph: str | None
    

EntityExtractStrategy = Callable[
    [
        List[Document],
        List[str],
        Dict[str, Any]
    ],
    Awaitable[EntityExtractionResult]
]