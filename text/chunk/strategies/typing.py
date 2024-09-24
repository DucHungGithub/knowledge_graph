from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    text_chunk: str = Field(description="The chunk after splitting docs")
    source_doc_indices: List[int]
    n_tokens: Optional[int] = Field(description="The number tokens of text chunk")
    
ChunkInput = Union[str,List[str],List[Tuple[str,str]]]
"""Input to a chunking strategy. Can be a string, a list of strings, or a list of tuples of (id, text)."""