from typing import Any, Dict, List, Optional
from models.base import Name
from pydantic import Field


class Document(Name):
    
    type: str = "text"
    """Type of the document."""

    text_unit_ids: List[str] = Field(default_factory=list)
    """list of text units in the document."""

    raw_content: str = ""
    """The raw text content of the document."""

    summary: Optional[str] = None
    """Summary of the document (optional)."""

    summary_embedding: Optional[List[float]] = None
    """The semantic embedding for the document summary (optional)."""

    raw_content_embedding: Optional[List[float]] = None
    """The semantic embedding for the document raw content (optional)."""

    attributes: Optional[Dict[str, Any]] = None
    """A dictionary of structured attributes such as author, etc (optional)."""