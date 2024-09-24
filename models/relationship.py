from typing import Any, Dict, List, Optional
from models.base import Identify


class Relationship(Identify):
    source: Optional[str] = None
    """The source entity name."""
    
    target: Optional[str] = None
    """The target entity name."""
    
    weight: Optional[float] = 1.0
    """The edge weight."""
    
    description: Optional[str] = None
    """A description of the relationship (optional)."""
    
    description_embedding: Optional[List[float]] = None
    """The semantic embedding for the relationship description (optional)."""
    
    text_unit_ids: Optional[List[str]] = None
    """List of text unit IDs in which the relationship appears (optional)."""

    document_ids: Optional[List[str]] = None
    """List of document IDs in which the relationship appears (optional)."""
    
    attributes: Optional[Dict[str, Any]] = None
    """Additional attributes associated with the relationship (optional). To be included in the search prompt"""