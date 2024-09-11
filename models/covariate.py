

from typing import Any, Dict, List, Optional
from models.base import Identify


class Covariate(Identify):
    
    subject_id: str
    """The subject id."""
    
    subject_type: str = "entity"
    """The subject type."""
    
    covariate_type: str = "claim"
    """The covariate type"""
    
    text_unit_ids: Optional[List[str]]
    """List of text unit IDs in which the covariate info appears (optional)."""
    
    document_ids: Optional[List[str]]
    """List of document IDs in which the covariate info appears (optional)."""
    
    attributes: Optional[Dict[str, Any]]