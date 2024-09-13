

from typing import Any, Dict, List, Optional
from models.base import Identify


class Covariate(Identify):
    
    subject_id: Optional[str] = None
    """The subject id."""
    
    subject_type: Optional[str] = "entity"
    """The subject type."""
    
    covariate_type: Optional[str] = "claim"
    """The covariate type"""
    
    text_unit_ids: Optional[List[str]] = None
    """List of text unit IDs in which the covariate info appears (optional)."""
    
    document_ids: Optional[List[str]] = None
    """List of document IDs in which the covariate info appears (optional)."""
    
    attributes: Optional[Dict[str, Any]] = None