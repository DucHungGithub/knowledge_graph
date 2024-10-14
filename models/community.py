from typing import Any, Dict, List, Optional
from models.base import Name


class Community(Name):
    
    level: str = ""
    """Community level."""

    entity_ids: Optional[List[str]] = None
    """List of entity IDs related to the community (optional)."""

    relationship_ids: Optional[List[str]] = None
    """List of relationship IDs related to the community (optional)."""

    covariate_ids: Optional[Dict[str, List[str]]] = None
    """Dictionary of different types of covariates related to the community (optional), e.g. claims"""

    attributes: Optional[Dict[str, Any]] = None
    """A dictionary of additional attributes associated with the community (optional). To be included in the search prompt."""