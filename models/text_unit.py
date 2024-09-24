from typing import Any, Dict, List, Optional

from models.base import Identify

class TextUnit(Identify):
 
    text: str
    """The text of the unit."""
    
    text_embedding: Optional[List[float]]
    """The text embedding for the text unit (optional)."""
    
    entity_ids: Optional[List[str]]
    """List of entity IDs related to the text unit (optional)."""
    
    relationship_ids: Optional[List[str]]
    """List of relationship IDs related to the text unit (optional)."""
    
    covariate_ids: Optional[Dict[str, List[str]]]
    """Dictionary of different types of covariates related to the text unit (optional)."""
    
    n_tokens: Optional[int]
    """The number of tokens in the text (optional)."""
    
    document_ids: Optional[List[str]]
    """List of document IDs in which the text unit appears (optional)."""
    
    attributes: Optional[Dict[str, Any]]
    """A dictionary of additional attributes associated with the text unit (optional)."""
    