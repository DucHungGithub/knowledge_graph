from typing import Any, Dict, List, Optional

from models.base import Name


class Entity(Name):

    type: Optional[str]
    """Type of the entity (can be any string, optional)."""
    
    description: Optional[str]
    """Description of the entity (optional)."""

    # description_embedding: Optional[List[float]]
    # """The semantic (i.e. text) embedding of the entity (optional)."""
    
    name_embedding: Optional[List[float]]
    """The semantic (i.e. text) embedding of the entity (optional)."""
    
    graph_embedding: Optional[List[float]]
    """The graph embedding of the entity, likely from node2vec (optional)."""
    
    community_ids: Optional[List[str]]
    """The community IDs of the entity (optional)."""
    
    text_unit_ids: Optional[List[str]]
    """List of text unit IDs in which the entity appears (optional)."""
    
    document_ids: Optional[List[str]]
    """List of document IDs in which the entity appears (optional)."""
    
    rank: Optional[int] = 1
    """Rank of the entity, used for sorting (optional). Higher rank indicates more important entity. This can be based on centrality or other metrics."""
    
    attributes: Optional[Dict[str, Any]]
    """Additional attributes associated with the entity (optional), e.g. start time, end time, etc. To be included in the search prompt."""