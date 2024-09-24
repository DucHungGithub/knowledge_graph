from typing import Annotated, List, Literal, Sequence, Tuple, TypedDict
import operator

from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel

from models.entity import Entity
from models.relationship import Relationship

members = ["EntityExtract", "RelationshipExtract"]
options = Literal["FINISH", "EntityExtract", "RelationshipExtract"]

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    entity_vectorstore: VectorStore
    entities: Tuple[str, str, str]
    rels: List[Relationship]
    k: int
    next: str
    
    

class RouteResponse(BaseModel):
    next: options
    
    
