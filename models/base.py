from typing import Optional
from pydantic import BaseModel

class Identify(BaseModel):
    id: str
    """The ID of the item."""
    
    short_id: Optional[str] = None
    """Human readable ID used to refer to this community in prompts or texts displayed to users, such as in a report text (optional)."""
    
    
class Name(Identify):
    title: str
    """The name/title of the item."""