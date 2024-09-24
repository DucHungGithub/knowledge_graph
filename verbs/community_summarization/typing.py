from typing import Any, Awaitable, Callable, Dict, List
from pydantic import BaseModel

class Finding(BaseModel):
    summary: str
    explanation: str
    
class CommunityReport(BaseModel):
    id: str
    community: str | int
    title: str
    summary: str
    full_content: str
    full_content_json: str
    rank: float
    level: int
    rank_explanation: str
    findings: List[Finding]
    
    
CommunityReportsStrategy = Callable[
    [
        str | int,
        str,
        int,
        Dict[str, Any]
    ],
    Awaitable[CommunityReport | None],
]