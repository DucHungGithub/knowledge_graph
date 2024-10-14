from typing import Any, Awaitable, Callable, Dict, List, Tuple
from pydantic import BaseModel, Field

class SummarizedDescriptionResult(BaseModel):
    """Entity summarization result class definition."""
    items: str | Tuple[str, str]
    description: str
    

SummarizationStrategy = Callable[
    [
        str | Tuple[str, str],
        List[str],
        Dict[str, Any]
    ],
    Awaitable[SummarizedDescriptionResult]
]