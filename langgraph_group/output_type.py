from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class Queries(BaseModel):
    """List of search queries"""
    queries: List[str] = Field(
        description="List of the generated search queries"
    )
    
    
class OutputResponse(BaseModel):
    is_enough: str = Field(
        description="A value indicating whether the current context has sufficient information to answer the question. "
                    "Returns `yes` if the context is enough, otherwise returns `no`."
    )
    
    reason_why: str = Field(
        description="Provide the reason for your answer."
    )
    
    
class RewriteResponse(BaseModel):
    new_query: str = Field(
        description="A string representing the suggested query or question to gather additional information needed. "
                    "this field should provide a specific query or instruction to obtain the missing data required to accurately answer the question."
    )
    reason_why: str = Field(
        description="Provide the reason for your answer."
    )