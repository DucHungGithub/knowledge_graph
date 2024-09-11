from pathlib import Path
from pydantic import BaseModel, Field

from configs.input import InputConfig
from .llm import LLMConfig

class GeneralConfig(BaseModel):
    llm: LLMConfig = Field(description="LLM model config")
    
    input: InputConfig = Field(description="Input config")
    
    encoding_model: str = Field(default="cl100k_base", description="Encoding model")
    
    max_messages: int = Field(default=5, description="The maximum number of messages")
    
    min_examples_required: int = Field(default=2, description="The minimum number of messages")
    