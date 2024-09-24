from pathlib import Path
from pydantic import BaseModel, Field

class InputConfig(BaseModel):
    input_dir: str | Path = Field(description="The path to the directory path of input folder")
    chunk_size: int = Field(description="The chunk size of the input text")
    chunk_overlap: int = Field(description="The chunk overlap of the input text")