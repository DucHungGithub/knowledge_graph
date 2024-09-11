from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    name: str = Field(description="The name of LLM model")
    model: str = Field(description="The type of LLM model")
    temperature: float = Field(description="The temperature of LLM")
    supports_json: bool = Field(default=False, description="Check the LLM model support json format")