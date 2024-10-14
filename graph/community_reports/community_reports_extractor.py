from typing import Any, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

from graph.community_reports.prompts import COMMUNITY_REPORT_PROMPT

class Finding(BaseModel):
    summary: str = Field(description="A short summary of finding.")
    explanation: str = Field(description="Multiple paragraphs of explanatory text grounded according to the grounding rules following the summary.")


class CommunityExtractorOutput(BaseModel):
    title: str = Field(description="Community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.")
    summary: str = Field(description="An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.")
    rating: float = Field(description="A float score between 0-10 that represents the severity of IMPACT posed by entities within the community.")
    rating_explanation: str = Field(description="A single sentence explanation of the IMPACT severity rating.")
    findings: List[Finding] = Field(description="A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.")

class CommunityReportsResult(BaseModel):
    output: str
    structured_output: Dict[str, Any]


class CommunityReportsExtractor:
    
    _llm: BaseChatModel
    _input_text_key: str
    _extraction_prompt: str
    _output_formatter_prompt: str
    _max_report_length: int
    
    def __init__(
        self,
        llm: BaseChatModel,
        input_text_key: str | None = None,
        extraction_prompt: str | None = None,
        max_report_length: int | None = None
    ):
        self._llm = llm
        self._input_text_key = input_text_key or "input_text"
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._max_report_length = max_report_length or 1500
        
    async def invoke(
        self,
        inputs: Dict[str, Any]
    ):
        
        template = ChatPromptTemplate.from_messages([
            ("system", self._extraction_prompt)
        ])
        
        prompt = template.format(**{self._input_text_key: inputs[self._input_text_key]})
        
        self._llm.max_tokens=self._max_report_length
        structured_llm = self._llm.with_structured_output(CommunityExtractorOutput)
        response = structured_llm.invoke(prompt)
        
        text_output = self._get_text_output(response)
        return CommunityReportsResult(
            structured_output=response.dict(),
            output=text_output
        )
        
        
    def _get_text_output(
        self,
        output: BaseModel
    ) -> str:
        title = output.title
        summary = output.summary
        findings = output.findings
        
        report_sections = "\n\n".join(
            f"## {f.summary}\n\n{f.explanation}" for f in findings
        )
        
        return f"# {title}\n\n{summary}\n\n{report_sections}"