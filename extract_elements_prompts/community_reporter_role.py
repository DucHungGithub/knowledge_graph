from typing import List

from langchain_core.language_models import BaseChatModel

from prompts.community_reporter_role import GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT


async def generate_community_reporter_role(
    llm: BaseChatModel,
    domain: str,
    persona: str,
    docs: str | List[str]
) -> str:
    docs_str = " ".join(docs) if isinstance(docs, List) else docs
    report_role_prompt = GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT.format(
        persona=persona,
        domain=domain,
        input_text=docs_str
    )
    
    response = llm.invoke(report_role_prompt)
    
    return str(response.content).strip()
    
    