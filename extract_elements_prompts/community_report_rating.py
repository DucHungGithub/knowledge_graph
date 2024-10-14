from typing import List

from langchain_core.language_models import BaseChatModel

from prompts.community_report_rating import GENERATE_REPORT_RATING_PROMPT


async def generate_community_report_rating(
    llm: BaseChatModel,
    domain: str,
    persona: str,
    docs: str | List[str]
) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - domain (str): The domain to generate a rating for
    - persona (str): The persona to generate a rating for for
    - docs (str | list[str]): Documents used to contextualize the rating

    Returns
    -------
    - str: The generated rating description prompt response.
    """
    docs_str = " ".join(docs) if isinstance(docs, List) else docs
    
    report_prompt = GENERATE_REPORT_RATING_PROMPT.format(
        domain=domain,
        persona=persona,
        input_text=docs_str
    )
    
    response = llm.invoke(report_prompt)
    
    return str(response.content).strip()