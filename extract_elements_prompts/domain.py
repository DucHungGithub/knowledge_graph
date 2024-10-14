from typing import List

from langchain_core.language_models import BaseChatModel

from prompts.domain import GENERATE_DOMAIN_PROMPT


async def generate_domain(llm: BaseChatModel, docs: str | List[str]) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - docs (str | list[str]): The domain to generate a persona for

    Returns
    -------
    - str: The generated domain prompt response.
    """

    docs_str = " ".join(docs) if isinstance(docs, List) else docs
    domain_prompt = GENERATE_DOMAIN_PROMPT.format(input_text=docs_str)
    
    response = llm.invoke(domain_prompt)
    return str(response.content)