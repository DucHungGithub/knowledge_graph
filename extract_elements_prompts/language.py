from typing import List


from langchain_core.language_models import BaseChatModel

from prompts.language import DETECT_LANGUAGE_PROMPT

async def detect_language(
    llm: BaseChatModel,
    docs: str | List[str]
) -> str:
    """Detect input language to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - docs (str | list[str]): The docs to detect language from

    Returns
    -------
    - str: The detected language.
    """
    
    docs_str = " ".join(docs) if isinstance(docs, List) else docs
    
    language_prompt = DETECT_LANGUAGE_PROMPT.format(input_text=docs_str)
    
    response = llm.invoke(language_prompt)
    return str(response.content)