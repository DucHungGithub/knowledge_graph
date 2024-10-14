import asyncio
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from extract_elements_prompts.defaults import DEFAULT_TASK
from prompts.persona import GENERATE_PERSONA_PROMPT

async def generate_persona(llm: BaseChatModel, domain: str, task: str = DEFAULT_TASK) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - domain (str): The domain to generate a persona for
    - task (str): The task to generate a persona for. Default is DEFAULT_TASK
    """
    
    formatted_task = task.format(domain=domain)
    
    persona_prompt = GENERATE_PERSONA_PROMPT.format(sample_task=formatted_task)
    
    response = llm.invoke(persona_prompt)
    
    return str(response.content)

