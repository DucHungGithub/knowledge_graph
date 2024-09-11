
from pathlib import Path
from langchain_core.language_models import BaseChatModel
import logging

from api import generate_indexing_prompts
from configs import GeneralConfig
from extract_elements_prompts.entity_extraction_prompt import ENTITY_EXTRACTION_FILENAME
from extract_elements_prompts.entity_summarization_prompt import ENTITY_SUMMARIZATION_FILENAME
from extract_elements_prompts.community_report_summarization import COMMUNITY_REPORT_SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)

async def prompt_tune(
    config: GeneralConfig,
    llm: BaseChatModel,
    domain: str | None = None,
    language: str | None = None,
    output: str = "output_prompts"
):
    prompts = await generate_indexing_prompts(
        config=config,
        llm=llm,
        domain=domain,
        language=language
    )
    
    output_path = Path(output)
    if output_path:
        logger.info(f"Writing prompts to {output_path}", exc_info=True)
        output_path.mkdir(parents=True, exist_ok=True)
        entity_extraction_prompt_path = output_path / ENTITY_EXTRACTION_FILENAME
        entity_summarization_prompt_path = output_path / ENTITY_SUMMARIZATION_FILENAME
        community_summarization_prompt_path = output_path / COMMUNITY_REPORT_SUMMARIZATION_PROMPT
        
        with entity_extraction_prompt_path.open("wb") as file:
            file.write(prompts[0].encode(encoding="utf-8", errors="strict"))
            
        with entity_summarization_prompt_path.open("wb") as file:
            file.write(prompts[1].encode(encoding="utf-8", errors="strict"))
            
        with community_summarization_prompt_path.open("wb") as file:
            file.write(prompts[2].encode(encoding="utf-8", errors="strict"))