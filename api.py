import logging
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel

from configs import GeneralConfig

from .text_unit import load_docs_in_chunks 
from extract_elements_prompts import (
    MAX_TOKEN_COUNT,
    create_community_summarization_prompt,
    create_entity_extraction_prompt,
    create_entity_summarization_prompt,
    detect_language,
    generate_community_report_rating,
    generate_community_reporter_role,
    generate_domain,
    generate_entity_relationship_examples,
    generate_entity_types,
    generate_persona,
)

logger = logging.getLogger(__name__)

async def generate_indexing_prompts(
    config: GeneralConfig,
    llm: BaseChatModel,
    domain: str | None = None,
    language: str | None = None,
    skip_entity_types: bool = False,
) -> Tuple[str, str, str]:
    """Generate indexing prompts.

    Parameters
    ----------
    - config: The GraphRag configuration.
    - domain: The domain to map the input documents to.
    - language: The language to use for the prompts.
    - skip_entity_types: Skip generating entity types.

    Returns
    -------
    tuple[str, str, str]: entity extraction prompt, entity summarization prompt, community summarization prompt
    """
    # Retrive documents
    doc_list: List[int] = load_docs_in_chunks(
        input_dir=config.input.input_dir,
        chunk_overlap=config.input.chunk_overlap,
        chunk_size=config.input.chunk_size,
    )
    
    if not domain:
        logger.info("Generating domain...", exc_info=True)
        domain = await generate_domain(llm=llm, docs=doc_list)
        logger.info(f"Generated domain: {domain}")
        
    if not language:
        logger.info("Generating language...", exc_info=True)
        language = await detect_language(llm=llm, docs=doc_list)
        logger.info(f"Detected language: {language}", exc_info=True)
        
        
    #---------Persona-----------
    logger.info("Generating persona...", exc_info=True)
    persona = await generate_persona(llm=llm, domain=domain)
    logger.info(f"Generated persona: {persona}", exc_info=True)
    
    
    #---------comminuty_report_raking-----------
    logger.info("Generating community report raking description...", exc_info=True)
    comminuty_report_raking = await generate_community_report_rating(
        llm=llm, domain=domain, persona=persona, docs=doc_list
    )
    logger.info(f"Generated community report raking description: {comminuty_report_raking}", exc_info=True)
    
    
    #---------entity_types-----------
    entity_types = None
    if not skip_entity_types:
        logger.info("Generating entity types...", exc_info=True)
        entity_types = await generate_entity_types(
            llm=llm,
            domain=domain,
            persona=persona,
            docs=doc_list,
            json_mode=config.llm.supports_json or False
        )
        logger.info(f"Generated entity types: {entity_types}", exc_info=True)
        
        
    #---------entity relationship-----------
    logger.info("Generating entity relationship examples...", exc_info=True)
    examples = await generate_entity_relationship_examples(
        llm=llm,
        entity_types=entity_types,
        language=language,
        persona=persona,
        docs=doc_list,
        json_mode=config.llm.supports_json
    )
    logger.info(f"Generated entity relationship examples: {examples}", exc_info=True)
    
    
    #---------entity extraction-----------
    logger.info("Generating Entity Extraction...", exc_info=True)
    entity_extraction_prompt = await create_entity_extraction_prompt(
        entity_types=entity_types,
        docs=doc_list,
        examples=examples,
        language=language,
        max_token_count=MAX_TOKEN_COUNT,
        min_examples_required=config.min_examples_required,
        encoding_model=config.encoding_model,
        json_mode=config.llm.supports_json
    )
    logger.info(f"Generated entity extraction prompt {entity_extraction_prompt}", exc_info=True)
    
    
    #---------entity summarization-----------
    logger.info("Generating entity summarization prompt...", exc_info=True)
    entity_summarization_prompt = await create_entity_summarization_prompt(
        persona=persona,
        language=language
    )
    logger.info(f"Generated entity summarization prompt: {entity_summarization_prompt}", exc_info=True)
    
    
    #---------Generate community report role-----------
    logger.info("Generating community report role...", exc_info=True)
    community_report_role = await generate_community_reporter_role(
        llm=llm,
        docs=doc_list,
        domain=domain,
        persona=persona
    )
    logger.info(f"Generated community report role: {community_report_role}", exc_info=True)
    
    
    #---------Generate community summarization-----------
    logger.info("Generating community summarization...", exc_info=True)
    community_summarization_prompt = create_community_summarization_prompt(
        persona=persona,
        language=language,
        report_rating_description=comminuty_report_raking,
        role=community_report_role
    )
    
    return (
        entity_extraction_prompt,
        entity_summarization_prompt,
        community_summarization_prompt
    )