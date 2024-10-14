from typing import Any, Dict, List, Tuple
from langchain_core.language_models import BaseChatModel

from graph.summarization.graph_summarization import SummarizeExtractor
from llm import load_openai_llm
from verbs.entities.summarization.typing import SummarizedDescriptionResult

import config as defs


async def run_gi(
    items: str | Tuple[str, str],
    descriptions: List[str],
    args: Dict[str, str]
) -> SummarizedDescriptionResult:
    model_config = args.get("llm", defs.MODEL_CONFIG)
    temperature = args.get("temperature",defs.TEMPERATURE)
    llm = load_openai_llm(model=model_config, temperature=temperature)
    return await run_summarize_description(llm=llm, items=items, descriptions=descriptions,args=args)



async def run_summarize_description(
    llm: BaseChatModel,
    items: str | Tuple[str, str],
    descriptions: List[str],
    args: Dict[str, Any]
) -> SummarizedDescriptionResult:
    """Run the entity extraction chain"""
    #Extraction Arguments
    summarize_prompt = args.get("summarize_prompt", None)
    entity_name_key = args.get("entity_name_key", "entity_name")
    input_descriptions_key = args.get("input_descriptions_key", "description_list")
    max_tokens = args.get("max_tokens", None)
    max_summary_length=args.get("max_summary_length", None)
    
    
    extractor = SummarizeExtractor(
        llm=llm,
        entity_name_key=entity_name_key,
        input_description_key=input_descriptions_key,
        summarization_prompt=summarize_prompt,
        max_input_tokens=max_tokens,
        max_summary_length=max_summary_length
    )
    
    result = await extractor.invoke(
        items=items,
        descriptions=descriptions
    )
    
    return result