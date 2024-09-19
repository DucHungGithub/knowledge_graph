from typing import Any, Dict, Iterable, List
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from graph.claims.graph_claims import ClaimExtractor
from verbs.covariates.typing import Covariate, CovariateExtractionResult
import config as defs
from llm import load_openai_llm
import config as defs

logger = logging.getLogger(__name__)


async def run_gi(
    input: List[Document],
    entity_types: List[str],
    resolved_entities_map: Dict[str, str],
    args: Dict[str, Any]
) -> CovariateExtractionResult:
    model_config = args.get("llm", defs.MODEL_CONFIG)
    temperature = args.get("temperature", defs.TEMPERATURE)
    llm = load_openai_llm(model=model_config, temperature=temperature)
    return await run_covariates(llm=llm, texts=input, entity_types=entity_types, resolved_entities_map=resolved_entities_map, args=args)
    

async def run_covariates(
    llm: BaseChatModel,
    texts: List[Document],
    entity_types: List[str],
    resolved_entities_map: Dict[str, str],
    args: Dict[str, Any]
) -> CovariateExtractionResult:
    
    # Extraction config
    extraction_prompt = args.get("extraction_prompt", None)
    max_gleanings = args.get("max_gleanings", defs.CLAIM_MAX_GLEANINGS)
    tuple_delimiter = args.get("tuple_delimiter", None)
    record_delimiter = args.get("record_delimiter", None)
    completion_delimiter = args.get("completion_delimiter", None)
    encoding_model = args.get("encoding_name", defs.ENCODING_MODEL)
    
    extractor = ClaimExtractor(
        llm=llm,
        extraction_prompt=extraction_prompt,
        max_gleanings=max_gleanings,
        encoding_model=encoding_model
    )
    
    claim_description =args.get("claim_description")
    if claim_description is None:
        logger.warning(f"claim_description is required for claim extraction")
        claim_description = defs.CLAIM_DESCRIPTION
    
    # texts = [texts] if isinstance(texts, str) else texts
    
    results = await extractor.invoke({
        "input_text": texts,
        "entity_specs": entity_types,
        "resolved_entities": resolved_entities_map,
        "claim_description": claim_description,
        "tuple_delimiter": tuple_delimiter,
        "record_delimiter": record_delimiter,
        "completion_delimiter": completion_delimiter
    })
    
    claim_data= results.output
    
    return CovariateExtractionResult(covariate_data=[
        create_covariate(item) for item in claim_data
    ])
    


def create_covariate(item: Dict[str, Any]) -> Covariate:
    source_text = item.get("source_text")
    if source_text is not None and isinstance(source_text, str):
        source_text = [source_text]
    return Covariate(
        subject_id=item.get("subject_id"),
        subject_type=item.get("subject_type"),
        object_id=item.get("object_id"),
        object_type=item.get("object_type"),
        type=item.get("type"),
        status=item.get("status"),
        start_date=item.get("start_date"),
        end_date=item.get("end_date"),
        description=item.get("description"),
        source_text=source_text,
        doc_id=item.get("doc_id"),
        record_id=item.get("record_id"),
        id=item.get("id")
    )