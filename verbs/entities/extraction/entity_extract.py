import logging
from enum import Enum
from typing import Any, Dict, List, cast

from langchain_core.documents import Document



from verbs.entities.extraction.typing import EntityExtractStrategy, EntityExtractionResult


logger = logging.getLogger(__name__)


class ExtractEntityStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"
    graph_intelligence_json = "graph_intelligence_json"
    nltk = "nltk"
    
    def __repr__(self) -> str:
        return f"{self.value}"
    
DEFAULT_ENTITY_TYPES = ["ORGANIZATION", "PERSON", "GEO", "EVENT"]

async def entity_extract(
    docs: List[Document],
    args: Dict[str, Any],
) -> EntityExtractionResult:
    entity_types = args.get("entity_types", None)
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
        
    args = args or {}
    strategy_exec = load_strategy(args.get("ENTITY_EXTRACT_TYPE", ExtractEntityStrategyType.graph_intelligence))
    
    results = await strategy_exec(docs, entity_types, args)
    return results
    
    

def load_strategy(
    strategy_type: ExtractEntityStrategyType
) -> EntityExtractStrategy:
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:  
            from verbs.entities.extraction.graph_intelligence.run import run_gi
            return run_gi
        
        case ExtractEntityStrategyType.nltk:
            from verbs.entities.extraction.graph_nltk.run import run_nltk
            return run_nltk
        
        case _:
            raise ValueError(f"Unknow strategy: {strategy_type}")
            