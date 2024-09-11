import logging
from enum import Enum
from typing import Any, Dict, List

from verbs.covariates.graph_intelligence.run import run_gi
from verbs.covariates.typing import CovariateExtractStrategy, CovariateExtractionResult


logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

class ExtractClaimsStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"
    
    def __repr__(self) -> str:
        return f"{self.value}"


async def extract_covariates(
    texts: List[str],
    entity_types: List[str] | None = None,
    resolved_entities_map: Dict[str, Any] = None,
    strategy: Dict[str, Any] | None = None,
    **kwargs
) -> CovariateExtractionResult:
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    
    resolved_entities_map = {}
    
    strategy = strategy or {}
    
    strategy_exec = load_strategy(
        strategy.get("type", ExtractClaimsStrategyType.graph_intelligence)
    )
    
    config = {**strategy}
    
    if resolved_entities_map is None:
        resolved_entities_map = {}
    
    results = await strategy_exec(texts, entity_types, resolved_entities_map, config)
    
    return results

def load_strategy(strategy_type: ExtractClaimsStrategyType) -> CovariateExtractStrategy:
    match strategy_type:
        case ExtractClaimsStrategyType.graph_intelligence:
            from .graph_intelligence.run import run_gi
            return run_gi
        case _:
            raise ValueError(f"Unknow claims type: {strategy_type}")