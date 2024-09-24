import pandas as pd
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from utils.uuid import gen_uuid
from pydantic import BaseModel, Field


class Covariate(BaseModel):
    """Covariate class"""
    
    covariate_type: Optional[str] = None
    subject_id: Optional[str] = None
    subject_type: Optional[str] = None
    object_id: Optional[str] = None
    object_type: Optional[str] = None
    type: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    source_text: Optional[List[str]] = None
    doc_id: Optional[str] = None
    record_id: Optional[int] = None
    id: Optional[str] = None
    
    


class CovariateExtractionResult(BaseModel):
    covariate_data: List[Covariate]
    
    def to_dataframe(self) -> pd.DataFrame:
        for cov in self.covariate_data:
            cov.id = gen_uuid()
        data = []
        for idx, cov in enumerate(self.covariate_data):
            cov_dict = cov.dict()
            cov_dict['human_readable_id'] = idx
            data.append(cov_dict)
        
        return pd.DataFrame(data)
    
    
CovariateExtractStrategy = Callable[
    [
        Iterable[str],
        List[str],
        Dict[str, str],
        Dict[str, Any]
    ],
    Awaitable[CovariateExtractionResult]
]