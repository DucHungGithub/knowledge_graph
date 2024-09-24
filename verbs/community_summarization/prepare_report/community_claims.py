# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from typing import List
import pandas as pd
from graph.community_reports.schemas import (
    CLAIM_DESCRIPTION,
    CLAIM_DETAILS,
    CLAIM_ID,
    CLAIM_STATUS,
    CLAIM_SUBJECT,
    CLAIM_TYPE
)
from verbs.covariates.typing import CovariateExtractionResult

_MISSING_DESCRIPTION = "No Description"

def prepare_community_reports_claims(
    covariate_data: CovariateExtractionResult,
    to: str = CLAIM_DETAILS,
    id_column: str = CLAIM_ID,
    description_column: str = CLAIM_DESCRIPTION,
    subject_column: str = CLAIM_SUBJECT,
    type_column: str = CLAIM_TYPE,
    status_column: str = CLAIM_STATUS,
    **kwargs
) -> pd.DataFrame:
    claim_df = covariate_data.to_dataframe()
    
    if claim_df.empty:
        return pd.DataFrame()
    claim_df = claim_df.fillna(value={description_column: _MISSING_DESCRIPTION})
    
    claim_df[to] = claim_df.apply(
        lambda x: {
            id_column: x[id_column],
            subject_column: x[subject_column],
            type_column: x[type_column],
            status_column: x[status_column],
            description_column: x[description_column]
        },
        axis=1
    )
    
    return claim_df