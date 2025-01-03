# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
import pandas as pd

from graph.community_reports.sort_context import sort_context
from utils import list_of_token

import graph.community_reports.schemas as schemas

def build_mixed_context(context: List[Dict], max_tokens: int) -> str:
    """
    Build parent context by concatenating all sub-communities' contexts.

    If the context exceeds the limit, we use sub-community reports instead.
    """
    sorted_context = sorted(
        context, key=lambda x: x[schemas.CONTEXT_SIZE], reverse=True
    )

    # replace local context with sub-community reports, starting from the biggest sub-community
    substitute_reports = []
    final_local_contexts = []
    exceeded_limit = True
    context_string = ""

    for idx, sub_community_context in enumerate(sorted_context):
        if exceeded_limit:
            if sub_community_context[schemas.FULL_CONTENT]:
                substitute_reports.append({
                    schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                    schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
                })
            else:
                # this sub-community has no report, so we will use its local context
                final_local_contexts.extend(sub_community_context[schemas.ALL_CONTEXT])
                continue

            # add local context for the remaining sub-communities
            remaining_local_context = []
            for rid in range(idx + 1, len(sorted_context)):
                remaining_local_context.extend(sorted_context[rid][schemas.ALL_CONTEXT])
            new_context_string = sort_context(
                local_context=remaining_local_context + final_local_contexts,
                sub_community_reports=substitute_reports,
            )
            if len(list_of_token(new_context_string)) <= max_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    if exceeded_limit:
        # if all sub-community reports exceed the limit, we add reports until context is full
        substitute_reports = []
        for sub_community_context in sorted_context:
            substitute_reports.append({
                schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
            })
            new_context_string = pd.DataFrame(substitute_reports).to_csv(
                index=False, sep=","
            )
            if len(list_of_token(new_context_string)) > max_tokens:
                break

            context_string = new_context_string
    return context_string
