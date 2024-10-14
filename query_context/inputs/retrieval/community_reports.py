import json
from typing import Any, List, cast

import pydgraph
import pandas as pd

from models.community_report import CommunityReport
from models.entity import Entity


def get_candidate_communities(
    selected_reports: List[CommunityReport],
    include_community_rank: bool = False,
    use_community_summary: bool = False
) -> pd.DataFrame:
    """Get all communities that are related to selected entities."""
                    
    
    return to_community_report_dataframe(
        reports=selected_reports,
        include_community_rank=include_community_rank,
        use_community_summary=use_community_summary
    )

def get_candidate_community_reports(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
) -> List[CommunityReport]:
    """Get all communities that are related to selected entities."""
    selected_reports = []
    txn = client.txn()
    
    selected_community_ids = [
        entity.community_ids for entity in selected_entities if entity.community_ids
    ]
    
    selected_community_ids = [
        item for sublist in selected_community_ids for item in sublist
    ]
    
    selected_community_ids = list(set(selected_community_ids))
    
    try:
        for com in selected_community_ids:
            query = f"""{{
                getCommunityReports(func: type(CommunityReport)) @filter(eq(community_id, {com})){{
                    id
                    short_id
                    title
                    community_id
                    summary
                    full_content
                    rank
                    summary_embedding
                    full_content_embedding
                    attributes
                }}
            }}
            """
            res = txn.query(query=query)
            ppl = json.loads(res.json)
            
            reports = ppl["getCommunityReports"]
            for report in reports:
                selected_reports.append(CommunityReport(**report))
            
            
    finally:
        txn.discard()
                    
    
    return selected_reports
    

def to_community_report_dataframe(
    reports: List[CommunityReport],
    include_community_rank: bool = False,
    use_community_summary: bool = False
) -> pd.DataFrame:
    """Convert a list of communities to a pandas dataframe."""
    if len(reports) == 0:
        return pd.DataFrame()
    
    # add header
    header = ["id", "title"]
    attribute_cols = list(reports[0].attributes.keys()) if reports[0].attributes else []
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    header.append("summary" if use_community_summary else "content")
    if include_community_rank:
        header.append("rank")
        
    records = []
    for report in reports:
        new_record = [
            report.short_id if report.short_id else "",
            report.title,
            *[
                str(report.attributes.get(field, ""))
                if report.attributes and report.attributes.get(field)
                else ""
                for field in attribute_cols
            ]
        ]
        new_record.append(
            report.summary if use_community_summary else report.full_content
        )
        if include_community_rank:
            new_record.append(str(report.rank))
        records.append(new_record)
        
    return pd.DataFrame(records, columns=cast(Any, header))