import logging
import random
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd


from models.community_report import CommunityReport
from models.entity import Entity


import colorlog

from external_utils.token import list_of_token

# Set up basic logging configuration with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# Get the logger and add the handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.addHandler(handler)

NO_COMMUNITY_RECORDS_WARNING: str = (
    "Warning: No community records added when building community context."
)


def build_community_context(
    community_reports: List[CommunityReport],
    entities: Optional[List[Entity]] = None,
    token_encoder: Optional[str] = None,
    use_community_summary: bool = True,
    column_delimiter: str = "|",
    shuffle_data: bool = True,
    include_community_rank: bool = False,
    min_community_rank: int = 0,
    community_rank_name: str = "rank",
    include_community_weight: bool = True,
    community_weight_name: str = "occurrence weight",
    normalize_community_weight: bool = True,
    max_tokens: int = 2500,
    single_batch: bool = True,
    context_name: str = "Reports",
    random_state: int = 6969,
) -> Tuple[str | List[str], Dict[str, pd.DataFrame]]:
    """
    Prepare community report data table as context data for system prompt.

    If entities are provided, the community weight is calculated as the count of text units associated with entities within the community.

    The calculated weight is added as an attribute to the community reports and added to the context data table.
    """
    
    is_community_weights = (
        entities
        and len(community_reports) > 0
        and include_community_weight
        and (
            community_reports[0].attributes is None
            or community_weight_name not in community_reports[0].attributes
        )
    )

    
    if is_community_weights:
        logger.info("Computing community weights...", exc_info=True)
        community_reports = compute_community_weights(
            community_reports=community_reports,
            entities=entities,
            weight_attribute=community_weight_name,
            normalize=normalize_community_weight
        )
        

    selected_reports = [report for report in community_reports if is_include(report, min_community_rank)]
    


    if selected_reports is None or len(selected_reports) == 0:
        return ([], {})
    
    if shuffle_data:
        random.seed(random_state)
        random.shuffle(selected_reports)
        
    
        
    attributes = (
        list(community_reports[0].attributes.keys())
        if community_reports[0].attributes
        else []
    )
    
    header = get_header(
        attributes=attributes,
        include_community_weight=include_community_weight,
        community_weight_name=community_weight_name,
        use_community_summary=use_community_summary,
        include_community_rank=include_community_rank,
        community_rank_name=community_rank_name
    )
    
    all_context_text: List[str] = []
    all_context_records: List[pd.DataFrame] = []
    
    batch_text = ""
    batch_tokens = 0
    batch_records = []
    
    def init_batch() -> None:
        nonlocal batch_text, batch_tokens, batch_records
        batch_text = (
            f"-----{context_name}-----" + "\n" + column_delimiter.join(header) + "\n"
        )
        batch_tokens = len(list_of_token(batch_text, token_encoder))
        batch_records = []
    
    def cut_batch() -> None:
        """Convert the current context records to pandas DataFrame and sort by weight and rank if exist"""
        record_df = convert_report_context_to_df(
            context_records=batch_records,
            header=header,
            weight_column=(
                community_weight_name if entities and include_community_weight else None
            ),
            rank_column=community_rank_name if include_community_rank else None,
        )
        if len(record_df) == 0:
            return
        current_context_text = record_df.to_csv(index=False, sep=column_delimiter)
        if not all_context_text and single_batch:
            current_context_text = f"-----{context_name}-----\n{current_context_text}"

        all_context_text.append(current_context_text)
        all_context_records.append(record_df)
        
    init_batch()
    
    for report in selected_reports:
        
        new_context_text, new_context = report_context_text(
            report=report, 
            attributes=attributes,
            use_community_summary=use_community_summary,
            include_community_rank=include_community_rank,
            column_delimiter=column_delimiter
        )
    
        new_tokens = len(list_of_token(new_context_text, token_encoder))
        
        
        if batch_tokens + new_tokens > max_tokens:
            
            # Add the current batch to the context data and start a new batch if we are in multi-batch mode
            cut_batch()
            if single_batch:
                break
            init_batch()
            
        batch_text += new_context_text
        batch_tokens += new_tokens
        batch_records.append(new_context)
        
    # Add the last batch if it has not been added
    if batch_text not in all_context_text:
        cut_batch()
    
    if len(all_context_records) == 0:
        logger.warning(NO_COMMUNITY_RECORDS_WARNING, exc_info=True)
        return ([], {})
    
    
    return all_context_text, {
        context_name.lower(): pd.concat(all_context_records, ignore_index=True)
    }
        
    
def compute_community_weights(
    community_reports: List[CommunityReport],
    entities: Optional[List[Entity]],
    weight_attribute: str = "occurrence",
    normalize: bool = True,
) -> List[CommunityReport]:
    """Calculate a community's weight as count of text units associated with entities within the community."""
    if not entities:
        return community_reports
    
    community_text_units = {}
    for entity in entities:
        if entity.community_ids:
            for community_id in entity.community_ids:
                if community_id not in community_text_units:
                    community_text_units[community_id] = []
                community_text_units[community_id].extend(entity.text_unit_ids)
    for report in community_reports:
        if not report.attributes:
            report.attributes = {}
        report.attributes[weight_attribute] = len(
            set(community_text_units.get(report.community_id, []))
        )
        
    if normalize:
        # normalize by max weight
        all_weights = [
            report.attributes[weight_attribute]
            for report in community_reports
            if report.attributes
        ]
        max_weight = max(all_weights)
        for report in community_reports:
            if report.attributes:
                report.attributes[weight_attribute] = (
                    report.attributes[weight_attribute] / max_weight
                )
    return community_reports








def rank_report_context(
    report_df: pd.DataFrame,
    weight_column: Optional[str] = "occurrence weight",
    rank_column: Optional[str] = "rank"
) -> pd.DataFrame:
    """Sort report context by community weight and rank if exist."""
    rank_attributes: List[str] = []
    
    if weight_column and weight_column in report_df.columns:
        rank_attributes.append(weight_column)
        original_count = len(report_df)
        report_df[weight_column] = pd.to_numeric(report_df[weight_column], errors='coerce')
        report_df = report_df.dropna(subset=[weight_column])
        dropped_count = original_count - len(report_df)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows due to invalid values in {weight_column}")
    
    if rank_column and rank_column in report_df.columns:
        rank_attributes.append(rank_column)
        original_count = len(report_df)
        report_df[rank_column] = pd.to_numeric(report_df[rank_column], errors='coerce')
        report_df = report_df.dropna(subset=[rank_column])
        dropped_count = original_count - len(report_df)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows due to invalid values in {rank_column}")
    
    if len(rank_attributes) > 0:
        report_df.sort_values(by=rank_attributes, ascending=False, inplace=True)
    
    return report_df




def convert_report_context_to_df(
    context_records: List[List[str]],
    header: List[str],
    weight_column: Optional[str] = None,
    rank_column: Optional[str] = None
) -> pd.DataFrame:
    """Convert report context records to pandas dataframe and sort by weight and rank if exist."""
    if len(context_records) == 0:
        return pd.DataFrame()
    
    record_df = pd.DataFrame(
        context_records,
        columns=header
    )
    
    return rank_report_context(
        report_df=record_df,
        weight_column=weight_column,
        rank_column=rank_column,
    )
    
    
def is_include(report: CommunityReport, min_community_rank: int) -> bool:
    return report.rank is not None and report.rank >= min_community_rank


def get_header(attributes: List[str], include_community_weight: bool, community_weight_name: str, use_community_summary: bool, include_community_rank: bool, community_rank_name: str) -> List[str]:
    header = ["id", "title"]
    attributes = [col for col in attributes if col not in header]
    
    if not include_community_weight:
        attributes = [col for col in attributes if col != community_weight_name]
    header.extend(attributes)
    header.append("summary" if use_community_summary else "content")
    if include_community_rank:
        header.append(community_rank_name)
    return header

def report_context_text(
    report: CommunityReport,
    attributes: List[str],
    use_community_summary: bool,
    include_community_rank: bool,
    column_delimiter: str
) -> Tuple[str, List[str]]:
    context: List[str] = [
        report.short_id if report.short_id else "",
        report.title,
        *[
            str(report.attributes.get(field, "")) if report.attributes else ""
            for field in attributes
        ]
    ]
    
    context.append(report.summary if use_community_summary else report.full_content)
    if include_community_rank:
        context.append(str(report.rank))
    result = column_delimiter.join(context) + "\n"
    return result, context
    