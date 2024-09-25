from typing import List, Optional
import pandas as pd
import ast

from models.community import Community
from models.community_report import CommunityReport
from models.covariate import Covariate
from models.document import Document
from models.entity import Entity
from models.relationship import Relationship
from models.text_unit import TextUnit
from query.inputs.loader.utils import (
    to_optional_dict,
    to_optional_float,
    to_optional_int,
    to_optional_list,
    to_optional_str,
    to_str
)


# def store_entity_semantic_embeddings(
#     entities: List[Entity],
#     vectorstore: BaseVectorStore,
# )





def read_entities(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    title_col: str = "title",
    type_col: Optional[str] = "type",
    description_col: Optional[str] = "description",
    name_embedding_col: Optional[str] = "name_embedding",
    # description_embedding_col: Optional[str] = "description_embedding",
    graph_embedding_col: Optional[str] = "graph_embedding",
    community_col: Optional[str] = "community_ids",
    text_unit_ids_col: Optional[str] = "text_unit_ids",
    document_ids_col: Optional[str] = "document_ids",
    rank_col: Optional[str] = "degree",
    attributes_cols: List[str] | None = None,
) -> List[Entity]:
    """Read entities from a dataframe."""
    entities = []
    for idx, row in df.iterrows():
        entity = Entity(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            type=to_optional_str(row, type_col),
            description=to_optional_str(row, description_col),
            name_embedding=to_optional_list(row, name_embedding_col, item_type=float),
            # description_embedding=to_optional_list(
            #     row, description_embedding_col, item_type=float
            # ),
            graph_embedding=to_optional_list(row, graph_embedding_col, item_type=float),
            community_ids=to_optional_list(row, community_col, item_type=str),
            text_unit_ids=to_optional_list(row, text_unit_ids_col),
            document_ids=to_optional_list(row, document_ids_col),
            rank=to_optional_int(row, rank_col),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        
        entities.append(entity)
    return entities


def read_relationships(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    source_col: str = "source",
    target_col: str = "target",
    description_col: Optional[str] = "description",
    description_embedding_col: Optional[str] = "description_embedding",
    weight_col: Optional[str] = "weight",
    text_unit_ids_col: Optional[str] = "text_unit_ids",
    document_ids_col: Optional[str] = "document_ids",
    attributes_cols: Optional[List[str]] = None,
) -> List[Relationship]:
    """Read relationships from a dataframe."""
    relationships = []
    for idx, row in df.iterrows():
        rel = Relationship(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            source=to_str(row, source_col),
            target=to_str(row, target_col),
            description=to_optional_str(row, description_col),
            description_embedding=to_optional_list(
                row, description_embedding_col, item_type=float
            ),
            weight=to_optional_float(row, weight_col),
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        relationships.append(rel)
    return relationships


def read_covariates(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    subject_col: str = "subject_id",
    subject_type_col: Optional[str] = "subject_type",
    covariate_type_col: Optional[str] = "covariate_type",
    text_unit_ids_col: Optional[str] = "text_unit_ids",
    document_ids_col: Optional[str] = "document_ids",
    attributes_cols: List[str] | None = None,
) -> List[Covariate]:
    """Read covariates from a dataframe."""
    covariates = []
    for idx, row in df.iterrows():
        cov = Covariate(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            subject_id=to_str(row, subject_col),
            subject_type=(
                to_str(row, subject_type_col) if subject_type_col else "entity"
            ),
            covariate_type=(
                to_str(row, covariate_type_col) if covariate_type_col else "claim"
            ),
            text_unit_ids=to_optional_list(row, text_unit_ids_col, item_type=str),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        covariates.append(cov)
    return covariates


def read_communities(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    title_col: str = "title",
    level_col: str = "level",
    entities_col: Optional[str] = "entity_ids",
    relationships_col: Optional[str] = "relationship_ids",
    covariates_col: Optional[str] = "covariate_ids",
    attributes_cols: List[str] | None = None,
) -> List[Community]:
    """Read communities from a dataframe."""
    communities = []
    for idx, row in df.iterrows():
        comm = Community(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            level=to_str(row, level_col),
            entity_ids=to_optional_list(row, entities_col, item_type=str),
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        communities.append(comm)
    return communities


def read_community_reports(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    title_col: str = "title",
    community_col: str = "community",
    summary_col: str = "summary",
    content_col: str = "full_content",
    rank_col: Optional[str] = "rank",
    summary_embedding_col: Optional[str] = "summary_embedding",
    content_embedding_col: Optional[str] = "full_content_embedding",
    attributes_cols: List[str] | None = None,
) -> List[CommunityReport]:
    """Read community reports from a dataframe."""
    reports = []
    for idx, row in df.iterrows():
        report = CommunityReport(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            community_id=to_str(row, community_col),
            summary=to_str(row, summary_col),
            full_content=to_str(row, content_col),
            rank=to_optional_float(row, rank_col),
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),
            full_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        reports.append(report)
    return reports


def read_text_units(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: Optional[str] = "short_id",
    text_col: str = "chunk",
    source_col: str = "source",
    entities_col: Optional[str] = "entity_ids",
    relationships_col: Optional[str] = "relationship_ids",
    covariates_col: Optional[str] = "covariate_ids",
    tokens_col: Optional[str] = "n_tokens",
    document_ids_col: Optional[str] = "document_ids",
    embedding_col: Optional[str] = "text_embedding",
    attributes_cols: List[str] | None = None,
) -> List[TextUnit]:
    """Read text units from a dataframe."""
    text_units = []
    for idx, row in df.iterrows():
        chunk = TextUnit(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            text=to_str(row, text_col),
            source=to_str(row, source_col),
            entity_ids=to_optional_list(row, entities_col, item_type=str),
            relationship_ids=to_optional_list(row, relationships_col, item_type=str),
            covariate_ids=to_optional_dict(
                row, covariates_col, key_type=str, value_type=str
            ),
            text_embedding=to_optional_list(row, embedding_col, item_type=float),  # type: ignore
            n_tokens=to_optional_int(row, tokens_col),
            document_ids=to_optional_list(row, document_ids_col, item_type=str),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        text_units.append(chunk)
    return text_units


def read_documents(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str = "short_id",
    title_col: str = "title",
    type_col: str = "type",
    summary_col: Optional[str] = "entities",
    raw_content_col: Optional[str] = "relationships",
    summary_embedding_col: Optional[str] = "summary_embedding",
    content_embedding_col: Optional[str] = "raw_content_embedding",
    text_units_col: Optional[str] = "text_units",
    attributes_cols: List[str] | None = None,
) -> List[Document]:
    """Read documents from a dataframe."""
    docs = []
    for idx, row in df.iterrows():
        doc = Document(
            id=to_str(row, id_col),
            short_id=to_optional_str(row, short_id_col) if short_id_col else str(idx),
            title=to_str(row, title_col),
            type=to_str(row, type_col),
            summary=to_optional_str(row, summary_col),
            raw_content=to_str(row, raw_content_col),
            summary_embedding=to_optional_list(
                row, summary_embedding_col, item_type=float
            ),
            raw_content_embedding=to_optional_list(
                row, content_embedding_col, item_type=float
            ),
            text_units=to_list(row, text_units_col, item_type=str),  # type: ignore
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        docs.append(doc)
    return docs