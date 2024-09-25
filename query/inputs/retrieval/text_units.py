# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
from typing import Any, List, cast

import pandas as pd
import pydgraph

from models.entity import Entity
from models.text_unit import TextUnit


def get_candidate_text_units(
    client: pydgraph.DgraphClient,
    selected_entities: List[Entity],
) -> pd.DataFrame:
    selected_text_ids = [
        entity.text_unit_ids for entity in selected_entities if entity.text_unit_ids
    ]
    
    selected_text_ids = [item for sublist in selected_text_ids for item in sublist]
    
    source_text = ' OR '.join([f'eq(id, "{name}")' for name in selected_text_ids])
    txn = client.txn()
    selected_text_units = []
    try:
        query = f"""{{
                queryTextUnit(func: type(TextUnit)) @filter({source_text}){{
                    id
                    short_id
                    text
                    source
                    text_embedding
                    entity_ids
                    relationship_ids
                    covariate_ids
                    n_tokens
                    document_ids
                    attributes
            }}
        }}
        """
        res = txn.query(query=query)
        ppl = json.loads(res.json)
        
        tus = ppl["queryEntity"]
        for tu in tus:
            if tu.get("attributes", None):
                tu["attributes"] = json.loads(tu["attributes"]) if tu["attributes"] else None
            selected_text_units.append(TextUnit(**tu))
            
    finally:
        txn.discard()
    
    return to_text_unit_dataframe(selected_text_units)


def to_text_unit_dataframe(text_units: List[TextUnit]) -> pd.DataFrame:
    """Convert a list of text units to a pandas dataframe."""
    
    if len(text_units) == 0:
        return pd.DataFrame()
    
    header = ["id", "text", "source"]
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)
    
    records = []
    for unit in text_units:
        new_record = [
            unit.short_id,
            unit.text,
            unit.source,
            *[
                str(unit.attributes.get(field, ""))
                if unit.attributes and unit.attributes.get(field)
                else ""
                for field in attribute_cols
            ]
        ]
        records.append(new_record)
        
    return pd.DataFrame(records, columns=cast(Any, header))