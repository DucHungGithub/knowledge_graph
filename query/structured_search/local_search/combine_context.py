# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
from typing import Any, Dict, List, Optional, Tuple
import logging

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import pydgraph

from models.community_report import CommunityReport
from models.covariate import Covariate
from models.entity import Entity
from models.relationship import Relationship
from models.text_unit import TextUnit
from query.inputs.retrieval.community_reports import get_candidate_communities
from query.inputs.retrieval.text_units import get_candidate_text_units
from query.system_prompt_builder.builders import LocalContextBuilder
from query.system_prompt_builder.entity_extraction import EntityVectorStoreKey, map_query_to_entities
from query.system_prompt_builder.history.conversation_history import ConversationHistory
from query.system_prompt_builder.local_context import get_candidate_context
from query.system_prompt_builder.process_context.community_context import build_community_context
from query.system_prompt_builder.process_context.covariate_context import build_covariates_context
from query.system_prompt_builder.process_context.entity_context import build_entity_context
from query.system_prompt_builder.process_context.relationship_context import build_relationship_context
from query.system_prompt_builder.process_context.textunit_context import build_text_unit_context, count_relationship
from utils import list_of_token


import colorlog

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

class LocalSearchMixedContext(LocalContextBuilder):
    """Build data context for local search prompt combining community reports and entity/relationship/covariate tables."""
    
    def __init__(
        self,
        entity_text_embeddings: VectorStore,
        text_embedder: Embeddings,
        # entities: List[Entity] = None,
        # text_units: Optional[List[TextUnit]] = None,
        # community_reports: Optional[List[CommunityReport]] = None,
        # relationships: Optional[List[Relationship]] = None,
        # covariates: Optional[Dict[str, List[Covariate]]] = None,
        token_encoder: Optional[str] = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID
    ) -> None:
        # if community_reports is None:
        #     community_reports = []
        # if relationships is None:
        #     relationships = []
        # if covariates is None:
        #     covariates = {}
        # if text_units is None:
        #     text_units = []
        # if entities is None:
        #     entities = []
        # self.entities = {entity.id: entity for entity in entities}
        # self.community_reports = {
        #     community.id: community for community in community_reports
        # }
        # self.text_units = {unit.id: unit for unit in text_units}
        # self.relationships = {
        #     relationship.id: relationship for relationship in relationships
        # }
        # self.covariates = covariates
        self.entity_text_embeddings = entity_text_embeddings
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder
        self.embedding_vectorstore_key = embedding_vectorstore_key
        
    def filter_by_entity_keys(
        self,
        entity_keys: List[int] | List[str]
    ) -> List[Document]:
        """Filter entity text embeddings by entity keys"""
        return self.entity_text_embeddings.get_by_ids(entity_keys)
    
    def build_context(
        self, 
        client: pydgraph.DgraphClient,
        query: str, 
        conversation_history: ConversationHistory | None = None,
        include_entity_names: Optional[List[str]] = None,
        exclude_entity_names: Optional[List[str]] = None,
        conversation_history_max_turns: Optional[int] = 5,
        conversation_history_user_turns_only: Optional[bool] = False,
        max_tokens: int = 8000,
        text_unit_prop: float = 0.5,
        community_prop: float = 0.25,
        top_k_mapped_entities: int = 10,
        top_k_relationships: int = 10,
        include_community_rank: bool = False,
        include_entity_rank: bool = False,
        rank_description: str = "number of relationships",
        include_relationship_weight: bool = False,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        use_community_summary: bool = False,
        min_community_rank: int = 0,
        community_context_name: str = "Reports",
        column_delimiter: str = "|",
        **kwargs: Dict[str, Any]
    ) -> Tuple[str | List[str], Dict[str, DataFrame]]:
        """
        Build data context for local search prompt.
        
        Build a context by combining community report and entity/relationship/covariate tables, and text units using a predefined ratio set by summary_prop.
        """
        if include_entity_names is None:
            include_entity_names = []
            
        if exclude_entity_names is None:
            exclude_entity_names = []
            
        if community_prop + text_unit_prop > 1:
            raise ValueError(f"The sum of community_prop and text_unit_prop should not exceed 1.")
        
        # Map query to entities
        # If there is conversation history, attached the previous user questions to the query
        if conversation_history:
            pre_user_questions = "\n".join(
                conversation_history.get_user_turns(conversation_history_max_turns)
            )
            query = f"{query}\n{pre_user_questions}"
            
        selected_entities = map_query_to_entities(
            client=client,
            query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            include_entity_names=include_entity_names,
            exclude_entity_names=exclude_entity_names,
            k=top_k_mapped_entities,
            oversample_scaler=2
        )
        
        
        # Build context variables
        final_context = []
        final_context_data = {}
        
        # Build conversation history context
        if conversation_history:
            (
                conversation_history_context,
                conversation_history_context_data,
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_tokens=max_tokens,
                recency_bias=False
            )
            
            
            if conversation_history_context.strip() != "":
                final_context.append(conversation_history_context)
                final_context_data = conversation_history_context_data
                max_tokens = max_tokens - len(list_of_token(conversation_history_context, self.token_encoder))
                
            print("## Conversation History Context---------------------: ")
            print(conversation_history_context)    
            
        # Build community context
        community_tokens = max(int(max_tokens * community_prop), 0)
        community_context, community_context_data = self._build_community_context(
            client=client,
            selected_entities=selected_entities,
            max_tokens=community_tokens,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            return_candidate_context=return_candidate_context,
            context_name=community_context_name
        )
        
        
        if community_context.strip() != "":
            final_context.append(community_context)
            final_context_data = {**final_context_data, **community_context_data}

        
            
        # Build local context: "entity-relationship-covariate" context
        local_prop = 1 - community_prop - text_unit_prop
        local_tokens = max(int(max_tokens * local_prop), 0)
        local_context, local_context_data = self._build_local_context(
            client=client,
            selected_entities=selected_entities,
            max_tokens=local_tokens,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            include_relationship_weight=include_relationship_weight,
            top_k_relationships=top_k_relationships,
            relationship_ranking_attribute=relationship_ranking_attribute,
            return_candidate_context=return_candidate_context,
            column_delimiter=column_delimiter
        )
        if local_context.strip() != "":
            final_context.append(str(local_context))
            final_context_data = {**final_context_data, **local_context_data}
            
        print("## Local Context: entity-relationship-covariate---------------------: ")
        print(local_context)
        
        # Build text unit context
        text_unit_tokens = max(int(max_tokens * text_unit_prop), 0)
        text_unit_context, text_unit_context_data = self._build_text_unit_context(
            client=client,
            selected_entities=selected_entities,
            max_tokens=text_unit_tokens,
            return_candidate_context=return_candidate_context
        )
        if text_unit_context.strip() != "":
            final_context.append(text_unit_context)
            final_context_data = {**final_context_data, **text_unit_context_data}
            
        print("## Text Unit Context: ---------------------: ")
        print(text_unit_context)
        
        
        print("Final Combine Context--------")
        print("\n\n".join(final_context))
        return ("\n\n".join(final_context), final_context_data)
        
        
    def _build_text_unit_context(
        self,
        client: pydgraph.DgraphClient,
        selected_entities: List[Entity],
        max_tokens: int = 8000,
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        context_name: str = "Source",
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Rank matching text units and add them to the context window until it hits the max_tokens limit"""
        if not selected_entities:
            return ("", {context_name.lower(): pd.DataFrame()})
        
        selected_text_units = []
        text_unit_ids_set = set()
        
        txn = client.txn()
        
        try:
            for index, entity in enumerate(selected_entities):
                for text_id in entity.text_unit_ids or []:
                    if text_id not in text_unit_ids_set:
                        query = f"""{{
                            getTextUnit(func: type(TextUnit)) @filter(eq(id,{text_id})){{
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
                        
                        if len(ppl.get("getTextUnit", [])) == 0:
                            continue
                        
                        text_unit_ids_set.add(text_id)
                        selected_unit = ppl.get("getTextUnit")[0]
                        if selected_unit.get("attributes", None):
                            selected_unit["attributes"] = json.loads(selected_unit["attributes"]) if selected_unit["attributes"] else None
                        text_unit_domain = TextUnit(**selected_unit)
                        num_relationships = count_relationship(
                            client=client, text_unit=text_unit_domain, entity=entity
                        )
                        if text_unit_domain.attributes is None:
                            text_unit_domain.attributes = {}
                        text_unit_domain.attributes["entity_order"] = index
                        text_unit_domain.attributes["num_relationships"] = num_relationships
                        selected_text_units.append(text_unit_domain)
                        
        finally:
            txn.discard()
        
        
        selected_text_units.sort(
            key=lambda x: (x.attributes["entity_order"], -x.attributes["num_relationships"])
        )
        
        for unit in selected_text_units:
            unit.attributes.pop("entity_order", None)
            unit.attributes.pop("num_relationships", None)
           
        context_text, context_data = build_text_unit_context(
            text_units=selected_text_units,
            token_encoder=self.token_encoder,
            max_tokens=max_tokens,
            shuffle_data=False,
            context_name=context_name,
            column_delimiter=column_delimiter
        )
        
        if return_candidate_context:
            candidate_context_data = get_candidate_text_units(
                selected_entities=selected_entities,
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                candidate_context_data["in_context"] = False
                context_data[context_key] = candidate_context_data
            else:
                if ("id" in candidate_context_data.columns and "id" in context_data[context_key].columns):
                    candidate_context_data["in_context"] = candidate_context_data["id"].isin(context_data[context_key]["id"])
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True
                    
        return (str(context_text), context_data)
        
        
        
        
    def _build_community_context(
        self,
        client: pydgraph.DgraphClient,
        selected_entities: List[Entity],
        max_tokens: int = 4000,
        use_community_summary: bool = False,
        column_delimiter: str = "|",
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        return_candidate_context: bool = False,
        context_name: str = "Reports"
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Add community data to the context window until it hits the max_tokens limit."""
        # if len(selected_entities) == 0 or len(self.community_reports) == 0:
        #     return ("", {context_name.lower(): pd.DataFrame()})
        
        if len(selected_entities) == 0:
            return ("", {context_name.lower(): pd.DataFrame()})
        
        community_matches = {}
        for entity in selected_entities:
            # Increase count of the community that this entity belongs to
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = (
                        community_matches.get(community_id, 0) + 1
                    )
                    

        # Sort communities by number of matched entities and rank
        selected_communities = []
        
        txn = client.txn()
        
        try:
            for community_id in community_matches:
                query = f"""
                {{
                    getCommunity(func: type(CommunityReport)) @filter(eq(community_id, {community_id})) {{
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
                
                coms = ppl.get("getCommunity", [])
                
                for com in coms:
                    if com.get("attributes", None):
                        com["attributes"] = json.loads(com["attributes"]) if com["attributes"] else None
                    selected_communities.append(CommunityReport(**com))
                
        finally:
            txn.discard()
        
        
        
        for community in selected_communities:
            if community.attributes is None:
                community.attributes = {}
            community.attributes["matches"] = community_matches[community.community_id]
        selected_communities.sort(
            key=lambda x: (x.attributes["matches"], x.rank),
            reverse=True
        )
        
        for community in selected_communities:
            del community.attributes["matches"]
        
        context_text, context_data = build_community_context(
            community_reports=selected_communities,
            entities=selected_entities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=False,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            max_tokens=max_tokens,
            single_batch=True,
            context_name=context_name
        )
        
        if isinstance(context_text, List) and len(context_text) > 0:
            context_text = "\n\n".join(context_text)
            
        if return_candidate_context:
            candidate_context_data = get_candidate_communities(
                selected_reports=selected_communities,
                use_community_summary=use_community_summary,
                include_community_rank=include_community_rank
            )
            context_key = context_name.lower()
            if context_key not in context_data:
                context_data[context_key] = candidate_context_data
                context_data[context_key]["in_context"] = False
            else:
                if (
                    "id" in candidate_context_data.columns
                    and "id" in context_data[context_key].columns
                ):
                    candidate_context_data["in_context"] = candidate_context_data["id"].isin(context_data[context_key]["id"])
                    context_data[context_key] = candidate_context_data
                else:
                    context_data[context_key]["in_context"] = True
        return (str(context_text), context_data)
    
    
    def _build_local_context(
        self,
        client: pydgraph.DgraphClient,
        selected_entities: List[Entity],
        max_tokens: int = 8000,
        include_entity_rank: bool = False,
        rank_description: str = "relationship count",
        include_relationship_weight: bool = False,
        top_k_relationships: int = 10,
        relationship_ranking_attribute: str = "rank",
        return_candidate_context: bool = False,
        column_delimiter: str = "|",
        covariates: str = "Covariates"
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """Build data context for local search prompt combining entity/relationship/covariate tables."""
        # Build entity context
        entity_context, entity_context_data = build_entity_context(
            selected_entities=selected_entities,
            token_encoder=self.token_encoder,
            max_tokens=max_tokens,
            column_delimiter=column_delimiter,
            include_entity_rank=include_entity_rank,
            rank_description=rank_description,
            context_name="Entities"
        )
        entity_tokens = len(list_of_token(entity_context, self.token_encoder))
        
        
        # Build relationship-covariate context
        added_entities = []
        final_context = []
        final_context_data = {}
        
        # Gradually add entities and associated metadata to the context until we reach limit
        for entity in selected_entities:
            current_context = []
            current_context_data = {}
            added_entities.append(entity)
            
            # Build relationship context
            (
                relationship_context,
                relationship_context_data
            ) = build_relationship_context(
                client=client,
                selected_entities=added_entities,
                token_encoder=self.token_encoder,
                max_tokens=max_tokens,
                column_delimiter=column_delimiter,
                top_k_relationships=top_k_relationships,
                include_relationship_weight=include_relationship_weight,
                relationship_ranking_attribute=relationship_ranking_attribute,
                context_name="Relationships"
            )
            current_context.append(relationship_context)
            current_context_data["relationships"] = relationship_context_data
            total_tokens = entity_tokens + len(list_of_token(relationship_context, self.token_encoder))
            
            covariate_context, covariate_context_data = build_covariates_context(
                client=client,
                selected_entities=added_entities,
                token_encoder=self.token_encoder,
                max_tokens=max_tokens,
                column_delimiter=column_delimiter,
                context_name=covariates
            )
            total_tokens += len(list_of_token(covariate_context, self.token_encoder))
            current_context.append(covariate_context)
            current_context_data[covariates.lower()] = covariate_context_data
                
            if total_tokens > max_tokens:
                logger.info("Reacher token limit - reverting to previous context state", exc_info=True)
                break
            
            final_context = current_context
            final_context_data = current_context_data
        
        # Attach entity context to final context
        final_context_text = entity_context + "\n\n" + "\n\n".join(final_context)
        final_context_data["entities"] = entity_context_data
        
        if return_candidate_context:
            # return all the candidate entities/relationships/covariates (not only those that were fitted into the context window)
            # and add a tag to indicate which records were included in the context window)
            candidate_context_data = get_candidate_context(
                client=client,
                selected_entities=selected_entities,
                include_entity_rank=include_entity_rank,
                entity_rank_description=rank_description,
                include_relationship_weight=include_relationship_weight
            )
            for key in candidate_context_data:
                candidate_df = candidate_context_data[key]
                if key not in final_context_data:
                    final_context_data[key] = candidate_df
                    final_context_data[key]["in_context"] = False
                else:
                    in_context_df = final_context_data[key]
                    
                    if "id" in in_context_df.columns and "id" in candidate_df.columns:
                        candidate_df["in_context"] = candidate_df["id"].isin(in_context_df["id"])
                        final_context_data[key] = candidate_df
                    else:
                        final_context_data[key]["in_context"] = True
        else:
            for key in final_context_data:
                final_context_data[key]["in_context"] = True
        return (final_context_text, final_context_data)