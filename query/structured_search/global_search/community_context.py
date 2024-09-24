from typing import Dict, List, Optional, Tuple

import pandas as pd
from models.community_report import CommunityReport
from models.entity import Entity
from query.system_prompt_builder.builders import GlobalContextBuilder
from query.system_prompt_builder.history.conversation_history import ConversationHistory
from query.system_prompt_builder.process_context.community_context import build_community_context


class GlobalCommunityContext(GlobalContextBuilder):
    """GlobalSearch community context builder."""
    
    def __init__(
        self,
        community_reports: List[CommunityReport],
        entities: Optional[List[Entity]] = None,
        token_encoder: Optional[str] = None,
        random_state: int = 6969
    ) -> None:
        self.community_reports = community_reports
        self.entities = entities
        self.token_encoder = token_encoder
        self.random_state = random_state
        
    def build_context(
        self,
        conversation_history: Optional[ConversationHistory] = None,
        use_community_summary: bool = True,
        column_delimiter: str = "|",
        shuffle_data: bool = True,
        include_community_rank: bool = False,
        min_community_rank: int = 0,
        community_rank_name: str = "rank",
        include_community_weight: bool = True,
        community_weight_name: str = "occurrence",
        normalize_community_weight: bool = True,
        max_tokens: int = 8000,
        context_name: str = "Reports",
        conversation_history_user_turns_only: bool = True,
        conversation_history_max_turns: Optional[int] = 5,
        **kwargs,
    ) -> Tuple[str | List[str], Dict[str, pd.DataFrame]]:
        """Prepare batches of community report data table as context data for global search"""
        
        conversation_history_context = ""
        final_context_data = {}
        if conversation_history:
            (
                conversation_history_context,
                conversation_history_context_data
            ) = conversation_history.build_context(
                include_user_turns_only=conversation_history_user_turns_only,
                max_qa_turns=conversation_history_max_turns,
                column_delimiter=column_delimiter,
                max_tokens=max_tokens,
                recency_bias=False
            )
            if conversation_history_context != "":
                final_context_data = conversation_history_context_data
                
        community_context, community_context_data = build_community_context(
            community_reports=self.community_reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
            use_community_summary=use_community_summary,
            column_delimiter=column_delimiter,
            shuffle_data=shuffle_data,
            include_community_rank=include_community_rank,
            min_community_rank=min_community_rank,
            community_rank_name=community_rank_name,
            include_community_weight=include_community_weight,
            community_weight_name=community_weight_name,
            normalize_community_weight=normalize_community_weight,
            max_tokens=max_tokens,
            single_batch=False,
            context_name=context_name,
            random_state=self.random_state,
        )
        if isinstance(community_context, List):
            final_context = [
                f"{conversation_history_context}\n\n{context}"
                for context in community_context
            ]
        else:
            final_context = f"{conversation_history_context}\n\n{community_context}"
            
        final_context_data.update(community_context_data)
        return (final_context, final_context_data)