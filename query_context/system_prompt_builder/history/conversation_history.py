from typing import Dict, List, Optional, Tuple

import pandas as pd

from query_context.system_prompt_builder.history.typing import ConversationRole, ConversationTurn, QATurn
from utils import list_of_token


class ConversationHistory:
    """Class for storing a conversation history."""
    turns: List[ConversationTurn]
    
    def __init__(self) -> None:
        self.turns = []
        
    @classmethod
    def from_list(
        cls,
        conversation_turns: List[Dict[str, str]],
    ) -> "ConversationHistory":
        """
        Create a conversation history from a list of conversation turns.
        
        Each turn is a dictionary in the form of {"role": "<conversation_role>", "content": "<turn content>"}
        """
        history = cls()
        for turn in conversation_turns:
            history.turns.append(
                ConversationTurn(
                    role=ConversationRole.from_string(
                        turn.get("role", ConversationRole.USER)
                    ),
                    content=turn.get("content","")
                )
            )
        return history
    
    def add_turn(self, role: ConversationRole, content: str) -> None:
        """Add a new turn to the conversation history."""
        self.turns.append(ConversationTurn(role=role, content=content))
        
    def to_qa_turns(self) -> List[QATurn]:
        """Convert conversation history to a list of QA turns."""
        qa_turns = List[QATurn]()
        current_qa_turn = None
        for turn in self.turns:
            if turn.role == ConversationRole.USER:
                if current_qa_turn:
                    qa_turns.append(current_qa_turn)
                current_qa_turn = QATurn(user_query=turn, assistant_answers=[])
            else:
                if current_qa_turn:
                    current_qa_turn.assistant_answers.append(turn)
        if current_qa_turn:
            qa_turns.append(current_qa_turn)
        return qa_turns
    
    def get_user_turns(self, max_user_turns: Optional[int] = 1) -> List[str]:
        """Get the last user turns in the conversation history."""
        user_turns = []
        for turn in self.turns[::-1]:
            if turn.role == ConversationRole.USER:
                user_turns.append(turn.content)
                if max_user_turns and len(user_turns) >= max_user_turns:
                    break
        return user_turns
    
    def build_context(
        self,
        token_encoder: Optional[str] = None,
        include_user_turns_only: bool = True,
        max_qa_turns: Optional[int] = 5,
        max_tokens: int = 8000,
        recency_bias: bool = True,
        column_delimiter: str = "|",
        context_name: str = "Conversation History"
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
        """
        Prepare conversation history as context data for system prompt.
        
        Parameters
        ----------
            user_queries_only: If True, only user queries (not assistant responses) will be included in the context, default is True.
            max_qa_turns: Maximum number of QA turns to include in the context, default is 1.
            recency_bias: If True, reverse the order of the conversation history to ensure last QA got prioritized.
            column_delimiter: Delimiter to use for separating columns in the context data, default is "|".
            context_name: Name of the context, default is "Conversation History".

        """
        qa_turns = self.to_qa_turns()
        if include_user_turns_only:
            qa_turns = [
                QATurn(user_query=qa_turn.user_query, assistant_answers=None)
                for qa_turn in qa_turns
            ]
            
        if recency_bias:
            qa_turns = qa_turns[::-1]
        if max_qa_turns and len(qa_turns) > max_qa_turns:
            qa_turns = qa_turns[:max_qa_turns]
            
        if len(qa_turns) == 0 or not qa_turns:
            return ("", {context_name: pd.DataFrame()})
        
        header = f"-----{context_name}-----" + "\n"

        turn_list = []
        current_context_df = pd.DataFrame()
        for turn in qa_turns:
            turn_list.append({
                "turn": ConversationRole.USER.__str__(),
                "content": turn.user_query.content
            })
            if turn.assistant_answers:
                turn_list.append({
                    "turn": ConversationRole.ASSISTANT.__str__(),
                    "content": turn.get_answer_text()
                })
                
            context_df = pd.DataFrame(turn_list)
            context_text = header + context_df.to_csv(sep=column_delimiter, index=False)
            
            if len(list_of_token(context_text, token_encoder)) > max_tokens:
                break
            current_context_df = context_df
        context_text = header + current_context_df.to_csv(
            sep=column_delimiter, index=False
        )
        return (context_text, {context_name.lower(): current_context_df})
                