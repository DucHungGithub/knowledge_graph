from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

class ConversationRole(str, Enum):
    """Enum for conversation roles"""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
    @staticmethod
    def from_string(value: str) -> "ConversationRole":
        """Convert string to ConversationRole."""
        if value == "system":
            return ConversationRole.SYSTEM
        
        if value == "user":
            return ConversationRole.USER
        
        if value == "assistant":
            return ConversationRole.ASSISTANT
        
        raise ValueError(f"Invalid Role: {value}")
    
    def __str__(self) -> str:
        return self.value
    
    
class ConversationTurn(BaseModel):
    """Data class for storing a single conversation turn."""
    
    role: ConversationRole
    content: str
    
    def __str__(self) -> str:
        return f"{self.role}: {self.content}"
    
class QATurn(BaseModel):
    """
    Data class for storing a QA turn.
    
    A QA turn contains a user question and one more multiple assistant answers.
    """
    
    user_query: ConversationTurn
    assistant_answers: Optional[List[ConversationTurn]] = None
    
    def get_answer_text(self) -> Optional[str]:
        """Return string representation of the QA turn."""
        return (
            "\n".join([answer.content for answer in self.assistant_answers])
            if self.assistant_answers
            else None   
        )
        
    def __str__(self) -> str:
        answers = self.get_answer_text()
        return (
            f"Question: {self.user_query.content}\nAnswer: {answers}"
            if answers
            else f"Question: {self.user_query.content}"
        )