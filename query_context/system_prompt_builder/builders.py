from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd

from query_context.system_prompt_builder.history.conversation_history import ConversationHistory

class GlobalContextBuilder(ABC):
    """Base class for global-search context builders."""
    
    @abstractmethod
    def build_context(
        self,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> Tuple[str | List[str], Dict[str, pd.DataFrame]]:
        """Build the context for the global search mode."""
        

class LocalContextBuilder(ABC):
    """Base class for local-search context builders."""
    
    @abstractmethod
    def build_context(
        self,
        query: str,
        conversation_history: Optional[ConversationHistory] = None,
        **kwargs
    ) -> Tuple[str | List[str], Dict[str, pd.DataFrame]]:
        """Build the context for the local search mode."""