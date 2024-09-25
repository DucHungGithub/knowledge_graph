
import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel


from generate_queries.base import BaseQuestionGen, QuestionResult
from generate_queries.prompts import QUESTION_SYSTEM_PROMPT
from query_context.system_prompt_builder.builders import LocalContextBuilder
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory


import colorlog

from utils import list_of_token

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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class LocalQuestionGen(BaseQuestionGen):
    
    def __init__(
        self,
        llm: BaseChatModel,
        context_builder: LocalContextBuilder,
        token_encoder: Optional[str] = None,
        system_prompt: str = QUESTION_SYSTEM_PROMPT,
        context_builder_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            context_builder_params=context_builder_params
        )
        self.system_prompt = system_prompt

    
    
    async def agenerate(
        self,
        question_history: List[str],
        context_data: Optional[str],
        question_count: int,
        **kwargs
    ) -> QuestionResult:
        """
        Generate a question based on the question history and context data.
        
        If context data is not provided, it will be generate by the local context builder
        """
        
        start_time = time.time()
        
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            # Construct current query and conversation history
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)
            
        if context_data is None:
            # Generate context data based on the question history
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params
            )
        else:
            context_records = {"context_data": context_data}
        
        logger.info(f"GENERATE QUESTION: {start_time}. LAST QUESTION: {question_text}")
        
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data, question_count=question_count
            )
            
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text}
            ]
            
            response = await self.llm.ainvoke(
                question_messages
            )
            
            return QuestionResult(
                response=response.content.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(system_prompt))
            )
        except Exception:
            logger.exception("Exception in generating questions")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(system_prompt)),
            )
    
    
    def generate(
        self,
        question_history: List[str],
        context_data: Optional[str],
        question_count: int,
        **kwargs
    ) -> QuestionResult:
        """
        Generate a question based on the question history and context data.
        
        If context data is not provided, it will be generate by the local context builder
        """
        
        start_time = time.time()
        
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            # Construct current query and conversation history
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)
            
        if context_data is None:
            # Generate context data based on the question history
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params
            )
        else:
            context_records = {"context_data": context_data}
        
        logger.info(f"GENERATE QUESTION: {start_time}. LAST QUESTION: {question_text}")
        
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data, question_count=question_count
            )
            
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text}
            ]
            
            response = self.llm.invoke(
                question_messages
            )
            
            return QuestionResult(
                response=response.content.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(system_prompt))
            )
        except Exception:
            logger.exception("Exception in generating questions")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(list_of_token(system_prompt)),
            )