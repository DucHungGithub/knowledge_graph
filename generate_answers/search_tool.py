import logging
from typing import Dict, List, Sequence, Type, TypedDict

import pydgraph
import weaviate
from weaviate.auth import AuthApiKey
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import format_document, PromptTemplate
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt import create_react_agent


import colorlog

from query_context.structured_search.local_search.combine_context import LocalSearchMixedContext
from query_context.structured_search.local_search.search import LocalSearch
from query_context.system_prompt_builder.entity_extraction import EntityVectorStoreKey
from generate_answers.prompts import rewrite_prompt
from query_context.system_prompt_builder.history.conversation_history import ConversationHistory
from query_context.system_prompt_builder.history.typing import ConversationRole
from tools.base import BaseToolTemplate, ToolInputSchemaRegistry

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



class GraphSearchToolConstruct(BaseModel):
    name: str = Field(default="GraphSearchTool")
    llm_model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0)
    k: int
    alpha: float = Field(default=0.7)
    embeding_model: str = Field(default="text-embedding-ada-002")
    http_host: str
    http_port: int
    http_secure: bool
    grpc_host: str
    grpc_port: int
    grpc_secure: bool = Field(default=False)
    auth_credentials: str
    index_name: str
    tool_description: str
    dgraph_host: str = Field(default="localhost:9080")
    text_key: str = Field(default="text")
    response_type: str = Field(default="multiple paragraphs")
    text_unit_prop: float = Field(default=0.5)
    community_prop: float = Field(default=0.25)
    conversation_history_max_turns: int = Field(default=5)
    conversation_history_user_turns_only: bool = Field(default=False)
    top_k_mapped_entities: int = Field(default=10)
    top_k_relationships: int = Field(default=10)
    include_entity_rank: bool = Field(default=True)
    include_relationship_weight: bool = Field(default=True)
    include_community_rank: bool = Field(default=False)
    return_candidate_context: bool = Field(default=False)
    embedding_vectorstore_key: str = Field(EntityVectorStoreKey.ID)
    tool_inputs_schema_registry: List[ToolInputSchemaRegistry]
    max_tokens: int = Field(default=5000)
    context: Annotated[dict, InjectedState] = Field(default=None)
    max_retry: int = Field(default=3, gt=0)

from pydantic import BaseModel, Field

class OutputResponse(BaseModel):
    is_enough: str = Field(
        description="A value indicating whether the current context has sufficient information to answer the question. "
                    "Returns `yes` if the context is enough, otherwise returns `no`."
    )
    new_query: str = Field(
        description="A string representing the suggested query or question to gather additional information needed. "
                    "If `is_enough` is `yes`, this field can be an empty string or ignored. If `is_enough` is `no`, "
                    "this field should provide a specific query or instruction to obtain the missing data required to accurately answer the question."
    )
    
class InputSchema(BaseModel):
    query: str = Field(
        description="The user question need to answer based on the relevant information."
    )

class GraphSearchTool(BaseToolTemplate):
    name = "GraphSearchTool"
    description = ""
    
    
    def __init__(
        self,
        construct_param: Dict
    ) -> None:
        self.construct_param = GraphSearchToolConstruct(**construct_param)
        self.weaviate_client = weaviate.connect_to_custom(
            http_host=self.construct_param.http_host,
            http_port=self.construct_param.http_port,
            http_secure=self.construct_param.http_secure,
            grpc_host=self.construct_param.grpc_host,
            grpc_port=self.construct_param.grpc_port,
            grpc_secure=self.construct_param.grpc_secure,
            auth_credentials=AuthApiKey(self.construct_param.auth_credentials),
        )
        self.vector_store = WeaviateVectorStore(
            client=self.weaviate_client,
            index_name=self.construct_param.index_name,
            text_key=self.construct_param.text_key,
            embedding=OpenAIEmbeddings(model=self.construct_param.embeding_model),
        )
        self.llm = ChatOpenAI(model=self.construct_param.llm_model,temperature=self.construct_param.temperature)
        self.description = self.construct_param.tool_description
        self.name = self.construct_param.name
        self.context_builder = LocalSearchMixedContext(
            entity_text_embeddings=self.vector_store
        )
        self.local_context_params = {
                "text_unit_prop": self.construct_param.text_unit_prop,
                "community_prop": self.construct_param.community_prop,
                "conversation_history_max_turns": self.construct_param.conversation_history_max_turns,
                "conversation_history_user_turns_only": self.construct_param.conversation_history_user_turns_only,
                "top_k_mapped_entities": self.construct_param.top_k_mapped_entities,
                "top_k_relationships": self.construct_param.top_k_relationships,
                "include_entity_rank": self.construct_param.include_entity_rank,
                "include_relationship_weight": self.construct_param.include_relationship_weight,
                "include_community_rank": self.construct_param.include_community_rank,
                "return_candidate_context": self.construct_param.return_candidate_context,
                "embedding_vectorstore_key": self.construct_param.embedding_vectorstore_key,  
                "max_tokens": self.construct_param.max_tokens,  
            }
        self.search_engine = LocalSearch(
            llm=self.llm,
            context_builder=self.context_builder,
            context_builder_params=self.local_context_params,
            response_type=self.construct_param.response_type
        )
        self.decide_llm = self.llm.with_structured_output(OutputResponse)
        self.dgraph_stub = pydgraph.DgraphClientStub(self.construct_param.dgraph_host)
        self.dgraph_client = pydgraph.DgraphClient(self.dgraph_stub)
        self.max_retry = self.construct_param.max_retry
        self.prompt_input = ChatPromptTemplate.from_messages([
            "system", rewrite_prompt
        ])
        
    async def run(self, context: Annotated[dict, InjectedState], **kwargs) -> List[str]:
        logger.info(f"""{self.construct_param.name} execute with: {kwargs}""")
        result = None
        query = kwargs["query"]
        start_result = OutputResponse(is_enough=False, new_query=query)
        context_summarize = ""
        history_conversation = ConversationHistory()
        while self.max_retry:  
            result = self.search_engine.search(
                        client=self.dgraph_client,
                        query=query,
                        conversation_history=history_conversation
                    )
                    
            
            context_summarize = result.response
            history_conversation.add_turn(role=ConversationRole.from_string(value="user"), content=context_summarize)
            
            prompt_input = self.prompt_input.format(
                context=context_summarize,
                question=start_result.new_query
            )
            
            start_result = self.decide_llm.invoke(prompt_input)
            if str(start_result.is_enough).lower() == "yes" :
                context_data, _ = history_conversation.build_context()
                return context_data
            
            self.max_retry -= 1
        
        context_data, _ = history_conversation.build_context()
        return context_data
        
    def build(self) -> StructuredTool:
        return StructuredTool(
            name=self.name,
            description=self.description,
            func=None,
            coroutine=self.run,
            args_schema=InputSchema,
            handle_tool_error=True,
            handle_validation_error=True
        )
        
    @classmethod
    def get_tool_param(cls) -> Type[BaseModel]:
        return GraphSearchToolConstruct
        