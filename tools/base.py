from typing_extensions import Annotated
import uuid
from abc import abstractmethod, ABC
from typing import Any, Union, Dict, Tuple, List, Union, Type, Optional, TypeVar
from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field, create_model
from langgraph.prebuilt import InjectedState


import logging
from langchain_core.callbacks.base import Callbacks
from langchain_core.tools import ToolException
from langchain_core.tools import StructuredTool
from langchain_core.runnables import (
    RunnableConfig,
)
from inspect import signature
from langchain_core.callbacks import (
    AsyncCallbackManager,
)
from contextvars import copy_context
from langchain_core.runnables.config import (
    patch_config,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import accepts_context
import asyncio
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
)
from langchain_core.pydantic_v1.dataclasses import dataclass
from typing import cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ToolInputSchemaRegistry(BaseModel):
    input_name: str = Field(default="3", description="PLAIN_TEXT")
    input_description: str = Field(default="3", description="PLAIN_TEXT")
    input_data_type: str = Field(default="3", description="PLAIN_TEXT")


class TypeAllowedBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ToolExecutionContextResult(TypeAllowedBaseModel):
    """
    Cause result may contains some Object like pandas dataframe which pydantic don't cover the validation,
    we have to put arbitrary_types_allowed = True config for ToolExecutionContextResult
    """

    pass


class BaseToolTemplate(ABC):
    name: str
    description: str

    kguru_is_tool: bool = True
    kguru_tool_name: str

    async def run(self, context: Annotated[dict, InjectedState], **kwargs) -> str:
        return ""

    @abstractmethod
    def build(self) -> "StructuredTool":
        pass

    def _build_tool_input_pydantic_schema(
        self, tool_inputs_schema_registry: List[ToolInputSchemaRegistry]
    ) -> Type[BaseModel]:
        """
        Construct pydantic schema for tool input.
        This helps dynamic tool input schema from construct param.
        """
        model_attribute = {}
        for tool_input_schema_registry in tool_inputs_schema_registry:
            model_attribute[tool_input_schema_registry.input_name] = (
                self._get_mapping_data_type(tool_input_schema_registry.input_data_type),
                Field(description=tool_input_schema_registry.input_description),
            )

        model_attribute["context"] = (
            Annotated[dict, InjectedState],
            Field(default=None),
        )
        tool_input_schema = create_model(
            f"ToolInput-{uuid.uuid4()}",
            **model_attribute,
            __config__=BaseModel.Config,
        )
        return tool_input_schema

    def _get_mapping_data_type(
        self, type: str
    ) -> Union[Type[str], Type[int], Type[List[str]]]:
        match type:
            case "str":
                return str
            case "int":
                return int
            case "List[str]":
                return List[str]
            case _:
                raise ServiceException(
                    ServiceErrorCode.SKILLED_AGENT_ID_NOT_FOUND,
                    f"Tool input data type: {type} is not supported",
                )

    def set_execution_result_to_context(
        self,
        execution_result: ToolExecutionContextResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """
        Check if context already have any tool_execution_context_result and set the result into context.
        """
        if context != None:
            if context.get("tool_execution_context_result") != None:
                context["tool_execution_context_result"].append(execution_result)
            else:
                context["tool_execution_context_result"] = [execution_result]

    def get_tool_execution_result_context(
        self, context: Optional[Dict[str, Any]]
    ) -> List[ToolExecutionContextResult]:
        if context == None or context["tool_execution_context_result"] == None:
            return []
        return context["tool_execution_context_result"]

    def get_latest_execution_result(
        self,
        context: Optional[Dict[str, Any]],
        result_class: Type[T],
    ) -> Optional[T]:
        """
        If tool executed many time, this function help to return latest execute result.
        """
        tool_execution_result_context = self.get_tool_execution_result_context(context)
        result = None
        for tool_execution_result in tool_execution_result_context:
            if tool_execution_result.__class__.__name__ == result_class.__name__:
                result = tool_execution_result
        return cast(Optional[T], result)
