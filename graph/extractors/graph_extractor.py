from collections.abc import Mapping
import re
from typing import Any, Dict, List
import logging

import networkx as nx
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
import tiktoken

from graph.extractors.prompt import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT
import config as defs
from utils.string import clean_str

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

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

class GraphExtractionResult(BaseModel):
    output: nx.Graph
    source_docs: Dict[Any, Any]
    
    class Config:
        arbitrary_types_allowed = True


class GraphExtractor:
    _llm: BaseChatModel
    _extraction_prompt: str
    _entity_types_key: str
    _record_delimiter_key: str
    _summarization_prompt: str
    _tuple_delimiter_key: str
    _completion_delimiter_key: str
    _max_gleanings: int
    _input_text_key: str
    
    def __init__(
        self,
        llm: BaseChatModel,
        tuple_delimiter_key: str | None = None,
        completion_delimiter_key: str | None = None,
        entity_types_key: str | None = None,
        encoding_model: str | None = None,
        extraction_prompt: str | None = None,
        max_gleanings: int | None = None,
        input_text_key: str | None = None,
        record_delimiter_key: str | None = None,
        join_descriptions: bool = True,
        
    ):
        self._llm = llm
        self._encoding_model = encoding_model
        self._extraction_prompt = extraction_prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else defs.ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._logit_bias = {yes[0]: 100, no[0]: 100}
        self._max_tokens = 1
        
    async def invoke(
        self,
        texts: List[str],
        prompt_variables: Dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        if prompt_variables is None:
            prompt_variables = {}
        
        all_records: Dict[int, str] = {}
        source_doc_map: Dict[int, str] = {}
        
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key) or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key) or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(self._completion_delimiter_key) or DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables.get(self._entity_types_key) or DEFAULT_ENTITY_TYPES
            )
        }

        
        for doc_index, text in enumerate(texts):
            try:
                result = await self._process_document(text=text, prompt_variables=prompt_variables)
                source_doc_map[doc_index] = text
                all_records[doc_index] = result
            except Exception as e:
                logger.info("Error extracting graph", e)
        
        output = await self._process_results(
            all_records,
            prompt_variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
            prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
        )
        
        
        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )
    
    async def _process_document(
        self,
        text: str,
        prompt_variables: Dict[str, str]
    ) -> str:

        template = ChatPromptTemplate.from_messages([
            ("system", self._extraction_prompt),
        ])
        
        prompt_variables = {
            **prompt_variables,
            self._input_text_key: text
        }

        prompt = template.format(**prompt_variables)
        
        response = self._llm.invoke(prompt)

        results = response.content or ""

        for i in range(self._max_gleanings):
            new_history = text + '\n' + results
            messages = [
                {"role": "system", "content": CONTINUE_PROMPT},
                {"role": "user", "content": new_history}
            ]

            response = self._llm.invoke(messages)
            results += response.content or ""

            if i >= self._max_gleanings - 1:
                break
            
            new_history = text + '\n' + results
            
            messages = [
                {"role": "system", "content": LOOP_PROMPT},
                {"role": "user", "content": new_history}
            ]

            old_max_tokens = self._llm.max_tokens
            old_logit_bias = self._llm.logit_bias
            self._llm.max_tokens = self._max_tokens
            self._llm.logit_bias = self._logit_bias
            
            response = self._llm.invoke(messages)

            if response.content.strip().upper() != "YES":
                break
            
            self._llm.max_tokens = old_max_tokens
            self._llm.logit_bias = old_logit_bias
            
        return results  
        
    # Copyright (c) 2024 Microsoft Corporation.
    # Licensed under the MIT License
    async def _process_results(
        self,
        results: dict[int, str],
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
            - tuple_delimiter - delimiter between tuples in an output record, default is '<|>'
            - record_delimiter - delimiter between records, default is '##'
        Returns:
            - output - unipartite graph in graphML format
        """
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes(data=True):
                        node = graph.nodes[entity_name]
                        print("Here is Node_____:")
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["entity_type"] = (
                            entity_type if entity_type != "" else node["entity_type"]
                        )
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    try:
                        weight = float(record_attributes[-1])
                    except ValueError:
                        weight = 1.0

                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )

        return graph


def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")