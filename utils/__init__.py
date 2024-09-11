import json
from pathlib import Path
import re
from typing import List
import uuid
import logging

import numpy as np
import pandas as pd
from langchain_core.documents import Document
import tiktoken
from json_repair import repair_json

from config import ENCODING_MODEL

logger = logging.getLogger(__name__)

def list_of_token(input_text: str, model_name: str = None, encoding_model: str = ENCODING_MODEL) -> List[int]:
    encoding = None
    if model_name is not None:
        encoding = tiktoken.encoding_for_model(model_name)
    else:
        encoding = tiktoken.get_encoding(encoding_model)
        
    return encoding.encode(input_text)
        


def documentsToDataframe(documents: List[Document]) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "id": chunk.id or uuid.uuid4().hex,
            "chunk": chunk.page_content,
            "n_tokens": len(list_of_token(chunk.page_content)),
            **chunk.metadata,
            
        }
        rows.append(row)
        
    return pd.DataFrame(rows)


def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        logger.info("Warning: Error decoding faulty json, attempting repair", exc_info=True)

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json") :]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            logger.exception("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                logger.exception("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result