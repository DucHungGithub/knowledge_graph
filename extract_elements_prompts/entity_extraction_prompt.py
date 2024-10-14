from pathlib import Path
from typing import List

from external_utils.token import list_of_token
from template import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    GRAPH_EXTRACTION_JSON_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
)


from config import ENCODING_MODEL

ENTITY_EXTRACTION_FILENAME = "entity_extraction.txt"


def create_entity_extraction_prompt(
    entity_types: str | List[str] | None,
    docs: List[str],
    examples: List[str],
    language: str,
    max_token_count: int,
    encoding_model: str = ENCODING_MODEL,
    json_mode: bool = False,
    output_path: Path | None = None,
    min_examples_required: int = 2
) -> str:
    """
    Create a prompt for entity extraction.

    Parameters
    ----------
    - entity_types (str | list[str]): The entity types to extract
    - docs (list[str]): The list of documents to extract entities from
    - examples (list[str]): The list of examples to use for entity extraction
    - language (str): The language of the inputs and outputs
    - encoding_model (str): The name of the model to use for token counting
    - max_token_count (int): The maximum number of tokens to use for the prompt
    - json_mode (bool): Whether to use JSON mode for the prompt. Default is False
    - output_path (Path | None): The path to write the prompt to. Default is None.
        - min_examples_required (int): The minimum number of examples required. Default is 2.

    Returns
    -------
    - str: The entity extraction prompt
    """
    
    prompt = (
        (GRAPH_EXTRACTION_JSON_PROMPT if json_mode else GRAPH_EXTRACTION_PROMPT)
        if entity_types
        else UNTYPED_GRAPH_EXTRACTION_PROMPT
    )
    
    if isinstance(entity_types, List):
        entity_types = ", ".join(map(str, entity_types))
        
    token_left = (
        max_token_count - len(list_of_token(prompt, encoding_model=encoding_model)) - len(list_of_token(entity_types, encoding_model=encoding_model))
        if entity_types
        else 0
    )
    
    examples_prompt = ""
    
    for i, output in enumerate(examples):
        input = docs[i]
        example_formatted = (
            EXAMPLE_EXTRACTION_TEMPLATE.format(
                n = i + 1, input_text = input, entity_types = entity_types, output = output
            )
            if entity_types
            else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
                n = i + 1, input_text = input, output = output
            )
        )
        
        example_tokens = len(list_of_token(
            example_formatted, encoding_model=encoding_model
        ))
        
        if i >= min_examples_required and example_tokens > token_left:
            break
        
        examples_prompt += example_formatted
        token_left -= example_tokens
    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_path = output_path / ENTITY_EXTRACTION_FILENAME
        
        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))
            
    return prompt