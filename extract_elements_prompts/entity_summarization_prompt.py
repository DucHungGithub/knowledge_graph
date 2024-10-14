from pathlib import Path


from template import ENTITY_SUMMARIZATION_PROMPT

ENTITY_SUMMARIZATION_FILENAME = "summarize_descriptions.txt"

def create_entity_summarization_prompt(
    persona: str,
    language: str,
    output_path: Path | None = None
) -> str:
    """
    Create a prompt for entity summarization.

    Parameters
    ----------
    - persona (str): The persona to use for the entity summarization prompt
    - language (str): The language to use for the entity summarization prompt
    - output_path (Path | None): The path to write the prompt to. Default is None.
    """
    
    prompt = ENTITY_SUMMARIZATION_PROMPT.format(persona=persona, language=language)
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_path = output_path / ENTITY_SUMMARIZATION_FILENAME
        
        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8",errors="strict"))
            
    return prompt