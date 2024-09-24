import logging
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
import tiktoken

from graph.claims.prompts import CLAIM_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT
import config as defs




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

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

class ClaimExtractorResult(BaseModel):
    """Claim extractor result class definition"""
    
    output: List[Dict[str, Any]]
    source_docs: Dict[str, Any]
    

class ClaimExtractor:
    """Claim extractor class"""
    
    _llm: BaseChatModel
    _extraction_prompt: str
    _summary_prompt: str
    _output_formatter_prompt: str
    _input_text_key: str
    _input_entity_spec_key: str
    _input_claim_description_key: str
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _completion_delimiter_key: str
    _max_gleanings: int
    
    def __init__(
        self,
        llm: BaseChatModel,
        extraction_prompt: str | None = None,
        input_text_key: str | None = None,
        input_entity_spec_key: str | None = None,
        input_claim_description_key: str | None = None,
        input_resolved_entities_key: str | None = None,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        completion_delimiter_key: str | None = None,
        encoding_model: str | None  = None,
        max_gleanings: int | None = None
    ):
        self._llm = llm
        self._extraction_prompt = extraction_prompt or CLAIM_EXTRACTION_PROMPT
        self._input_text_key = input_text_key or "input_text"
        self._input_entity_spec_key = input_entity_spec_key or "entity_specs"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = completion_delimiter_key or "completion_delimiter"
        self._input_claim_description_key = input_claim_description_key or "claim_description"
        self._input_resolved_entities_key = input_resolved_entities_key or "resolved_entities"
        self._max_gleanings = max_gleanings if max_gleanings is not None else defs.CLAIM_MAX_GLEANINGS
        
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._logit_bias = {yes[0]: 100, no[0]: 100}
        self._max_tokens = 1 
        
    async def invoke(
        self,
        inputs: Dict[str, Any],
        prompt_variables: Dict[str, Any] | None = None
    ) -> ClaimExtractorResult:
        if prompt_variables is None:
            prompt_variables = {}
            
        texts = inputs[self._input_text_key]
        entity_spec = inputs[self._input_entity_spec_key]
        claim_description = inputs[self._input_claim_description_key]
        resolved_entities = inputs.get(self._input_resolved_entities_key, {})
        
        tuple_delimiter = prompt_variables.get(self._tuple_delimiter_key) or DEFAULT_TUPLE_DELIMITER
        record_delimiter = prompt_variables.get(self._record_delimiter_key) or DEFAULT_RECORD_DELIMITER
        completion_delimiter = prompt_variables.get(self._completion_delimiter_key) or DEFAULT_COMPLETION_DELIMITER
        
        source_doc_map = {}
        
        prompt_args = {
            self._input_entity_spec_key: entity_spec,
            self._input_claim_description_key: claim_description,
            self._tuple_delimiter_key: tuple_delimiter,
            self._record_delimiter_key: record_delimiter,
            self._completion_delimiter_key: completion_delimiter
        }
        
        all_claims: List[Dict[str, Any]] = []
        
        for index, doc in enumerate(texts):
            document_id = doc.id
            try:
                claims = await self._process_document(prompt_args, doc.page_content, index)
                all_claims += [
                    self._clean_claim(c, document_id, resolved_entities) for c in claims
                ]
                source_doc_map[document_id] = doc.page_content
            except Exception as e:
                logger.exception(f"Error extracting claim: {e}", exc_info=True)
                continue
            
        return ClaimExtractorResult(
            output=all_claims,
            source_docs=source_doc_map
        )
                
    
    
    def _clean_claim(
        self,
        claim: Dict[str, Any],
        document_id: str,
        resolved_entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        obj = claim.get("object_id", claim.get("object"))
        subject = claim.get("subject_id", claim.get("subject"))
        
        obj = resolved_entities.get(obj, obj)
        subject = resolved_entities.get(subject, subject)
        claim["object_id"] = obj
        claim["subject_id"] = subject
        claim["doc_id"] = document_id
        return claim
     
     
               
    async def _process_document(
        self,
        prompt_args: Dict[str, Any],
        text: str,
        doc_index: int
    ) -> List[Dict[str, Any]]:
        record_delimiter = prompt_args.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        completion_delimiter = prompt_args.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )
        
        template = ChatPromptTemplate.from_messages([
            ("system", self._extraction_prompt)
        ])
        
        prompt_variables = {
            **prompt_args,
            self._input_text_key: text
        }
        
        prompt = template.format(**prompt_variables)
        
        response = self._llm.invoke(prompt)
        
        results = response.content or ""
        
        claims = results.strip().removesuffix(completion_delimiter)
        
        for i in range(self._max_gleanings):
            new_history = text + '\n' + results           
            
            message = [
                {"role": "system", "content" : CONTINUE_PROMPT},
                {"role": "user", "content": new_history}
            ]
            
            response = self._llm.invoke(message)
            
            extension = response.content or ""
            claims += record_delimiter + extension.strip().removesuffix(
                completion_delimiter
            )
            
            
            if i >= self._max_gleanings -1:
                break
            
            new_history = text + '\n' + results
            
            message = [
                {"role": "system", "content": LOOP_PROMPT},
                {"role": "user", "content": new_history}
            ]
            
            old_max_tokens = self._llm.max_tokens
            old_logit_bias = self._llm.logit_bias
            self._llm.max_tokens = self._max_tokens
            self._llm.logit_bias = self._logit_bias
            
            response = self._llm.invoke(message)

            if response.content.strip().upper() != "YES":
                break
            
            self._llm.max_tokens = old_max_tokens
            self._llm.logit_bias = old_logit_bias
            
        results = self._parse_claim_tuples(claims, prompt_args)
            
        for r in results:
            r["doc_id"]  = f"{doc_index}"
            
        return results
    
    
    def _parse_claim_tuples(
        self,
        claims: str,
        prompt_variables: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse claim tuples."""
        record_delimiter = prompt_variables.get(
            self._record_delimiter_key, DEFAULT_RECORD_DELIMITER
        )
        completion_delimiter = prompt_variables.get(
            self._completion_delimiter_key, DEFAULT_COMPLETION_DELIMITER
        )
        tuple_delimiter = prompt_variables.get(
            self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER
        )
        
        def pull_field(index: int, fields: List[str]) -> str | None:
            return fields[index].strip() if len(fields) > index else None
        
        result: List[Dict[str, Any]] = []
        
        claims_values = claims.strip().removesuffix(completion_delimiter).split(record_delimiter)
        
        for claim in claims_values:
            claim = claim.strip().removeprefix("(").removesuffix(")")
            
            if claim == completion_delimiter:
                continue
            
            claim_fields = claim.split(tuple_delimiter)
            result.append({
               "subject_id": pull_field(0, claim_fields),
                "object_id": pull_field(1, claim_fields),
                "type": pull_field(2, claim_fields),
                "status": pull_field(3, claim_fields),
                "start_date": pull_field(4, claim_fields),
                "end_date": pull_field(5, claim_fields),
                "description": pull_field(6, claim_fields),
                "source_text": pull_field(7, claim_fields),
                "doc_id": pull_field(8, claim_fields), 
            })
        return result