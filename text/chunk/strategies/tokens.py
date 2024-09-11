from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple
import tiktoken
import logging

from verbs.text.chunk.strategies.typing import TextChunk


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""
    
    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    encode: Callable[[str], List[int]]
    """ Function to decode a list of token ids to a string"""
    decode: Callable[[List[int], str]]
    """ Function to encode a string to a list of token ids"""



# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# Adapted from - https://github1s.com/microsoft/graphrag/blob/main/graphrag/index/verbs/text/chunk/strategies/tokens.py#L64-L81
def split_text_on_tokens(
    texts: List[str],
    enc: Tokenizer,
) -> List[TextChunk]:
    """Split incoming text and return chunks."""
    
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)
        mapped_ids.append((source_doc_idx, encoded))

    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]
    
    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        chunk_text = enc.decode([id for _, id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


def run(
    input: List[str],
    args: Dict[str, Any],
) -> Iterable:
    import config as defs
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)
    
    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=enc.encode,
            decode=enc.decode
        )
    )


