from typing import Any, Dict, Iterable, List

import nltk

from verbs.text.chunk.strategies.typing import TextChunk

TextChunk


def run(
    texts: List[str],
    args: Dict[str, Any]
) -> Iterable[TextChunk]:
    for id, text in enumerate(texts):
        sentences = nltk.sent_tokenize(text=text)
        for s in sentences:
            yield TextChunk(
                text_chunk=s,
                source_doc_indices=[id]
            )
    