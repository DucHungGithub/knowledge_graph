import json
import logging

from typing import Any, Dict
from langchain_core.language_models import BaseChatModel

from graph.community_reports.community_reports_extractor import CommunityReportsExtractor
from llm import load_openai_llm
from external_utils.uuid import gen_uuid
from verbs.community_summarization.typing import CommunityReport
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


async def run_gi(
    community: str | int,
    text: str,
    level: int,
    args: Dict[str, Any]
) -> CommunityReport | None:
    model_config = args.get("llm", defs.MODEL_CONFIG)
    temperature = args.get("temperature", defs.TEMPERATURE)
    llm = load_openai_llm(model=model_config, temperature=temperature)
    return await run_generate_community_reports(llm=llm, community=community, text=text, level=level, args=args)

async def run_generate_community_reports(
    llm: BaseChatModel,
    community: int| str,
    text: str,
    level: int,
    args: Dict[str, Any]
) -> CommunityReport | None:
    extractor = CommunityReportsExtractor(
        llm=llm,
        extraction_prompt=args.get("extraction_prompt", None),
        max_report_length=args.get("max_report_length", None)
    )
    try:
        results = await extractor.invoke({"input_text": text})
        report = results.structured_output
        if report is None or len(report.keys()) == 0:
            logger.warning("No report found for community: %s", community, exc_info=True)
            return None
            
        return CommunityReport(
            id=gen_uuid(),
            community=community,
            level=level,
            rank=float(report.get("rating", -1)),
            title=report.get("title", f"Community Report: {community}"),
            rank_explanation=report.get("rating_explanation", ""),
            summary=report.get("summary", ""),
            findings=report.get("findings", []),
            full_content=results.output,
            full_content_json=json.dumps(report, indent=4, ensure_ascii=False)
        )
    except Exception as e:
        logger.exception(f"Error processing community: {community}", exc_info=True)
        return None
    