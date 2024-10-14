import logging

import colorlog
from langchain_core.language_models import BaseChatModel

from dgraph_ingest.prompt import HISTORY_UPDATE_PROMPT



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

class InformationGen:
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = HISTORY_UPDATE_PROMPT
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        
    def generate(
        self,
        old_information: str = "",
        new_information: str = ""
    ) -> str:
        """
        Generate the new combined descriptions
        """
        if old_information.strip() == "":
            return new_information
        
        system_prompt = ""
        
        try:
            system_prompt = self.system_prompt.format(
                old_information=old_information, new_information=new_information
            )
            
            question_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            response = self.llm.invoke(
                question_messages
            )
            
            return response.content
        except Exception:
            logger.exception("Exception in generating questions")
            return new_information