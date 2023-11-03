import marvin
from marvin import ai_fn
from rag.config import MARVIN_MODEL_NAME, MARVIN_MODEL_TEMPERATURE

marvin.settings.llm_model = MARVIN_MODEL_NAME
marvin.settings.llm_temperature = MARVIN_MODEL_TEMPERATURE

@ai_fn
def get_language(text: str) -> str:
    """
    Given `text`, returns the language in which it is written.
    """ 