from tenacity import retry, before_sleep_log, stop_after_attempt, wait_exponential, retry_if_exception, wait_random
from google.genai.errors import APIError
import logging 

logger = logging.getLogger(__name__)

def _is_retryable_error(exception: Exception) -> bool:
    if isinstance(exception, (APIError)):
        code = getattr(exception, "status_code", 0)
        return code == 429 or code >= 500
    return False

retry_gemini_call = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=6, max=100) + wait_random(0,2),
    retry=retry_if_exception(_is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
