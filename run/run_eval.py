import logging
from config.settings import Settings
from pipelines.evaluate import evaluate

logger = logging.getLogger(__name__)

def main():
    
    settings = Settings()
    
    evaluate.with_options(enable_cache=False)(settings=settings)

if __name__ == "__main__":
    main()