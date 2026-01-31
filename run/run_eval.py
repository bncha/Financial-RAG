import logging
import click
from config.settings import Settings
from pipelines.evaluate import evaluate

logger = logging.getLogger(__name__)

@click.command
def main():
    
    settings = Settings()
    
    evaluate(settings=settings)

if __name__ == "__main__":
    main()