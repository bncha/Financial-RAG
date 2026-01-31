import click
from config.settings import Settings
from pipelines.digital_data_etl import digital_data_etl

@click.command()
def main():
    """
    Run the digital_data_etl pipeline.
    """
    # Initialize settings (loads from .env automatically)
    settings = Settings()
    
    # Run the pipeline
    digital_data_etl(settings=settings)

if __name__ == "__main__":
    main()
