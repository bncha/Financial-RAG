from zenml import pipeline
from config.settings import Settings
from steps.digital_data_etl.read_data import read_corpus
from steps.digital_data_etl.get_or_create_document import get_or_create_document

@pipeline(enable_cache=False)
def digital_data_etl(settings: Settings):
    """
    Pipeline to read data from HF and ingest into MongoDB.
    """
    # Read the corpus from Hugging Face
    corpus = read_corpus(settings=settings)
    
    # Process and ingest documents
    get_or_create_document(
        corpus=corpus, 
        db=settings.db.MONGO_DB, 
        collection=settings.db.MONGO_COL_PAGE, 
        settings=settings
    )


