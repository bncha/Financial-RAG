from zenml import pipeline
from config.settings import Settings
from steps.feature_engineering_pipeline.chunk_and_embed import chunk_and_embed
from steps.feature_engineering_pipeline.create_bm25_index import create_bm25_index


@pipeline(enable_cache=False)
def feature_engineering_pipeline(settings: Settings, 
                                collection_vector: str, 
                                collection_bm25: str,
                                db: str):

    chunk_and_embed(settings=settings, 
                    collection=collection_vector)
    create_bm25_index(settings=settings,
                    collection=collection_bm25, 
                    db=db)