import logging
from config.settings import Settings
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline

logger = logging.getLogger(__name__)

def main():
    settings = Settings()
    
    feature_engineering_pipeline.with_options(enable_cache=False)(settings=settings,
                                collection_vector=settings.db.QDRANT_EMBED,
                                collection_bm25=settings.db.MONGO_COL_PAGE,
                                db=settings.db.MONGO_DB)

if __name__ == "__main__":
    main()