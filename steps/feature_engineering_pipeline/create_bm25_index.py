from zenml import step
from pymongo.operations import SearchIndexModel
from config.settings import Settings
from domain.base import NoSQLDocument
import logging

logger = logging.getLogger(__name__)

@step
def create_bm25_index(db: str, collection: str, settings: Settings):
    
    mongodb_client = NoSQLDocument(db=db, collection=collection, settings=settings)

    index_name = settings.db.BM25_INDEX_NAME

    indexes = mongodb_client.collection.list_search_indexes()
    for index in indexes:
        if index.get("name") == index_name:
            logger.info(f"Index '{index_name}' exists.")
            return

    search_index_model = SearchIndexModel(
    definition={
        "mappings": {
            "dynamic": False,
            "fields": {
                "text_content": {
                    "type": "string",
                    "analyzer": "lucene.standard"
                },
                "probable_referenced_years": {
                    "type": "number"  
                }
            }
        }
    },
    name=index_name
)
    mongodb_client.create_search_index(search_index_model)