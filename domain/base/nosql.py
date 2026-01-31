from typing import Any, Dict, Optional, Generator, List
import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from config.settings import Settings

logger = logging.getLogger(__name__)

class NoSQLDocument:
    """
    Manager class for MongoDB interactions.
    """

    client: MongoClient
    db_name: str
    collection_name: str
    search_index_name: str

    def __init__(
        self, 
        db: str, 
        collection: str, 
        settings: Settings, 
        search_index_name: Optional[str] = None 
    ):
        self.client = MongoClient(settings.MONGODB_HOST)
        self.db_name = db
        self.collection_name = collection
        self.search_index_name = search_index_name or settings.db.BM25_INDEX_NAME
        
    @property
    def collection(self) -> Collection:

        return self.client[self.db_name][self.collection_name]
    
    def get_or_create_document(self, query: dict):

        try: 
            logger.info("Finding document...")

            existing_doc = self.collection.find_one(query)
            
            if existing_doc:
                logger.info("Found document, didn't insert.")
            else:
                logger.info("Didn't find document, inserting...")
                self.collection.insert_one(query)
                logger.info("Inserted successfully.")
                
        except Exception as e:
            logger.error(f"Error during get_or_create: {e}", exc_info=True)
            raise e
    
    def bulk_insert(self, documents: list):

        try: 
            logger.info(f"Inserting {len(documents)} documents...")
            self.collection.insert_many(documents)
            logger.info("Inserting many completed.")

        except Exception as e:
            logger.error(f"Error during bulk insert: {e}", exc_info=True)
            raise e
    
    def bulk_find(self, batch_size: int = 10000, query: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:

        if query is None:
            query = {} 
        
        logger.info(f"Bulk finding with query: {query}")
        
        try:
            cursor = self.collection.find(query).batch_size(batch_size)
            for doc in cursor:
                yield doc
        except Exception as e:
            logger.error(f"Error during bulk find: {e}", exc_info=True)
            raise e    
    
    def create_search_index(self, index: SearchIndexModel):

        try:
            self.collection.create_search_index(index)
            logger.info("Search index created successfully.")
        except Exception as e:
            logger.error(f"Error during search index creation: {e}", exc_info=True)
            raise e
            

    def search(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Performs a bm25 index search with filter on probable_referenced_years
        Only returns the column needed 
        """
        logger.info(f"Searching for: '{query}' with filters: {filters}")
        
        search_operator = {
            "text": {
                "query": query,
                "path": "text_content"
            }
        }
        
        final_search_stage = {}

        if filters and filters.get('probable_referenced_years'):
            logger.info(f"Filtering by year: {filters['probable_referenced_years']}")
            year_int = int(filters['probable_referenced_years'])

            final_search_stage = {
                "$search": {
                    "index": self.search_index_name,
                    "compound": {
                        "must": [
                            search_operator 
                        ],
                        "filter": [
                            {
                                "equals": {
                                    "path": "probable_referenced_years", 
                                    "value": year_int
                                }
                            }
                        ]
                    }
                }
            }
        else:
            final_search_stage = {
                "$search": {
                    "index": self.search_index_name,
                    **search_operator
                }
            }

        pipeline = [
            final_search_stage,
            {
                "$limit": limit
            },
            {
                "$project": {
                    "_id": 0,
                    "markdown_content": 1,
                    "score": {"$meta": "searchScore"}, 
                    "corpus_id": 1,
                    "exact_filename": 1,
                    #"doc_id": 1,
                    "probable_referenced_years": 1
                }
            }
        ]

        try:
            cursor = self.collection.aggregate(pipeline)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []