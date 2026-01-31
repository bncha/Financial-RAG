from typing import List, Generator, Optional, Dict, Any
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.settings import Settings

logger = logging.getLogger(__name__)

class VectorDocument:
    """
    Manager class for Qdrant vector database interactions.
    """

    client: QdrantClient
    collection_name: str

    def __init__(self, collection: str, settings: Settings):
        logger.info(f"Initializing QdrantManager for collection: {collection}")
        
        self.collection_name = collection
        self.settings = settings
        
        self.client = QdrantClient(
            url=self.settings.QDRANT_CLOUD_URL,
            api_key=self.settings.QDRANT_APIKEY
        )

    def get_or_create_collection(self, payload_index_field: str = "probable_referenced_years"):
        """
        Checks if the collection exists. If not, creates it using the 
        global GEN_AI settings for vector size.
        """

        vector_size = self.settings.ai.EMBED_SIZE

        try:
            self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' found.")
        except Exception:
            logger.info(f"Collection '{self.collection_name}' not found. Creating with size {vector_size}...")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                
                if payload_index_field:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=payload_index_field,
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                    logger.info(f"Created payload index on '{payload_index_field}'.")

                logger.info(f"Collection '{self.collection_name}' created successfully.")
                
            except Exception as create_error:
                logger.error(f"Failed to create collection '{self.collection_name}': {create_error}", exc_info=True)
                raise create_error

    def upsert(self, points: List[models.PointStruct]):
        """
        Upserts points into the collection.
        """

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Upserted {len(points)} points to collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert points to collection '{self.collection_name}': {e}", exc_info=True)
            raise e
    
    
    def search(self, query_vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[models.ScoredPoint]:
        """
        Performs a vector similarity search.
        """

        logger.info(f"Searching with vector (dim: {len(query_vector)}), limit: {limit}, filters: {filters}")
        
        query_filter = None
        
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key, 
                        match=models.MatchValue(value=value) 
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter 
            )
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}", exc_info=True)
            return []