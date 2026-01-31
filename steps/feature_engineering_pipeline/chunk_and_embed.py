from zenml import step
from qdrant_client.http.models import PointStruct
from domain.base import NoSQLDocument, VectorDocument
from config.settings import Settings
from google import genai
from google.genai import types
from typing import List, Dict, Any, Generator
from config.retry_policy import retry_gemini_call
from itertools import batched
import logging

logger = logging.getLogger(__name__)

@step
def chunk_and_embed(collection: str, settings: Settings):
    logger.info(f"Starting chunk_and_embed step. Collection: {collection}")

    qdrant_client = VectorDocument(collection=collection, settings=settings)
    qdrant_client.get_or_create_collection()
    genai_client = genai.Client(api_key=settings.GEMINI_APIKEY)


    documents_gen = _get_content(query={"markdown_content": {"$ne": None}, "corpus_id": {"$ne": None}}, batch_size=100, settings=settings)

    for batch in batched(documents_gen, 50):

        _process_batch(list(batch), genai_client, qdrant_client, settings)


def _process_batch(batch: List[Dict[str, Any]], genai_client: genai.Client, qdrant_client: VectorDocument, settings: Settings):

    try:    
        batch_ids = [int(d["corpus_id"]) for d in batch]

        existing_points = qdrant_client.client.retrieve(
            collection_name=qdrant_client.collection_name,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False
        )
        existing_ids = {p.id for p in existing_points}
        new_docs = [d for d in batch if int(d["corpus_id"]) not in existing_ids]

        if not new_docs:
            logger.info("Batch already exists. Skipping.")
            return

        texts_to_embed = [d["markdown_content"] for d in new_docs]
        embeddings = _embed_content_bulk(genai_client, texts_to_embed, settings)

        points = []
        for doc, vector in zip(new_docs, embeddings):
            points.append(_construct_point(doc, vector, settings))

        qdrant_client.upsert(points) 
        logger.info(f"Successfully processed batch of {len(points)} documents.")

    except Exception as e:
        logger.error(f"Permanent failure in batch: {e}")
        raise e


@retry_gemini_call
def _embed_content_bulk(genai_client: genai.Client, texts: List[str], settings: Settings) -> List[List[float]]:
    """Calls Gemini to embed a LIST of strings with retry logic"""
    logger.info(f"Embedding batch of {len(texts)} texts")
    
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.ai.EMBED_SIZE
    )

    result = genai_client.models.embed_content(
        model=settings.ai.EMBED_MODEL,
        contents=texts,
        config=config
    )
    
    return [e.values for e in result.embeddings]


def _get_content(query: Dict[str, Any], batch_size: int, settings: Settings) -> Generator[Dict[str, Any], None, None]:

    mongodb_client = NoSQLDocument(db=settings.db.MONGO_DB, collection=settings.db.MONGO_COL_PAGE, settings=settings)

    return mongodb_client.bulk_find(query=query, batch_size=batch_size)


def _construct_point(document: Dict[str, Any], vector: List[float], settings: Settings) -> PointStruct: 

    return PointStruct(
        id=int(document["corpus_id"]),
        vector=vector,
        payload={
            "corpus_id": document["corpus_id"],
            "doc_id": document.get("doc_id"),
            "page": document.get("page"),
            "exact_filename": document.get("exact_filename"),
            "probable_referenced_years": document.get("probable_referenced_years"),
            "markdown_content": document["markdown_content"]
        }
    )