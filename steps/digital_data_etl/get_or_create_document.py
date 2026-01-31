from domain.base import NoSQLDocument
from config.settings import Settings
from domain.base import PageDocument
from datasets import Dataset
from zenml import step
from google import genai
from typing import Any
import markdown2
from bs4 import BeautifulSoup
from config.retry_policy import retry_gemini_call
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@step
def get_or_create_document(corpus: Dataset, db: str, collection: str, settings: Settings):
    """
    Batch processing of documents in the corpus. 
    Checks if the document is already registered in the database.
    If not, it calls the LLM to extract the text and saves it to the database.
    """

    mongodb_client = NoSQLDocument(db=db, collection=collection, settings=settings)
    genai_client = genai.Client(api_key=settings.GEMINI_APIKEY)
    batch_size = 50 
    
    for i in range(0, len(corpus), batch_size):

        batch_slice = corpus.select(range(i, min(i + batch_size, len(corpus))))
        
        # Extracting IDs we are interested in for this batch
        ids_to_check = [int(cid) for cid in batch_slice["corpus-id"]]
        
        # Efficient DB Query: Only fetch IDs that already exist in THIS batch
        existing_docs = mongodb_client.collection.find(
            {"corpus_id": {"$in": ids_to_check}},
            {"corpus_id": 1} 
        )
        existing_ids = {doc["corpus_id"] for doc in existing_docs}

        # Filtering the batch: Only process what we DON'T have
        to_process = [row for row in batch_slice if row["corpus-id"] not in existing_ids]

        if not to_process:
            logger.info(f"Batch {i//batch_size}: All documents already exist. Skipping.")
            continue

        # Multithread process the LLM calls for the missing documents
        with ThreadPoolExecutor(max_workers=10) as executor:
            new_documents = list(executor.map(
                lambda row: process_single_row(row, genai_client, settings), 
                to_process
            ))

        # Bulk insert the new documents after filtering out multithread erros
        valid_docs = [d for d in new_documents if d is not None]
        if valid_docs:
            mongodb_client.bulk_insert(valid_docs)
    
    
def process_single_row(row, client, settings):
    """Encapsulates the logic for a single row for threading."""
    try:
        # Apply LLM OCR 
        ocr_text = _llm_ocr(client, row["image"], settings)
        
        # Validate with Pydantic and return dict
        return PageDocument(
            corpus_id=int(row["corpus-id"]),
            doc_id=str(row["doc-id"]),
            page=row["page"],
            exact_filename=row["image_filename"],
            probable_referenced_years=row["probable_referenced_years"],
            markdown_content=ocr_text,
            text_content=_create_cleaned_text(ocr_text)
        ).model_dump()

    # If the threading fails in any way, we return None, which is filtered out later
    except Exception as e:
        logger.error(f"Failed processing row: {e}")
        return None

def _create_cleaned_text(ocr_text: str) -> str:
    html = markdown2.markdown(ocr_text)
    soup = BeautifulSoup(html, "html.parser")
    text_content = soup.get_text()
    return text_content


@retry_gemini_call
def _llm_ocr(client : genai.Client, image : Any, settings: Settings) -> str:
    """Calls an LLM to do OCR"""

    model = settings.ai.GEN_OCR

    prompt =  """You are an OCR printer. 
    Your role is to copy the exact text you see in the image in markdown format. Do your best.
    Do not make up new text that isn't on the page. Except for table, for table, your role is to describe, in a few sentences, the main aspects of its content.
    Before sending in your copy, make sure your output has followed these instructions."""

    response = client.models.generate_content(
        model=model,
        contents = [prompt, image]
    )
    return response.text
