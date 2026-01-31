from domain.base import HFDatasetService
from datasets import Dataset, DatasetDict
from typing import Union, Optional
from config.settings import Settings 
from zenml import step
import logging
import re

logger = logging.getLogger(__name__)

@step
def read_corpus(settings: Settings, name: Optional[str] = "corpus", split: Optional[str] = "test") -> Union[Dataset, DatasetDict]: 
    """Extract the corpus from HF API and prepare it for the pipeline"""

    logger.info("Running read_corpus")
    
    handler = HFDatasetService(settings)

    corpus_url = "ibm-research/REAL-MM-RAG_FinReport_BEIR"
    
    corpus_dataset = handler.load_data(path=corpus_url, name=name, split=split)

    corpus_dataset = corpus_dataset.map(_get_year)
    corpus_dataset = corpus_dataset.map(_get_page)
    corpus_dataset = corpus_dataset.select(range(2))

    return corpus_dataset 

def _get_year(data):

    publication_year = int(data["doc-id"][-4:])

    data["probable_referenced_years"] = [publication_year, publication_year - 1, publication_year - 2, publication_year - 3]

    return data

def _get_page(data):
    
    page = int(re.search(r"_page_(\d+)", data["image_filename"]).group(1))

    data["page"] = page

    return data
