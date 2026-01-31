from domain.base import HFDatasetService
from datasets import Dataset
from config.settings import Settings
from zenml import step
import duckdb 
from typing import Optional
import logging
logger = logging.getLogger(__name__)


@step
def get_testset_data(settings: Settings, 
                query_name: Optional[str] = "queries", 
                answer_name: Optional[str] = "qrels") -> Dataset: 

    handler = HFDatasetService(settings)

    corpus_url = "ibm-research/REAL-MM-RAG_FinReport_BEIR"

    testset_q = handler.load_data(path=corpus_url, name=query_name, split="test")

    testset_a =  handler.load_data(path=corpus_url, name=answer_name, split="test")

    arrow_q = testset_q.with_format("arrow")[:]
    arrow_a = testset_a.with_format("arrow")[:]

    
    duckdb_query = """
    WITH base AS (
        SELECT 
            a."query-id" as query_id,
            a."corpus-id" as corpus_id,
            a.answer,
            q.query AS original,
            q.rephrase_level_1,
            q.rephrase_level_2,
            q.rephrase_level_3
        FROM arrow_a AS a
        INNER JOIN arrow_q AS q 
            ON a."query-id" = q."query-id"
    )
    
    SELECT 
        query_id,
        'original' AS difficulty,
        original AS query_text,
        answer
    FROM base
    
    UNION ALL
    
    SELECT 
        query_id,
        'level_1' AS difficulty,
        rephrase_level_1 AS query_text,
        answer
    FROM base

    UNION ALL
    
    SELECT 
        query_id,
        'level_2' AS difficulty,
        rephrase_level_2 AS query_text,
        answer
    FROM base

    UNION ALL
    
    SELECT 
        query_id,
        'level_3' AS difficulty,
        rephrase_level_3 AS query_text,
        answer
    FROM base
    """

    joined_arrow_table = duckdb.query(duckdb_query).to_arrow_table()

    final_dataset = Dataset(joined_arrow_table)

    return final_dataset


