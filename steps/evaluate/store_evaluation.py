from zenml import step
from datasets import Dataset
from domain.base import NoSQLDocument
from config.settings import Settings
from domain.base.eval_document import EvaluationDocument
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@step
def store_evaluation(dataset: Dataset, settings: Settings):
    
    try:
        mongodb_client = NoSQLDocument(
            db=settings.db.MONGO_DB, 
            collection=settings.db.MONGO_COL_EVAL, 
            settings=settings
        )

        # Dataset -> Pandas -> Dicts
        df = dataset.to_pandas()
        records = df.to_dict(orient="records")

        if not records:
            logger.warning("No records found in dataset.")
            return

        validated_data = [
            EvaluationDocument(**rec).model_dump() 
            for rec in records
        ]

        mongodb_client.bulk_insert(validated_data)
        logger.info(f"Successfully stored {len(validated_data)} evaluation records.")

    except Exception as e:
        logger.error(f"Failed to store evaluation: {e}")
        raise e