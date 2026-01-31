from zenml import step
from datasets import Dataset
from typing import Dict
from llm_project.generation import GenerationService
from config.settings import Settings
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@step
def evaluate_answer(dataset: Dataset, settings: Settings) -> Dataset:

    judge = GenerationService(settings)
    
    logger.info(f"Starting evaluation on {len(dataset)} rows...")

    return dataset.map(
        _judge_batch, 
        batched=True, 
        batch_size=10, 
        fn_kwargs={"judge": judge} 
    )


def _judge_batch(batch, judge: GenerationService) -> Dict:

    iterator = list(zip(
        batch["query_text"], 
        batch["answer"], 
        batch["generated_response"]
    ))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # executor.map maintains the order of the results
        results = list(executor.map(
            lambda x: judge.llm_judge_qa(x[0], x[1], x[2]), 
            iterator
        ))

    dict_results = [res.model_dump() for res in results]

    # list to dict
    return {
        key: [row[key] for row in dict_results]
        for key in dict_results[0].keys()
    }