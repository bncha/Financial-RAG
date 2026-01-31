import logging
from typing import Dict
from zenml import step 
from datasets import Dataset
from llm_project.graph_engine import app
import uuid
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@step 
def generate_answers(dataset: Dataset) -> Dataset:
    indices = range(2) 
    small_dataset = dataset.select(indices)
    
    return small_dataset.map(_batch_prompt, batched=True, batch_size=10)


def _process_single(prompt: str) -> Dict:
    try : 
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        inputs = {"messages": [HumanMessage(content=prompt)]}
            
        result = app.invoke(inputs, config=config)
        final_output = result.get("final_output")
        return final_output.model_dump()

    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        return {"logic_steps": "Error", "generated_response": "Error", "summary_sources": "Error", "sources": ["Error"]}


def _batch_prompt(batch) -> Dict:
    prompts = batch["query_text"]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_process_single, prompts))

    return {
        "logic_steps": [r["logic_steps"] for r in results],
        "generated_response": [r["generated_response"] for r in results],
        "summary_sources": [r["summary_sources"] for r in results],
        "sources": [r["sources"] for r in results]
    }
    
