from pydantic import BaseModel
from datetime import datetime
from typing import List

class EvaluationDocument(BaseModel):
    query_id: int
    difficulty: str
    query_text: str
    answer: str
    generated_response: str
    time: datetime
    reasoning: str
    answer_accuracy: int
    model: str
    logic_steps: str 
    summary_sources: str
    sources: List[str] 