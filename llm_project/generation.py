import logging
from typing import List
from google import genai
from google.genai import types
from typing import Dict
import datetime
import json
from pydantic import BaseModel, Field
from config.retry_policy import retry_gemini_call
from config.settings import Settings

logger = logging.getLogger(__name__)

class JudgeContent(BaseModel):
    reasoning: str = Field(description="Brief explanation of why the score was given.")
    answer_accuracy: int = Field(description="The rating (0, 2, or 4).")

class JudgeResponse(JudgeContent):
    time: datetime.datetime = Field(default_factory= datetime.datetime.now, description="The time the response was generated.")
    model: str = Field(description="The model used to generate the response.")

class GenerationService:
    def __init__(self, settings: Settings):
        self.client = genai.Client(api_key=settings.GEMINI_APIKEY)
        self.model_chat = settings.ai.GEN_CHAT
        self.model_judge = settings.ai.GEN_JUDGE
        
    @retry_gemini_call
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generates a response using Gemini based on the query and retrieved context.
        """

        logger.info("Generating response...")
        
        prompt = f"""You are a helpful assistant for analyzing financial reports.
                Use the following context to answer the user's question.
                If the answer is not in the context, say you don't know.
                
                Context:
                {context}
                
                Question:
                {query}
                
                Answer:"""

        try:
            response = self.client.models.generate_content_stream(
                model=self.model_chat,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        ),
                    ]
                )
            )
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
            logger.info("Predicted answer generated successfully.")
            return full_response
        except Exception as e:
            logger.error(f"Error generating predicted answer : {e}", exc_info=True)
            return "I encountered an error while generating the response."

    @retry_gemini_call
    def llm_judge_qa(self, q: str, truth: str, pred: str) -> JudgeResponse:
        """
        Generates a JUDGE response using Gemini based on the query and retrieved context.
        """

        logger.info("LLM-AS-A-JUDGE QUERYING")

        try: 
            judge_prompt = f"""
                Question: {q}
                Reference Answer: {truth}
                User Answer: {pred}

                Instruction: You are a world class state of the art assistant for rating 
                a User Answer given a Question. The Question is completely answered by the Reference Answer.
                Say 4, if User Answer is full contained and equivalent to Reference Answer
                in all terms, topics, numbers, metrics, dates and units.
                Say 2, if User Answer is partially contained and almost equivalent to Reference Answer
                in all terms, topics, numbers, metrics, dates and units.
                Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,
                numbers, metrics, dates and units or the User Answer do not answer the question.
                Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above. Return JSON.
                """
            
            response = self.client.models.generate_content(
                    model=self.model_judge,
                    contents=judge_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": JudgeContent.model_json_schema()
                    }
            )
            logger.info("LLM-as-a-judge generated successfully.")

            llm_data = json.loads(response.text)

            return JudgeResponse(model = self.model_judge,
            **llm_data)
                
        except Exception as e:
            logger.error(f"LLM-as-a-judge error generating response: {e}", exc_info=True)

            return JudgeResponse(
                model=self.model_judge,
                reasoning=f"I encountered an error while generating the response: {str(e)}",
                answer_accuracy=0
            )