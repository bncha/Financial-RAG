from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.types import Command
from config.settings import settings
from langgraph.checkpoint.memory import MemorySaver
import uuid
from llm_project.retrieval import RetrievalService
from llm_project.generation import GenerationService
from langchain_core.messages import RemoveMessage
from langchain_core.messages import get_buffer_string
import logging

checkpointer = MemorySaver()

retrieval_service = RetrievalService(settings)
generation_service = GenerationService(settings)


llm = ChatGoogleGenerativeAI(   
    model=settings.ai.GEN_CHAT, 
    temperature=0, 
    streaming=True,
    google_api_key=settings.GEMINI_APIKEY,
    max_retries=5,

)

small_llm = ChatGoogleGenerativeAI(
    model=settings.ai.GEN_CHAT_LITE,
    temperature=0,
    streaming=False,
    google_api_key=settings.GEMINI_APIKEY,
    max_retries=5

)

# --- 1. SCHEMAS & STRUCTURED OUTPUTS ---


class YearExtraction(BaseModel):
    """Extracted year context from a financial query."""

    year: Optional[int] = Field(
        default=None, 
        description="The specific 4-digit year mentioned or implied. None if topic changed."
    )
    year_reasoning: str = Field(description="Brief explanation of why this year was chosen (e.g., 'Relative to 2023').")


class RouteDecision(BaseModel):
    """
    Schema for the initial classification of a user request.
    Used to determine the operational path and persona of the agent.
    """

    is_financial: bool = Field(
        description="True if the query relates to financial metrics, markets, or corporate performance."
    )
    needs_retrieval: bool = Field(
        description="True if the context in current messages is insufficient and requires external document search."
    )
    context_aware_query: str = Field(
        description="A standalone version of the user's query, incorporating chat history for better search results inside a Financial RAG"
    )


class FinalOutput(BaseModel):
    """
    The finalized, production-ready response delivered to the end-user.
    """

    logic_steps: str = Field(
        description="Internal logic: identifying the exact figure requested and discarding unrelated data.")
    generated_response: str = Field(
        description="The precise answer to the user's question."
    )
    summary_sources: str = Field(
        description="""An expert's summary on the documents retrieved. Go document per document.
        For exemple, provide precisely the name of the products, company, metrics that were discussed in the document and the year of the document."""
    )
    sources: List[str] = Field(
        description="List of document identifiers or PDF names utilized to derive the answer."
    )

class CleanDocument(Dict):
    markdown_content: str 
    source: str 

# --- 2. STATE DEFINITION ---


class AgentState(MessagesState):
    """
    Represents the internal state of the graph.
    """
    is_financial: bool
    needs_retrieval: bool
    retrieved_context: Optional[List[CleanDocument]]
    year_match: Optional[int]
    context_aware_query: str
    final_output: FinalOutput

# --- 3. TOOLS (Logic Only) ---

@tool
def retrieve_financial_reports(query: str, year_match: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Retrieves financial documents from the vector database.
    """

    search_filters = {"probable_referenced_years": year_match} if year_match else {}

    documents = retrieval_service.retrieve(
        query=query, 
        top_k=3, 
        filters=search_filters
    )
    
    return documents

# --- 4. NODES ---

def year_match_node(state: AgentState) -> Dict[str, Any]:
    """
    Manages the 'Sticky Year Filter' state.
    
    Determines if the current query refers to a new year, maintains the 
    previous year context, or clears it based on topic shifts.
    """

    current_year = state.get("year_match", None)
    messages = state.get("messages")

    last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    structured_llm = small_llm.with_structured_output(YearExtraction)

    system_prompt = (
        "You are a financial state manager. Your job is to track the 'Active Year' of a conversation.\n"
        f"Current Active Year: {current_year if current_year else 'None'}\n\n"
        "Rules:\n"
        "1. If the user mentions a specific year (e.g., '2023', 'FY24'), update it.\n"
        "2. If it's a follow-up (e.g., 'What was revenue?'), KEEP the current year.\n"
        "3. If relative (e.g., 'the year before'), calculate it from the current year.\n"
        "4. If they change topic (e.g., 'Who is the CEO?'), return null for year."
    )

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human_msg.content) 
    ])

    logging.info(f"--- [YEAR LOGIC] {result.year_reasoning} -> {result.year} ---")
    
    return {"year_match": result.year}

def check_relevance(state: AgentState) -> Command[Literal["retrieve_node", "generate"]]:
    """
    Orchestration node that classifies the query and directs graph flow.
    """

    structured_llm = small_llm.with_structured_output(RouteDecision)
    system_instruction = (
        "Classify the user request. Determine if it is financial in nature "
        "and if current context suffices to answer."
    )

    result: RouteDecision = structured_llm.invoke(
        [SystemMessage(content=system_instruction)] + state["messages"]  # here the LLM will see "source summary" to infer if retrieval is needed
    )
    
    goto = "retrieve_node" if (result.is_financial and result.needs_retrieval) else "generate"
    
    return Command(
        update={
            "is_financial": result.is_financial,
            "needs_retrieval": result.needs_retrieval,
            "context_aware_query": result.context_aware_query
        },
        goto=goto
    )


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Manual retrieval node that extracts multi-turn context for the RAG tool.
    """

    current_year = state.get("year_match", None)
    
    retrieved_data = retrieve_financial_reports.invoke({
        "query": state["context_aware_query"], 
        "year_match": current_year
    }) 

    context_parts = []
    for i, doc in enumerate(retrieved_data):
        content = doc.get("markdown_content")
        source = doc.get("source")
        
        formatted_doc = (
            f"<document index='{i+1}'>\n"
            f"<source>{source}</source>\n"
            f"<content>\n{content}\n</content>\n"
            f"</document>"
        )
        context_parts.append(formatted_doc)

    final_context_str = "\n\n".join(context_parts)
    
    # If no docs found, provide a clear text indicator
    if not final_context_str:
        final_context_str = "<document>\nNo financial reports found matching the criteria.\n</document>"

    tool_msg = ToolMessage(
        content=final_context_str,
        tool_call_id=uuid.uuid4() 
    )
    
    return {"retrieved_context": tool_msg}


def get_system_prompt(is_financial: bool, needs_retrieval: bool) -> str:
    """
    Generates a system instruction based on the classified intent.
    """

    if is_financial :
        return (
            "You are a Senior Financial Analyst. Your hallmark is precision, transparancy and brevity. Always cite your sources\n"
            "STRICT RULES:\n"
            "1. ONLY answer the specific question asked. If the user asks for 'Price X', do not provide 'Price Y' or 'Total Z'.\n"
            "2. DO NOT summarize the document. DO NOT provide context about other acquisitions or company performance unless explicitly requested.\n"
            "3. If the context contains multiple data points, filter out everything except the exact match for the user's query.\n"
            "4. If you cannot find the specific number, state that specifically rather than providing a 'nearby' or 'aggregate' number.\n"
            "5. NO filler sentences. Start directly with the answer."
        )
    return (
        "You are a helpful, friendly everyday assistant. "
        "Answer the question clearly using simple language, avoiding heavy jargon."
)

def generate_answer(state: AgentState) -> Dict[str, Any]:
    """
    Final synthesis node that generates a structured response using dynamic prompting.
    """

    structured_llm = llm.with_structured_output(FinalOutput)
    
    tool_messages = state.get("retrieved_context", [])

    if not state.get("needs_retrieval"): 
        dialogue_history = state["messages"] 
        # if there is no retrieval context, there is still dialogue context to pass
        context_str = get_buffer_string(dialogue_history) 
    else: 
        context_str = tool_messages.content

    system_prompt = get_system_prompt(state.get("is_financial"), state.get("needs_retrieval"))
    
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]

    user_query = human_messages[-1].content

    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context_str}\n\n Real Question: {user_query}, Context Aware Question (rephrased): {state.get('context_aware_query', '')}")
    ])

    history_text = f"Answer: {response.generated_response}\n Rational: {response.logic_steps}\n Sources Summary: {response.summary_sources}\n Sources: {', '.join(response.sources)}"
    ai_msg = AIMessage(content=history_text)

    messages_to_remove = [
        RemoveMessage(id=m.id) for m in state["messages"][:-4]
    ]
    
    return {
        "final_output": response,
        "messages": messages_to_remove + [ai_msg] # add full ai message to keep dialogue integrity in the "messages" key
    }
    

# --- 5. GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

workflow.add_node("year_match_node", year_match_node)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generate", generate_answer)

workflow.add_edge(START, "year_match_node")
workflow.add_edge("year_match_node", "check_relevance")
workflow.add_edge("retrieve_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=checkpointer)