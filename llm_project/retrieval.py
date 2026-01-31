import logging
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder
from config.retry_policy import retry_gemini_call
from config.settings import Settings
from domain.base import NoSQLDocument, VectorDocument

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, settings: Settings):

        self.settings = settings
        self.genai_client = genai.Client(api_key=settings.GEMINI_APIKEY)
        self.model = settings.ai.GEN_RETRIEVAL
        

        self.nosql_client = NoSQLDocument(
            db=settings.db.MONGO_DB, 
            collection=settings.db.MONGO_COL_PAGE,
            settings=settings
        )
        self.vector_client = VectorDocument(
            collection=settings.db.QDRANT_EMBED, 
            settings=settings
        )
        self.vector_size = settings.ai.EMBED_SIZE
        self.embed_model = settings.ai.EMBED_MODEL

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    @retry_gemini_call
    #legacy
    def extract_filter_year(self, query: str, history: Optional[List[Dict]] = None, current_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """
                                
                               ------LEGACY------

        Extracts filters (e.g., year) from the conversation history and current query using Gemini.
        1. Regex Search
        2. If no match, LLM Semantic Search based on the last 3 messages and Year already in the filter
        """

        logger.info("Extracting filters from conversation...")

        # regex search
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            new_year = int(year_match.group(0))
            logger.info(f"Sticky Filter Updated via Regex: {new_year}")
            return {"probable_referenced_years": new_year}


        if history is None: history = []
        if current_filter is None: current_filter = {}

        # format history
        history_lines = []
        for turn in history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines)


        prompt = f"""
        You are a conversational state manager for financial analysis.
        
        Current Status:
        - The user is currently analyzing data for Year: {current_filter}
        
        Task:
        Determine the correct "year" filter for the NEW query.
        
        Rules:
        1. **KEEP**: If the new query is a follow-up (e.g., "What was the revenue?", "and operating cost?"), KEEP the current year.
        2. **RELATIVE**: If the query is relative (e.g., "What about the year before?"), calculate the new year based on the Current Year.
        3. **CLEAR**: If the query changes topic completely (e.g., "What does this company do?", "Who is the CEO?"), return empty {{}}.
        
        Conversation History:
        {history_str}
        
        New Query:
        {query}
        
        Output JSON only (e.g., {{"probable_referenced_years": 2020}} or {{}})
        
        Output:"""

        try:
            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            text = response.text
            filter = json.loads(text)
            logger.info(f"Extracted filters: {filter}")
            return filter
        except Exception as e:
            logger.error(f"Error extracting filters: {e}", exc_info=True)
            return current_filter

    
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Orchestrates the retrieval process:
        1. Generate multiple similar queries
        2. Embed query
        3. Parallel retrieval (BM25 + Vector)
        4. Reciprocal Rank Fusion (RRF)
        5. Reranking
        """

        logger.info(f"Starting retrieval for query: '{query}' with filters: {filters}")
        
        # Generate multiple queries to maximize search relevance
        queries = self._generate_query(query)
        logger.info(f"Generated {len(queries)} variations: {queries}")
        
        all_ranked_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:  
            
            future_to_query = {executor.submit(self._embed_query, q): q for q in queries}

            query_pairs = []   
            for f in as_completed(future_to_query):
                text = future_to_query[f]
                try : 
                    vector = f.result()
                    query_pairs.append((text, vector))
                except Exception as e: 
                    logging.error(f"Embedding error : {e} ")

            future_threads = []
            for q_text, q_vector in query_pairs:
                future_threads.append(executor.submit(self._bm25_search, q_text, limit = top_k * 5, filters=filters))
                future_threads.append(executor.submit(self._vector_search, q_vector, limit = top_k * 5, filters=filters))
                
            for future in as_completed(future_threads):
                try: 
                    result = future.result()
                    list_of_docs = result['data']
                    if list_of_docs: 
                        all_ranked_list.append(list_of_docs)
                except Exception as e:
                    logging.error(f"Retrieving list of  from vector search or bm25 search error : {e}")

        fused_results = self._reciprocal_rank_fusion(all_ranked_list, k=top_k*3)
        logger.info(f"FUSION complete. Returning {len(fused_results)} documents.")
        
        reranked_results = self._rerank(query, fused_results, top_k=top_k)

        logger.info(f"RERANK complete. Returning {len(reranked_results)} documents.")

        final_docs = []
        for doc in reranked_results:

            final_docs.append({
                "markdown_content": doc["markdown_content"],
                "source": doc["exact_filename"]
            })
        logger.info(f"END OF RETRIEVAL PROCESS")
        return final_docs

    @retry_gemini_call
    def _generate_query(self, query: str) -> List[str]:

        prompt = f"""You are a financial research assistant.
        Generate 3 different search queries based on the user question to maximize retrieval from financial reports.
        Include:
        1. A keyword-focused variation.
        2. A conceptual/semantic variation.
        3. A simplified variation.
        
        User Question: {query}
        
        Output only the 3 queries, one per line. Do not number them."""

        try: 
            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )

            generated_queries = [line.strip() for line in response.text.split("\n") if line.strip()]

            # Combine and Deduplicate (in case LLM repeats the user query)
            return list(dict.fromkeys([query] + generated_queries))
        except Exception as e:
            logger.error(f"Error generating query: {e}", exc_info=True)
            return [query]

    @retry_gemini_call
    def _embed_query(self, query: str) -> List[float]:
        try:
            result = self.genai_client.models.embed_content(
                model=self.embed_model,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.vector_size
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query: {e}", exc_info=True)
            raise e

    def _bm25_search(self, query: str, limit: int, filters: Dict) -> Dict:
        results = self.nosql_client.search(query, limit=limit, filters=filters)
        logger.info(f"BM25 Retrieval complete. Returning {len(results)} documents.")
        return {'type': 'bm25', 'data': results}


    def _vector_search(self, query_vector: List[float], limit: int, filters: Dict) -> Dict:
        results = self.vector_client.search(query_vector, limit=limit, filters=filters)

        # Normalize vector results to match BM25 structure
        normalized_results = []
        for point in results:
            payload = point.payload
            normalized_results.append({
                "markdown_content": payload.get("markdown_content"),
                "corpus_id": payload.get("corpus_id"),
                #"doc_id": payload.get("doc_id"),
                "exact_filename": payload.get("exact_filename"),
                "probable_referenced_years": payload.get("probable_referenced_years"),
                "score": point.score
            })
        logger.info(f"Vector Retrieval complete. Returning {len(results)} documents.")
        
        return {'type': 'vector', 'data': normalized_results}

    def _reciprocal_rank_fusion(self, list_of_ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:

        fusion_scores = defaultdict(float)
        doc_map = {} # To keep track of the document as we see them
        
        for ranked_list in list_of_ranked_lists:
            # Iterate through the docs for one specific query and retrieval method
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get('corpus_id')
                if doc_id is None: 
                    continue
                
                score = 1 / (k + rank + 1)
                
                fusion_scores[doc_id] += score
                
                # Store the doc object if we haven't seen it yet
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by accumulated fusion score
        sorted_doc_ids = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)
        
        # Return the full documents (text + metadata)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]


    def _rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        if not documents:
            return []
            
        pairs = [[query, doc['markdown_content']] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
            
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_docs[:top_k]
