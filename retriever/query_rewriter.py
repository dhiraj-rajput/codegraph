"""
Query Rewriter — Leverages an LLM to expand natural language queries into
keyword-rich, structurally contextualized search tokens (HyDE + Keyword Extraction).
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from config.settings import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """The structured output of a rewritten query."""
    original_query: str
    keywords: List[str]                  # General search terms (for BM25)
    expected_symbols: List[str]          # Exact function/class names expected (for Symbol Index)
    expected_files: List[str]            # Probable file paths/names
    hyde_snippet: str = ""               # High-level synthetic code answer (for Vector search)


class LLMQueryExpander:
    """
    Expands a vague natural language query into a highly specific
    structural search parameter using a fast LLM (e.g., Gemini 1.5 Flash).
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.llm
        self.client_type = None
        self.client = None
        import os

        if self.config.provider == "openai" and OpenAI:
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)
                self.client_type = "openai"
            else:
                logger.warning("OPENAI_API_KEY missing. Query expansion disabled.")
        elif self.config.provider == "huggingface":
            # Using HF Inference API via OpenAI-compatible endpoint
            token = os.environ.get("HUGGINGFACE_TOKEN")
            if token:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://api-inference.huggingface.co/v1/",
                    api_key=token,
                )
                self.client_type = "huggingface"
                logger.info(f"Initialized Hugging Face provider with {self.config.hf_model_path}")
            else:
                logger.warning("HUGGINGFACE_TOKEN missing. Query expansion disabled.")
        elif self.config.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if key:
                import google.generativeai as genai
                genai.configure(api_key=key)
                self.client_type = "gemini"
                # Store the model directly as client for simplicity
                self.client = genai.GenerativeModel(
                    self.config.fast_model_name,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                    }
                )
            else:
                logger.warning("GEMINI_API_KEY missing. Query expansion disabled.")
        else:
            logger.warning("No valid LLM client found or configured. Query expansion disabled.")

    def expand(self, query: str) -> ExpandedQuery:
        """
        Rewrite the query using an LLM. Fallback to basic extraction if unavailable.
        """
        if not self.client_type:
            # Fallback
            from retriever.vectorless_retriever import extract_identifiers, extract_keywords
            return ExpandedQuery(
                original_query=query,
                keywords=extract_keywords(query),
                expected_symbols=extract_identifiers(query),
                expected_files=[],
            )

        prompt = f"""You are a senior software engineer optimizing a search query for a highly precise Code Repository Retrieval System.
The user is asking: {query}

Your job is to rewrite this query into structured arrays to maximize search recall.
Think about the vocabulary mismatch: if they ask about "routing", the code might use `Router`, `endpoint`, `@app.get`, etc.

Output MUST be a valid JSON object with the following schema:
{{
  "keywords": ["list", "of", "general", "BM25", "search", "terms", "synonyms"],
  "expected_symbols": ["list", "of", "CamelCase", "or", "snake_case", "identifier", "names"],
  "expected_files": ["list", "of", "probable", "file", "paths", "like", "auth/routes.py", "utils.py"]
}}
"""

        try:
            if self.client_type in ["openai", "huggingface"]:
                model = (self.config.hf_model_path 
                         if self.client_type == "huggingface" 
                         else self.config.fast_model_name)
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={ "type": "json_object" } if self.client_type == "openai" else None
                )
                raw = response.choices[0].message.content
            elif self.client_type == "gemini":
                response = self.client.generate_content(prompt)
                raw = response.text
                
            data = json.loads(raw)
            
            # Merge original query tokens into keywords to be safe
            base_kws = query.lower().split()
            keywords = list(set([k.lower() for k in data.get("keywords", [])] + base_kws))
            
            eq = ExpandedQuery(
                original_query=query,
                keywords=keywords,
                expected_symbols=data.get("expected_symbols", []),
                expected_files=data.get("expected_files", []),
            )
            logger.info(f"Expanded query: symbols={len(eq.expected_symbols)} kws={len(eq.keywords)}")
            return eq
            
        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}")
            from retriever.vectorless_retriever import extract_identifiers, extract_keywords
            return ExpandedQuery(
                original_query=query,
                keywords=extract_keywords(query),
                expected_symbols=extract_identifiers(query),
                expected_files=[],
            )
