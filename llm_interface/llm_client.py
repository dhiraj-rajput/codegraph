"""
LLM client — unified interface for local Ollama APIs.

Handles prompt formatting, API calls, and response parsing.
"""

import json
import time
import logging
from typing import Optional, Dict
from dataclasses import dataclass

from config.settings import DEFAULT_CONFIG, OLLAMA_HOST

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLMClient:
    """
    Unified LLM client for Ollama.
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.llm
        self._ollama_client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def _init_ollama(self):
        if self._ollama_client is not None:
            return
        try:
            import ollama
            from ollama import Client
            self._ollama_client = Client(host=OLLAMA_HOST)
        except (ImportError, Exception) as e:
            logger.warning(f"Ollama client init failed: {e}. Is ollama installed?")

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a query to the local Ollama provider.
        """
        start = time.perf_counter()

        temp = temperature if temperature is not None else self.config.temperature
        
        self._init_ollama()
        if not self._ollama_client:
            return LLMResponse(content="[Ollama not available]")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Using ollama-python SDK
            response = self._ollama_client.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            )
            
            content = response.get("message", {}).get("content", "")
            # Ollama provides prompt_eval_count and eval_count
            in_tokens = response.get("prompt_eval_count", 0)
            out_tokens = response.get("eval_count", 0)

            result = LLMResponse(
                content=content,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                total_tokens=in_tokens + out_tokens,
                model=self.config.model_name,
            )
        except Exception as e:
            logger.error(f"Ollama API error (is it running?): {e}")
            result = LLMResponse(content=f"[Ollama error: {e}]")

        result.latency_ms = (time.perf_counter() - start) * 1000
        self._total_input_tokens += result.input_tokens
        self._total_output_tokens += result.output_tokens
        self._total_calls += 1

        return result

    def parse_json_response(self, response: LLMResponse) -> Optional[Dict]:
        """Parse a JSON response from the LLM."""
        text = response.content.strip()
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:200]}")
            return None

    # ── Usage Stats ──────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def usage_stats(self) -> Dict:
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset_stats(self):
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
