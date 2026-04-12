"""
LLM client — unified interface for OpenAI and Anthropic APIs.

Handles prompt formatting, API calls, token counting, and response parsing.
Tracks token usage for efficiency evaluation.
"""

import json
import logging
from typing import Optional, Dict
from dataclasses import dataclass, field

from config.settings import DEFAULT_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

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
    Unified LLM client supporting OpenAI and Anthropic.

    Handles:
    - API key management
    - Prompt formatting
    - Token tracking for efficiency evaluation
    - Graceful fallback if API keys are missing
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.llm
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client_initialized = False
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def _init_openai(self):
        if self._openai_client is not None:
            return
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except (ImportError, Exception) as e:
            logger.warning(f"OpenAI client init failed: {e}")

    def _init_anthropic(self):
        if self._anthropic_client is not None:
            return
        try:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        except (ImportError, Exception) as e:
            logger.warning(f"Anthropic client init failed: {e}")

    def _init_gemini(self):
        if self._gemini_client_initialized:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini_client_initialized = True
        except (ImportError, Exception) as e:
            logger.warning(f"Gemini client init failed: {e}")

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a query to the configured LLM provider.

        Returns LLMResponse with content, token counts, and latency.
        """
        import time
        start = time.perf_counter()

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        if self.config.provider == "openai":
            response = self._query_openai(prompt, system_prompt, temp, max_tok)
        elif self.config.provider == "anthropic":
            response = self._query_anthropic(prompt, system_prompt, temp, max_tok)
        elif self.config.provider == "gemini":
            response = self._query_gemini(prompt, system_prompt, temp, max_tok)
        else:
            # Fallback: return empty response
            logger.warning(f"Unknown provider: {self.config.provider}")
            response = LLMResponse(content="[No LLM provider configured]")

        response.latency_ms = (time.perf_counter() - start) * 1000
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_calls += 1

        return response

    def _query_openai(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self._init_openai()
        if not self._openai_client:
            return LLMResponse(content="[OpenAI not available]")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            usage = response.usage
            return LLMResponse(
                content=response.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                model=self.config.model_name,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(content=f"[OpenAI error: {e}]")

    def _query_anthropic(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self._init_anthropic()
        if not self._anthropic_client:
            return LLMResponse(content="[Anthropic not available]")

        try:
            response = self._anthropic_client.messages.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                system=system_prompt if system_prompt else "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model=self.config.model_name,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(content=f"[Anthropic error: {e}]")

    def _query_gemini(
        self, prompt: str, system_prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self._init_gemini()
        if not self._gemini_client_initialized:
            return LLMResponse(content="[Gemini not available]")

        import google.generativeai as genai
        try:
            # Build configuration
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Gemini models accept a system instruction natively
            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=generation_config
            )

            response = model.generate_content(prompt)
            # Gemini Python SDK doesn't always strictly track token counts cleanly in the free tier basic return objects, 
            # but we can mock or pull whatever metadata is available
            return LLMResponse(
                content=response.text,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self.config.model_name,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(content=f"[Gemini error: {e}]")

    def parse_json_response(self, response: LLMResponse) -> Optional[Dict]:
        """Parse a JSON response from the LLM (used by judge)."""
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
