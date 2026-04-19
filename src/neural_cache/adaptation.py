from __future__ import annotations

import re
from typing import Any

from neural_cache.config import AdaptationConfig, ResponseAdaptationMode


class ResponseAdaptor:
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self._llm_client = None

    def adapt(
        self,
        cached_query: str,
        cached_response: str,
        new_query: str,
        similarity: float,
    ) -> str:
        if self.config.mode == ResponseAdaptationMode.NONE:
            return cached_response

        if similarity >= self.config.passthrough_threshold:
            return cached_response

        if self.config.mode == ResponseAdaptationMode.TEMPLATE_FILL:
            return self._template_fill(cached_response, new_query)
        elif self.config.mode == ResponseAdaptationMode.LLM_REFINE:
            return self._llm_refine(cached_query, cached_response, new_query)
        elif self.config.mode == ResponseAdaptationMode.HYBRID:
            if similarity >= 0.92:
                return self._template_fill(cached_response, new_query)
            else:
                return self._llm_refine(cached_query, cached_response, new_query)

        return cached_response

    def _template_fill(self, response: str, new_query: str) -> str:
        adapted = response
        query_phrases = self._extract_key_phrases(new_query)
        if len(query_phrases) > 0:
            context = f'Based on information related to: "{query_phrases[0]}"\n\n'
            adapted = context + adapted
        return adapted

    def _llm_refine(
        self,
        cached_query: str,
        cached_response: str,
        new_query: str,
    ) -> str:
        if self._llm_client is None:
            return self._template_fill(cached_response, new_query)

        prompt = self.config.refinement_prompt_template.format(
            cached_query=cached_query,
            cached_response=cached_response,
            new_query=new_query,
        )

        try:
            response = self._llm_client.generate(
                prompt=prompt,
                max_tokens=self.config.max_adaptation_tokens,
                temperature=0.3,
            )
            return response
        except Exception:
            return self._template_fill(cached_response, new_query)

    def _extract_key_phrases(self, text: str) -> list[str]:
        phrases = []
        quoted = re.findall(r'"([^"]+)"', text)
        phrases.extend(quoted)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrases.extend(entities)
        tech_terms = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+(?:_[a-z]+)*\b', text)
        phrases.extend(tech_terms)
        return phrases if phrases else [text.strip()[:50]]

    def set_llm_client(self, client: Any) -> None:
        self._llm_client = client
