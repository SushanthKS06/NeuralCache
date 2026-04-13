
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from neural_cache.config import LLMConfig

class LLMClient(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict[str, Any]]:

class OpenAIClient(LLMClient):

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI

        self.config = config
        kwargs = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._client = AsyncOpenAI(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            timeout=self.config.timeout_seconds,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        metadata = {
            "model": self.config.model,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "finish_reason": choice.finish_reason,
            "provider": "openai",
        }

        return text, metadata

class AnthropicClient(LLMClient):

    def __init__(self, config: LLMConfig):
        import anthropic

        self.config = config
        kwargs = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key

        self._client = anthropic.AsyncAnthropic(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict[str, Any]]:
        message = await self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text if message.content else ""

        metadata = {
            "model": self.config.model,
            "input_tokens": message.usage.input_tokens if message.usage else 0,
            "output_tokens": message.usage.output_tokens if message.usage else 0,
            "provider": "anthropic",
        }

        return text, metadata

class LocalLLMClient(LLMClient):

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI

        self.config = config
        base_url = config.base_url or "http://localhost:8000/v1"
        self._client = AsyncOpenAI(base_url=base_url, api_key=config.api_key or "local")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            timeout=self.config.timeout_seconds,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        metadata = {
            "model": self.config.model,
            "provider": "local",
            "finish_reason": choice.finish_reason,
        }

        return text, metadata

def create_llm_client(config: LLMConfig) -> LLMClient:

    provider = config.provider.lower()

    if provider == "openai":
        return OpenAIClient(config)
    elif provider == "anthropic":
        return AnthropicClient(config)
    elif provider in ("local", "vllm", "ollama"):
        return LocalLLMClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
