"""
LLM abstraction — build the configured LLM instance.
Supports OpenAI, Anthropic, and Ollama (local).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Build and return the configured LLM.
    Result is cached — the same instance is reused across the application.
    """
    from src.config import settings

    _provider = provider or settings.LLM_PROVIDER
    _model = model or settings.LLM_MODEL
    _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if _provider == "openai":
        return _build_openai(_model, _temp, settings)
    if _provider == "anthropic":
        return _build_anthropic(_model, _temp, settings)
    if _provider == "ollama":
        return _build_ollama(_model, _temp, settings)

    raise ValueError(
        f"Unknown LLM provider: '{_provider}'. Choose 'openai', 'anthropic', or 'ollama'."
    )


def _build_openai(model: str, temperature: float, settings: Any) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    logger.info("LLM: OpenAI / %s (temp=%.1f)", model, temperature)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=settings.LLM_MAX_TOKENS,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,  # Enables streaming out of the box
    )


def _build_anthropic(model: str, temperature: float, settings: Any) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    logger.info("LLM: Anthropic / %s (temp=%.1f)", model, temperature)
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=settings.LLM_MAX_TOKENS,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
    )


def _build_ollama(model: str, temperature: float, settings: Any) -> BaseChatModel:
    from langchain_community.chat_models import ChatOllama

    logger.info("LLM: Ollama / %s @ %s (temp=%.1f)", model, settings.OLLAMA_BASE_URL, temperature)
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.OLLAMA_BASE_URL,
    )
