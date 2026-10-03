from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and supported aliases."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {"anthorpic": "anthropic", "google": "gemini", "google_genai": "gemini", "open_router": "openrouter"}
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported provider: {value!r}. Expected one of {sorted(supported)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate a chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}
    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI
        kwargs = dict(common)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("custom provider requires base_url")
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(common)
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(common)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(common)
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)
    from langchain_openrouter import ChatOpenRouter
    kwargs = dict(common)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ChatOpenRouter(**kwargs)
