from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a complete LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    from model_provider import normalize_provider
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    api_keys = {
        "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY", "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    }
    base_urls = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL"}
    model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=os.getenv(api_keys.get(provider, "")) or None,
        base_url=os.getenv(base_urls.get(provider, "")) or None,
    )
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model.model_name),
        temperature=0.0,
        api_key=os.getenv(api_keys.get(judge_provider, "")) or model.api_key,
        base_url=os.getenv(base_urls.get(judge_provider, "")) or model.base_url,
    )
    # Example knobs:
    # - LLM_PROVIDER / LLM_MODEL
    # - OPENAI_API_KEY
    # - GEMINI_API_KEY
    # - ANTHROPIC_API_KEY
    # - OLLAMA_BASE_URL
    # - OPENROUTER_API_KEY
    # - CUSTOM_BASE_URL / CUSTOM_API_KEY
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "550")),
        compact_keep_messages=max(1, int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))),
        model=model,
        judge_model=judge_model,
    )
