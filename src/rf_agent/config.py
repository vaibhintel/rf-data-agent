"""Central configuration. Everything tunable lives here, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderConfig:
    """One OpenAI-compatible LLM endpoint (Ollama, vLLM, or a hosted API)."""
    name: str
    base_url: str
    model: str
    api_key: str = "not-needed"
    timeout_s: float = 60.0


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8                 # hard cap on reasoning/tool steps per question
    max_tool_result_chars: int = 4000  # tool output is truncated before it reaches the model
    temperature: float = 0.0           # deterministic decoding for reproducible evals


@dataclass(frozen=True)
class Settings:
    providers: list[ProviderConfig] = field(default_factory=list)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_seed: int = 7


def load_settings() -> Settings:
    """Primary provider is a local Ollama model; an optional fallback is added if configured."""
    providers = [
        ProviderConfig(
            name="ollama-local",
            base_url=os.getenv("RF_AGENT_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("RF_AGENT_MODEL", "qwen2.5:7b"),
            api_key=os.getenv("RF_AGENT_API_KEY", "ollama"),
        )
    ]
    if os.getenv("RF_AGENT_FALLBACK_BASE_URL"):
        providers.append(
            ProviderConfig(
                name="fallback",
                base_url=os.environ["RF_AGENT_FALLBACK_BASE_URL"],
                model=os.getenv("RF_AGENT_FALLBACK_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("RF_AGENT_FALLBACK_API_KEY", ""),
            )
        )
    return Settings(
        providers=providers,
        agent=AgentConfig(max_steps=int(os.getenv("RF_AGENT_MAX_STEPS", "8"))),
        data_seed=int(os.getenv("RF_AGENT_SEED", "7")),
    )
