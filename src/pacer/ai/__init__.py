"""AI layer: language model access, prompts, and (next) routing and pipelines.

May import from: domain, services (pipelines build on them). Must not be imported by domain.

    prompts.py     versioned system prompts, loaded from src/pacer/prompts/<name>/vN.md
    llm_client.py  the LLMClient interface, LiteLLMClient, LLMResponse, LLM errors
    pricing.py     our explicit price table (cost per call; unknown is None, never 0)
    clients.py     building a Claude client (cloud) or an Ollama client (this computer)
    check.py       python -m pacer.ai.check: a harmless connection check
"""

from .llm_client import (
    LiteLLMClient,
    LLMAuthError,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMUnavailableError,
)
from .prompts import Prompt, PromptError, load_prompt

__all__ = [
    "LLMAuthError",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMUnavailableError",
    "LiteLLMClient",
    "Prompt",
    "PromptError",
    "load_prompt",
]
