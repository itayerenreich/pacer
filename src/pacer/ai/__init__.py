"""AI layer: language model access, prompts, and (next) routing and pipelines.

May import from: domain, services, and repository interfaces (Protocols). Must not be imported by domain.

    prompts.py     versioned system prompts, loaded from src/pacer/prompts/<name>/vN.md
    llm_client.py  the LLMClient interface, LiteLLMClient, LLMResponse, LLM errors
    pricing.py     our explicit price table (cost per call; unknown is None, never 0)
    clients.py     building a Claude client (cloud) or an Ollama client (this computer)
    routing.py     which model serves which AI task, and the privacy rule (PERSONAL data stays local)
    structured.py  complete_structured(): JSON that matches a Pydantic model, validated, one correction
    checks.py      the small schema and text used by the connection checks
    anonymization.py  names -> tokens before any cloud model, and back; AnonymizedText
    transcription.py  local speech-to-text (Whisper) and the folder where recordings wait, then go
    meeting_pipeline.py  staff meeting: transcribe -> anonymize -> extract -> validate -> proposals
    talk_pipeline.py     personal talk: anonymize -> summarize -> validate -> progress and note proposals
    issues.py            what content validation corrected, reported with the proposals
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
