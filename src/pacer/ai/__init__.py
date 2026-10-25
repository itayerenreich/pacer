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
    meeting_pipeline.py  staff meeting: text -> anonymize -> extract -> validate -> proposals
    talk_pipeline.py     personal talk: anonymize -> summarize -> validate -> progress and note proposals
    issues.py            what content validation corrected, reported with the proposals
    sensitive.py         health and diagnoses: removed before sending, never kept in memory
    safety.py            risk situations: found in code before any model, answered with a referral
    risk_screen.py       the safety gate: the phrase list, then a model as a second net that only adds
    coach.py             the counselor coach: gate -> trainee context -> library -> Claude -> validation
    embeddings.py        text -> vector, with open models in Ollama (bge-m3, qwen3-embedding)
    library.py           index the library's chunks, and search them by meaning
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
