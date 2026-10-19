"""Building clients for the two kinds of models Pacer uses.

Which model does which TASK is the router's job (next step). These functions only know
how to connect: Claude through the Anthropic API, open models through the local Ollama.
"""

from pacer.config import get_ai_settings

from .llm_client import LiteLLMClient, LLMConfigError

DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"  # fast and cheap; the router picks per task
DEFAULT_LOCAL_MODEL = "llama3.2"  # small enough for a 16 GB laptop without a GPU


# Models whose native JSON schema request fails through LiteLLM 1.104: for a model missing from its
# table, LiteLLM asks for JSON by FORCING a tool, and these models reject forced tools ("tool_choice:
# type tool is not supported for this model", the first evaluation run, 2026-10-19). For them the
# schema goes in the message instead, and Pydantic validates the answer (one correction attempt).
NO_NATIVE_SCHEMA = frozenset({"claude-sonnet-5-5"})


def claude_client(model: str = DEFAULT_CLAUDE_MODEL) -> LiteLLMClient:
    """A Claude model in Anthropic's cloud. Data sent here LEAVES this computer."""
    key = get_ai_settings().anthropic_api_key
    if key is None or not key.get_secret_value().strip():
        raise LLMConfigError("ANTHROPIC_API_KEY is missing. Add it to .env (never to Git).")
    native = False if model in NO_NATIVE_SCHEMA else None  # None: as LiteLLM's table says
    return LiteLLMClient(f"anthropic/{model}", runs_locally=False, api_key=key, native_schema=native)


# Ollama's default context window is 2,048 tokens, and it CUTS a longer message without an error. A
# library question with five passages, or a talk with its goals and the JSON schema, is longer.
LOCAL_CONTEXT_TOKENS = 8192


def ollama_client(
    model: str = DEFAULT_LOCAL_MODEL, *, temperature: float | None = 0.2, context_tokens: int = LOCAL_CONTEXT_TOKENS
) -> LiteLLMClient:
    """An open model served by Ollama on this computer. Data stays local.

    Open models accept a temperature; a low one makes their answers steadier.
    """
    return LiteLLMClient(
        f"ollama/{model}",
        runs_locally=True,
        api_base=get_ai_settings().ollama_base_url,
        temperature=temperature,
        timeout_s=600,  # a CPU-only laptop: minutes for a long answer, more on the first call (loading)
        retries=0,  # if Ollama is down, retrying won't help: fail fast with a clear message
        options={"num_ctx": context_tokens},
        # Ollama constrains the answer to the JSON schema ("format"), so even a small model cannot write
        # broken JSON. Without it, gemma3:4b broke its JSON on Hebrew abbreviations (רמב"ם: a quote
        # inside a string). LiteLLM's table does not know this about Ollama, so it is said here.
        native_schema=True,
    )
