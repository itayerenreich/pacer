"""Building clients for the two kinds of models Pacer uses.

Which model does which TASK is the router's job (next step). These functions only know
how to connect: Claude through the Anthropic API, open models through the local Ollama.
"""

from pacer.config import get_ai_settings

from .llm_client import LiteLLMClient, LLMConfigError

DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"  # fast and cheap; the router picks per task
DEFAULT_LOCAL_MODEL = "llama3.2"  # small enough for a 16 GB laptop without a GPU


def claude_client(model: str = DEFAULT_CLAUDE_MODEL) -> LiteLLMClient:
    """A Claude model in Anthropic's cloud. Data sent here LEAVES this computer."""
    key = get_ai_settings().anthropic_api_key
    if key is None or not key.get_secret_value().strip():
        raise LLMConfigError("ANTHROPIC_API_KEY is missing. Add it to .env (never to Git).")
    return LiteLLMClient(f"anthropic/{model}", runs_locally=False, api_key=key)


def ollama_client(model: str = DEFAULT_LOCAL_MODEL, *, temperature: float | None = 0.2) -> LiteLLMClient:
    """An open model served by Ollama on this computer. Data stays local.

    Open models accept a temperature; a low one makes their answers steadier.
    """
    return LiteLLMClient(
        f"ollama/{model}",
        runs_locally=True,
        api_base=get_ai_settings().ollama_base_url,
        temperature=temperature,
        timeout_s=180,  # a CPU-only laptop is slow, especially on the first call (model loading)
        retries=0,  # if Ollama is down, retrying won't help: fail fast with a clear message
    )
