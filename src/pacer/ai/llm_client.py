"""One door to every language model: the LLMClient interface, and its LiteLLM implementation.

Why an interface of our own (Adapter pattern, like the repositories):
* The rest of Pacer calls `client.complete(prompt, message)` and gets an LLMResponse.
  It never imports litellm, anthropic or ollama, so swapping a model is configuration.
* Tests pass a FakeLLMClient with fixed answers: fast, free, deterministic, no network.

Rules this module enforces for EVERY call:
* A system prompt is mandatory: `complete()` takes a Prompt object, loaded from a versioned file.
* The call is logged with the prompt version, model, tokens, cost and time, but never with
  the message or answer text: they may contain personal data.
* Provider errors become our own small set of errors (LLMError and subclasses), so callers
  don't depend on litellm's exception classes.
"""

import logging
import os
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, SecretStr

from .pricing import cost_usd
from .prompts import Prompt

# Use the price table bundled with the pinned litellm version. By default litellm downloads a
# fresh table on import: a network call we did not choose, and prices changing under our feet.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402  (must come after the setting above)
from litellm import exceptions as litellm_errors  # noqa: E402

litellm.suppress_debug_info = True  # no promotional/debug banners in our logs
for _name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # its INFO lines duplicate ours; errors still show

log = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """What a model answered, and what it cost. Frozen: a result is a record, not a draft."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    prompt_ref: str  # e.g. "connection_check@v1": which exact prompt produced this
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the price is unknown; 0.0 for local models
    latency_ms: int
    truncated: bool  # True when the answer hit max_tokens and was cut off


class LLMClient(Protocol):
    """Anything that can answer a prompt. LiteLLMClient in production, a fake in tests."""

    model: str
    runs_locally: bool  # True: the data never leaves this computer (the router relies on this)
    native_schema: bool  # True: the provider itself constrains the answer to a JSON Schema

    def complete(
        self,
        prompt: Prompt,
        user_message: str,
        *,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """response_schema: a JSON Schema the answer should follow (see structured.py).
        A hint to the provider; the caller still validates the answer."""
        ...


# --- errors -------------------------------------------------------------------------------


class LLMError(Exception):
    """A model call failed. Subclasses say why, so callers can react differently."""


class LLMConfigError(LLMError):
    """Pacer is not set up for this call (e.g. no API key in .env)."""


class LLMAuthError(LLMError):
    """The provider rejected the key (wrong, revoked or expired)."""


class LLMUnavailableError(LLMError):
    """The model could not be reached: Ollama not running, network down, provider overloaded."""


class LLMRateLimitError(LLMError):
    """Too many requests, or the spend limit was reached."""


_ERROR_MAP: list[tuple[type[Exception], type[LLMError]]] = [
    (litellm_errors.AuthenticationError, LLMAuthError),
    (litellm_errors.PermissionDeniedError, LLMAuthError),
    (litellm_errors.RateLimitError, LLMRateLimitError),
    (litellm_errors.BudgetExceededError, LLMRateLimitError),
    (litellm_errors.Timeout, LLMUnavailableError),
    (litellm_errors.APIConnectionError, LLMUnavailableError),
    (litellm_errors.ServiceUnavailableError, LLMUnavailableError),
    (litellm_errors.InternalServerError, LLMUnavailableError),
]


def translate_error(error: Exception) -> LLMError:
    """A provider exception as one of OUR errors (LLMError and subclasses)."""
    for source, target in _ERROR_MAP:
        if isinstance(error, source):
            return target(str(error))
    return LLMError(str(error))


# --- the LiteLLM implementation -------------------------------------------------------------


class LiteLLMClient:
    """A client for one model, through LiteLLM.

    model:        LiteLLM's name, e.g. "anthropic/claude-haiku-5-5" or "ollama/llama3.2".
    runs_locally: must be True only for models served on this machine (Ollama).
    temperature:  only for models that accept it. Claude 5.5 models reject it, so the default
                  is None: precision comes from the prompt, structured output and validation.
    """

    def __init__(
        self,
        model: str,
        *,
        runs_locally: bool,
        api_key: SecretStr | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        timeout_s: float = 60,
        retries: int = 2,
        native_schema: bool | None = None,
    ) -> None:
        self.model = model
        self.runs_locally = runs_locally
        # Known once, from litellm's bundled model table (no network): Claude yes, llama3.2 no.
        # native_schema=False overrides it: the schema then goes in the message, and Pydantic validates.
        self.native_schema = (
            bool(litellm.supports_response_schema(model=model)) if native_schema is None else native_schema
        )
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._retries = retries

    def complete(
        self,
        prompt: Prompt,
        user_message: str,
        *,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not user_message.strip():
            raise ValueError("user_message is empty")

        started = time.perf_counter()
        try:
            raw = litellm.completion(**self._request(prompt, user_message, max_tokens, response_schema))
        except Exception as error:  # every provider failure leaves as one of OUR errors
            translated = translate_error(error)
            log.warning(
                "LLM call failed: prompt=%s model=%s error=%s",
                prompt.ref,
                self.model,
                type(translated).__name__,
            )
            raise translated from error

        response = self._to_response(raw, prompt, started)
        log.info(
            "LLM call: prompt=%s model=%s tokens=%s->%s cost=%s latency=%sms%s",
            response.prompt_ref,
            response.model,
            response.input_tokens,
            response.output_tokens,
            "unknown" if response.cost_usd is None else f"${response.cost_usd:.6f}",
            response.latency_ms,
            " TRUNCATED" if response.truncated else "",
        )
        return response

    def _request(
        self, prompt: Prompt, user_message: str, max_tokens: int, response_schema: dict[str, Any] | None
    ) -> dict:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "timeout": self._timeout_s,
            "num_retries": self._retries,  # retries transient failures (rate limit, overload)
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if response_schema is not None and self.native_schema:
            # The provider constrains the answer to this shape. For Anthropic, litellm also removes
            # the keywords Claude rejects (minLength, maximum...) and keeps them as descriptions.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True},
            }
        if self._api_key is not None:
            request["api_key"] = self._api_key.get_secret_value()  # passed per call, never put in os.environ
        if self._api_base is not None:
            request["api_base"] = self._api_base
        return request

    def _to_response(self, raw, prompt: Prompt, started: float) -> LLMResponse:
        choice = raw.choices[0]
        input_tokens, output_tokens = raw.usage.prompt_tokens, raw.usage.completion_tokens
        # Our own price table (pricing.py): unknown price -> None, never a silent 0.
        cost = 0.0 if self.runs_locally else cost_usd(self.model, input_tokens, output_tokens)
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            prompt_ref=prompt.ref,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round((time.perf_counter() - started) * 1000),
            truncated=choice.finish_reason == "length",
        )
