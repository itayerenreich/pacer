"""Connection check: send one harmless Hebrew question to Claude and to the local Ollama model.

    uv run python -m pacer.ai.check              # both
    uv run python -m pacer.ai.check --only claude
    uv run python -m pacer.ai.check --only local
    uv run python -m pacer.ai.check --only structured   # JSON through the router (Claude)

Costs a fraction of a cent. Sends no personal data: the question is fixed in this file.
"""

import argparse

from pacer.logging_config import configure_logging

from .checks import SAMPLE_TEXT, TextDigest
from .clients import DEFAULT_CLAUDE_MODEL, DEFAULT_LOCAL_MODEL, claude_client, ollama_client
from .llm_client import LLMClient, LLMError, LLMUnavailableError
from .prompts import load_prompt
from .routing import AITask, get_router
from .structured import StructuredOutputError, complete_structured

QUESTION = "שלום! במשפט אחד: מה זה מכינה קדם-צבאית?"


def check(name: str, make_client) -> bool:
    print(f"\n=== {name} ===")
    try:
        client: LLMClient = make_client()
        response = client.complete(load_prompt("connection_check"), QUESTION, max_tokens=200)
    except LLMUnavailableError:
        print(
            "Not reachable. For the local model: is Ollama running? (open the Ollama app, or `ollama serve`)"
        )
        return False
    except LLMError as error:
        print(f"Failed: {type(error).__name__}: {error}")
        return False

    cost = "unknown" if response.cost_usd is None else f"${response.cost_usd:.6f}"
    print(f"Answer:  {response.text}")
    print(f"Model:   {response.model}   prompt: {response.prompt_ref}")
    print(
        f"Tokens:  {response.input_tokens} in, {response.output_tokens} out   cost: {cost}   time: {response.latency_ms} ms"
    )
    if response.truncated:
        print("Warning: the answer was cut off at max_tokens")
    return True


def check_structured() -> bool:
    print("\n=== Structured output (router -> structured_check) ===")
    router = get_router()
    try:
        result = complete_structured(
            router.client_for(AITask.STRUCTURED_CHECK),
            load_prompt("structured_check"),
            SAMPLE_TEXT,
            TextDigest,
            max_tokens=router.route(AITask.STRUCTURED_CHECK).max_tokens,
        )
    except StructuredOutputError as error:
        print(f"Failed after {error.attempts} attempt(s): {error.reasons}")
        return False
    except LLMError as error:
        print(f"Failed: {type(error).__name__}: {error}")
        return False

    cost = "unknown" if result.cost_usd is None else f"${result.cost_usd:.6f}"
    print(f"Object:   {result.value!r}")
    print(f"Attempts: {result.attempts}   prompt: {result.prompt_ref}   cost: {cost}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.ai.check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", choices=["claude", "local", "structured"])
    args = parser.parse_args()
    configure_logging()

    results = []
    if args.only in (None, "claude"):
        results.append(check(f"Claude ({DEFAULT_CLAUDE_MODEL}, cloud)", claude_client))
    if args.only in (None, "local"):
        results.append(check(f"Ollama ({DEFAULT_LOCAL_MODEL}, this computer)", ollama_client))
    if args.only in (None, "structured"):
        results.append(check_structured())
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
