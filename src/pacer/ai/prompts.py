"""System prompts as versioned files.

A system prompt is the most important piece of logic in an AI feature, so it is
treated like code: one file per version, reviewed in pull requests, measured in
the evaluation (stage 4). Every model call in Pacer goes out with one of these.

Layout (inside the package, so it ships with the code):

    src/pacer/prompts/<name>/v1.md
    src/pacer/prompts/<name>/v2.md   <- a change in behavior is a NEW version, never an edit

Each file starts with a small header:

    ---
    name: connection_check
    version: 1
    description: One line: what this prompt is for
    ---
    You are ...   (the system prompt itself)
"""

import re
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_VERSION_FILE = re.compile(r"^v(\d+)\.md$")


class Prompt(BaseModel):
    """One version of one system prompt. Frozen: a prompt never changes after loading."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1)
    description: str = Field(min_length=5)
    system: str = Field(min_length=20, repr=False)

    @property
    def ref(self) -> str:
        """How the prompt is named in logs and results, e.g. "connection_check@v1"."""
        return f"{self.name}@v{self.version}"


class PromptError(Exception):
    """A prompt file is missing or malformed."""


def available_versions(name: str, directory: Path = PROMPTS_DIR) -> list[int]:
    folder = directory / name
    if not folder.is_dir():
        return []
    return sorted(int(m.group(1)) for f in folder.iterdir() if (m := _VERSION_FILE.match(f.name)))


@cache
def load_prompt(name: str, version: int | None = None, directory: Path = PROMPTS_DIR) -> Prompt:
    """Load a prompt; without `version`, the newest one. Cached: files are read once per process."""
    versions = available_versions(name, directory)
    if not versions:
        raise PromptError(f"No prompt named {name!r} in {directory}")
    version = versions[-1] if version is None else version
    if version not in versions:
        raise PromptError(f"Prompt {name!r} has no version {version} (has {versions})")

    path = directory / name / f"v{version}.md"
    header, system = _split_header(path)
    if header.get("name") != name or header.get("version") != str(version):
        raise PromptError(
            f"{path}: header says {header.get('name')}@v{header.get('version')}, file is {name}@v{version}"
        )
    return Prompt(name=name, version=version, description=header.get("description", ""), system=system)


def _split_header(path: Path) -> tuple[dict[str, str], str]:
    """'---' / key: value lines / '---' / body. Deliberately tiny: no YAML dependency."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise PromptError(f"{path}: must start with a '---' header block")
    header = {}
    for line in parts[1].strip().splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            raise PromptError(f"{path}: header line without ':': {line!r}")
        header[key.strip()] = value.strip()
    return header, parts[2].strip()
