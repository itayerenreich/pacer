"""Which parts of a file are which sources: a small JSON file next to the library's text.

One file can hold several sources (your compilation holds five classical works), and a citation
must name the WORK ("רמב"ם, הלכות תלמוד תורה פרק ב׳"), not the file (decision 2026-10-18). The
manifest says how to find the sections in the file and gives each one its metadata. It holds no
text of the sources, so it can live in Git (library/*.json) while the text stays in data/library/.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import EvidenceLevel, SourceKind, SourceLicense


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: int = Field(ge=1, description="Which section of the file, 1-based, in the order it appears")
    title: str = Field(min_length=2, max_length=300)
    author: str | None = None
    year: int | None = None
    kind: SourceKind
    evidence_level: EvidenceLevel
    license: SourceLicense
    language: str = "he"
    local_only: bool = False
    # How many paragraphs at the start of the section are the curator's own summary, not the
    # source. They become a chunk of their own: a modern-Hebrew bridge to an old text.
    summary_paragraphs: int = Field(default=0, ge=0)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="The file name in data/library/")
    # How sections are found: "numbered" lines ("1. ..."), Word "headings", or the "whole" file is one.
    split: Literal["numbered", "headings", "whole"] = "whole"
    sources: list[SourceEntry] = Field(min_length=1)


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
