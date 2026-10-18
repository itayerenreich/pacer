"""The knowledge library: the sources the AI may answer from, and the chunks they are split into.

A KnowledgeSource is one book, study, guideline or internal document. Its text is split into
Chunks: the unit that is searched, retrieved and cited. A chunk keeps where it came from (section
and pages), because a citation is built from that, never from what a model wrote.

Embeddings (a chunk as a vector) are not part of the domain: they belong to a model, and the same
chunk can have embeddings from several models while we compare them. See repositories/library.py.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .entity import Entity


class SourceKind(StrEnum):
    BOOK = "book"
    META_ANALYSIS = "meta_analysis"  # or a systematic review
    STUDY = "study"
    GUIDELINE = "guideline"  # a professional guideline (e.g. a ministry or an association)
    INTERNAL = "internal"  # the academy's own material
    CLASSIC = "classic"  # a classical Jewish source: Bible and commentaries, Talmud, Rambam...


class EvidenceLevel(StrEnum):
    """How strong a source is as EVIDENCE, from strongest to weakest. Shown with every answer, and
    used to prefer stronger sources when two chunks fit a question about equally well.

    TRADITION is outside that ladder on purpose (decision 2026-10-18): the Rambam is a source of
    values and educational wisdom, not an empirical finding. It is shown as such ("מקור מסורתי"),
    and on an empirical question a study comes first."""

    SYSTEMATIC = "systematic"  # meta-analyses and systematic reviews
    RESEARCH = "research"  # a single peer-reviewed study
    PROFESSIONAL = "professional"  # a professional book or guideline
    PRACTICE = "practice"  # experience: internal material, practitioner writing
    TRADITION = "tradition"  # classical sources: values and wisdom, not empirical evidence

    @property
    def rank(self) -> int:
        """4 for the strongest evidence, 1 for the weakest, 0 for a source that is not evidence."""
        return _RANK[self]

    @property
    def is_empirical(self) -> bool:
        return self is not EvidenceLevel.TRADITION

    @property
    def label(self) -> str:
        """How it is named to the staff, next to a citation."""
        return _LABEL[self]


_RANK = {
    EvidenceLevel.SYSTEMATIC: 4,
    EvidenceLevel.RESEARCH: 3,
    EvidenceLevel.PROFESSIONAL: 2,
    EvidenceLevel.PRACTICE: 1,
    EvidenceLevel.TRADITION: 0,
}
_LABEL = {
    EvidenceLevel.SYSTEMATIC: "סקירה שיטתית / מטא-אנליזה",
    EvidenceLevel.RESEARCH: "מחקר",
    EvidenceLevel.PROFESSIONAL: "ספרות מקצועית",
    EvidenceLevel.PRACTICE: "ניסיון מעשי",
    EvidenceLevel.TRADITION: "מקור ערכי-מסורתי",
}


class SourceLicense(StrEnum):
    OWNED_COPY = "owned_copy"  # a copy the academy owns: full text stays on this computer
    OPEN_ACCESS = "open_access"
    PUBLIC_DOMAIN = "public_domain"  # e.g. classical texts
    INTERNAL = "internal"  # the academy's own document


class KnowledgeSource(Entity):
    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(min_length=2, max_length=300)
    author: str | None = Field(default=None, max_length=200)
    year: int | None = Field(default=None, ge=1900, le=2100)
    kind: SourceKind
    evidence_level: EvidenceLevel
    language: str = Field(default="he", pattern=r"^[a-z]{2}$")
    license: SourceLicense
    # True: no part of this source may go to a model that does not run on this computer.
    local_only: bool = False
    # The file it was imported from (a name, not a path). Importing that file again replaces
    # every source that came from it.
    origin_file: str | None = Field(default=None, max_length=300)
    added_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def citation(self) -> str:
        """How the source is named in an answer: 'Title (Author, Year)'."""
        details = ", ".join(str(part) for part in (self.author, self.year) if part)
        return f"{self.title} ({details})" if details else self.title


CHUNK_MAX_LENGTH = 8_000  # characters; a chunk is a few hundred words, this only stops a broken split


class Chunk(Entity):
    """One searchable piece of a source.

    chunking names the strategy that produced it ("fixed-300-50", "structure-400"): while we
    compare strategies, chunks of several strategies of the same source live side by side, and
    a search always looks at one strategy.
    """

    model_config = ConfigDict(validate_assignment=True)

    source_id: int
    chunking: str = Field(min_length=1, max_length=50)
    ordinal: int = Field(ge=0)  # position in the source, within this chunking
    text: str = Field(min_length=1, max_length=CHUNK_MAX_LENGTH)
    section: str | None = Field(default=None, max_length=300)  # chapter or heading, if known
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _pages_in_order(self) -> Self:
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("page_end cannot be before page_start")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("Give both page_start and page_end, or neither")
        return self

    @property
    def pages(self) -> str | None:
        """'עמ׳ 45' or 'עמ׳ 45–46', for a citation."""
        if self.page_start is None:
            return None
        if self.page_start == self.page_end:
            return f"עמ׳ {self.page_start}"
        return f"עמ׳ {self.page_start}–{self.page_end}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())
