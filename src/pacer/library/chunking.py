"""Splitting a source into chunks: the unit that is searched, retrieved and cited.

Why split at all: a search compares the question with each chunk as ONE vector. A whole chapter is
an average of many ideas and matches none of them well; a single sentence has lost its context
("start with a short personal talk": with whom?). A chunk is the middle: one idea, with enough
context to stand on its own.

Two strategies, measured against each other (task 8), both by WORDS rather than characters (a
Hebrew word costs a model more tokens than an English one, and a character count hides that):

* fixed-N-M     windows of N words, each repeating the last M words of the one before (overlap), so
                an idea cut at a border is still whole in one of the two chunks. Ignores the layout.
* structure-N   paragraphs kept whole and grouped up to N words; a paragraph longer than N is split
                at sentence ends. A curator's summary is a chunk of its own. Never mixes two sources.

Each chunk keeps WHERE it came from: the labels of its paragraphs ("הלכה ב", "פירוש המלבי"ם") and
its pages. A citation is built from that, never from what a model wrote.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

SUMMARY = "תקציר"  # the locator of a curator's summary paragraph
_LABEL = re.compile(r"^(?P<label>[^:\n]{2,40}?):\s")  # "הלכה ב:", "פסוק:", "פירוש המלבי"ם (מלא):" -> a label
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


@dataclass(frozen=True)
class Unit:
    """A paragraph of one source, with where it is."""

    text: str
    locator: str | None = None  # "הלכה ב", "תקציר"...
    page: int | None = None


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    locators: tuple[str, ...]
    page_start: int | None
    page_end: int | None

    @property
    def section(self) -> str | None:
        return ", ".join(self.locators)[:300] or None

    @property
    def words(self) -> int:
        return len(self.text.split())


def label_of(paragraph: str) -> str | None:
    """A short label before a colon at the start of a paragraph: "הלכה ב: ..." -> "הלכה ב"."""
    match = _LABEL.match(paragraph.lstrip(" \"'״׳“”"))  # a quoted paragraph: "הלכה א: ..."
    if match is None:
        return None
    label = match["label"].strip()
    return label if len(label.split()) <= 6 else None


def _pages(units: list[Unit]) -> tuple[int | None, int | None]:
    pages = [u.page for u in units if u.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def _draft(text: str, units: list[Unit]) -> ChunkDraft:
    locators = tuple(dict.fromkeys(u.locator for u in units if u.locator))  # unique, in order
    return ChunkDraft(" ".join(text.split()), locators, *_pages(units))


def fixed(units: list[Unit], size: int, overlap: int) -> list[ChunkDraft]:
    """Windows of `size` words, each starting `size - overlap` words after the one before."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    words: list[tuple[str, Unit]] = [(w, u) for u in units for w in u.text.split()]
    drafts: list[ChunkDraft] = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        drafts.append(_draft(" ".join(w for w, _ in window), [u for _, u in window]))
        if start + size >= len(words):
            break
        start += size - overlap
    return drafts


def structure(units: list[Unit], max_words: int) -> list[ChunkDraft]:
    """Whole paragraphs, grouped up to max_words; a longer paragraph is split at sentence ends."""
    pieces: list[Unit] = []
    for unit in units:
        pieces += _split_long(unit, max_words)

    drafts: list[ChunkDraft] = []
    group: list[Unit] = []

    def close() -> None:
        if group:
            drafts.append(_draft(" ".join(u.text for u in group), list(group)))
            group.clear()

    for piece in pieces:
        if piece.locator == SUMMARY:  # a summary stands alone
            close()
            drafts.append(_draft(piece.text, [piece]))
            continue
        if group and sum(len(u.text.split()) for u in group) + len(piece.text.split()) > max_words:
            close()
        group.append(piece)
    close()
    return drafts


def _split_long(unit: Unit, max_words: int) -> list[Unit]:
    if len(unit.text.split()) <= max_words:
        return [unit]
    parts: list[Unit] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(unit.text):
        if current and len(" ".join([*current, sentence]).split()) > max_words:
            parts.append(Unit(" ".join(current), unit.locator, unit.page))
            current = []
        current.append(sentence)  # a single sentence longer than max_words stays whole
    if current:
        parts.append(Unit(" ".join(current), unit.locator, unit.page))
    return parts


Strategy = Callable[[list[Unit]], list[ChunkDraft]]

# The strategies by name. The name is stored with every chunk; a different setting is a new name.
STRATEGIES: dict[str, Strategy] = {
    "fixed-200-40": lambda units: fixed(units, size=200, overlap=40),
    "structure-220": lambda units: structure(units, max_words=220),
}
DEFAULT_CHUNKING = "fixed-200-40"  # chosen by measurement, 2026-10-18 (stage 4 doc, part 11); revisit with more texts
