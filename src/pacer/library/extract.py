"""From a file to blocks: paragraphs of text, in order, with the page each one is on.

A block is the smallest unit we keep from the original layout: a paragraph in Word or in a text
file, a paragraph-ish run of lines in a PDF. Pages exist only in PDF; Word has no fixed pages (they
depend on the screen and the font), so a Word source is cited by its sections instead.

Supported: .docx (python-docx), .pdf with a text layer (pypdf), .txt and .md. A scanned PDF has no
text layer; it is refused with a clear message instead of producing an empty library.
"""

import re
from dataclasses import dataclass
from pathlib import Path


class ExtractionError(Exception):
    """The file cannot be read as text."""


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None = None  # 1-based, only for formats with pages
    heading: bool = False  # styled as a heading in the original (Word "Heading N")


def extract(path: Path) -> list[Block]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        blocks = _docx(path)
    elif suffix == ".pdf":
        blocks = _pdf(path)
    elif suffix in (".txt", ".md"):
        blocks = _plain(path.read_text(encoding="utf-8"))
    else:
        raise ExtractionError(f"Unsupported file type {suffix!r}: use .docx, .pdf, .txt or .md")
    blocks = [b for b in blocks if b.text.strip()]
    if not blocks:
        raise ExtractionError(f"No text found in {path.name} (a scanned PDF needs OCR first)")
    return blocks


def _docx(path: Path) -> list[Block]:
    import docx  # imported here: only this format needs it

    document = docx.Document(str(path))
    return [
        Block(text=p.text, heading=(p.style is not None and (p.style.name or "").lower().startswith("heading")))
        for p in document.paragraphs
    ]


def _pdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ExtractionError(f"{path.name} is encrypted")
    blocks: list[Block] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # In a PDF a paragraph is a run of lines; an empty line separates paragraphs.
        blocks += [Block(text=chunk, page=number) for chunk in re.split(r"\n\s*\n", text)]
    return blocks


def _plain(text: str) -> list[Block]:
    return [Block(text=paragraph) for paragraph in re.split(r"\n\s*\n", text)]
