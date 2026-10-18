"""Cleaning extracted text, with Hebrew in mind.

What is cleaned, and why:
* Invisible characters (zero-width, direction marks, soft hyphen): they split words in a way the
  eye cannot see, and a word split in two matches nothing.
* Unicode normalization (NFC): the same letter can be stored in two byte sequences.
* Lines broken in the middle of a paragraph (PDF): joined back. A word broken with a hyphen at the
  end of a line is joined without the hyphen.
* Page headers and footers (PDF): a short line that repeats on most pages (a book title, a running
  chapter name) is noise in every chunk; a bare page number too.

What is KEPT: niqqud and cantillation marks. They belong to the source and are shown in a citation
exactly as written. They are removed only from the text that goes to the embedding model
(ai/embeddings.for_embedding), where they would make "חֲנֹךְ" and "חנוך" look like different words.
"""

import re
import unicodedata
from collections import Counter

from .extract import Block

_INVISIBLE = re.compile("[­؜​-‏‪-‮⁠-⁩﻿]")
_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", _INVISIBLE.sub("", text))
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # a word hyphenated across a line break
    text = re.sub(r"\s*\n\s*", " ", text)  # a line break inside a paragraph is a space
    return re.sub(r"[ \t ]+", " ", text).strip()


def clean_blocks(blocks: list[Block]) -> tuple[list[Block], list[str]]:
    """The blocks cleaned, and what was removed (for the "before and after" report)."""
    removed: list[str] = []
    repeated = _repeated_lines(blocks)
    cleaned: list[Block] = []
    for block in blocks:
        lines = []
        for line in block.text.split("\n"):
            if line.strip() in repeated or _PAGE_NUMBER.match(line):
                removed.append(line.strip())
                continue
            lines.append(line)
        text = clean_text("\n".join(lines))
        if text:
            cleaned.append(Block(text=text, page=block.page, heading=block.heading))
    return cleaned, removed


def _repeated_lines(blocks: list[Block]) -> set[str]:
    """Short lines that appear on at least half of the pages (at least 3): headers and footers."""
    pages = {b.page for b in blocks if b.page is not None}
    if len(pages) < 3:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        lines = {line.strip() for b in blocks if b.page == page for line in b.text.split("\n")}
        seen.update(line for line in lines if 0 < len(line) <= 80)
    return {line for line, count in seen.items() if count >= max(3, len(pages) // 2)}
