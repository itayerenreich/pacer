"""From a file and its manifest to sources and chunks in the database.

    blocks = extract(file)                    paragraphs, with pages if the format has them
    blocks, removed = clean_blocks(blocks)    invisible characters, broken lines, page headers
    sections = split_sections(blocks, ...)    which paragraphs belong to which source
    units = units_of(section, entry)          paragraphs with their labels ("הלכה ב", "תקציר")
    drafts = STRATEGIES[name](units)          chunks, per strategy

Importing the same file again REPLACES every source that came from it (and, by ON DELETE CASCADE,
their chunks and embeddings): the file is the truth, the database follows it. One transaction:
a file that fails halfway leaves the library as it was.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from pacer.domain import Chunk, KnowledgeSource
from pacer.repositories import UnitOfWork

from .chunking import STRATEGIES, SUMMARY, ChunkDraft, Unit, label_of
from .clean import clean_blocks
from .extract import Block, extract
from .manifest import Manifest, SourceEntry

_NUMBERED = re.compile(r"^\d{1,3}\.\s")


class ManifestError(Exception):
    """The manifest does not fit the file."""


@dataclass(frozen=True)
class Section:
    heading: str | None
    blocks: list[Block]


def split_sections(blocks: list[Block], how: str) -> list[Section]:
    """Sections in the order they appear. Text before the first heading is dropped (a title page)."""
    if how == "whole":
        return [Section(None, blocks)]
    sections: list[Section] = []
    for block in blocks:
        is_heading = bool(_NUMBERED.match(block.text)) if how == "numbered" else block.heading
        if is_heading:
            sections.append(Section(block.text, []))
        elif sections:
            sections[-1].blocks.append(block)
    return sections


def units_of(section: Section, entry: SourceEntry) -> list[Unit]:
    units = []
    for index, block in enumerate(section.blocks):
        locator = SUMMARY if index < entry.summary_paragraphs else label_of(block.text)
        units.append(Unit(block.text, locator, block.page))
    return units


@dataclass
class SourcePlan:
    entry: SourceEntry
    heading: str | None
    words: int
    chunks: dict[str, list[ChunkDraft]] = field(default_factory=dict)


@dataclass
class FilePlan:
    file: str
    removed: list[str]
    sources: list[SourcePlan]


def plan(path: Path, manifest: Manifest, strategies: list[str] | None = None) -> FilePlan:
    """Everything that would be imported, without a database: for a preview, and for import."""
    if path.name != manifest.file:
        raise ManifestError(f"The manifest is for {manifest.file}, not {path.name}")
    names = strategies or list(STRATEGIES)
    unknown = [name for name in names if name not in STRATEGIES]
    if unknown:
        raise ManifestError(f"Unknown chunking strategies: {', '.join(unknown)}")
    blocks, removed = clean_blocks(extract(path))
    sections = split_sections(blocks, manifest.split)
    plans = []
    for entry in manifest.sources:
        if entry.section > len(sections):
            raise ManifestError(f"The manifest names section {entry.section}, the file has {len(sections)}")
        section = sections[entry.section - 1]
        units = units_of(section, entry)
        source_plan = SourcePlan(entry, section.heading, sum(len(u.text.split()) for u in units))
        for name in names:
            source_plan.chunks[name] = STRATEGIES[name](units)
        plans.append(source_plan)
    return FilePlan(path.name, removed, plans)


def import_file(uow: UnitOfWork, path: Path, manifest: Manifest, strategies: list[str] | None = None) -> FilePlan:
    file_plan = plan(path, manifest, strategies)
    with uow() as repos:
        for old in repos.sources.list_all():
            if old.origin_file == file_plan.file:
                repos.sources.delete(old.key)
        for source_plan in file_plan.sources:
            entry = source_plan.entry
            source = repos.sources.add(
                KnowledgeSource(
                    title=entry.title,
                    author=entry.author,
                    year=entry.year,
                    kind=entry.kind,
                    evidence_level=entry.evidence_level,
                    language=entry.language,
                    license=entry.license,
                    local_only=entry.local_only,
                    origin_file=file_plan.file,
                )
            )
            for name, drafts in source_plan.chunks.items():
                repos.chunks.add_many(
                    [
                        Chunk(
                            source_id=source.key,
                            chunking=name,
                            ordinal=i,
                            text=d.text,
                            section=d.section,
                            page_start=d.page_start,
                            page_end=d.page_end,
                        )
                        for i, d in enumerate(drafts)
                    ]
                )
    return file_plan


def preview(file_plan: FilePlan, *, samples: int = 3) -> str:
    """A Markdown report of what an import would store: per source and strategy, how many chunks,
    their sizes, and a few of them in full, so a person can judge whether the split makes sense."""
    lines = [f"# תצוגה מקדימה: {file_plan.file}", ""]
    if file_plan.removed:
        lines += [f"**הוסר בניקוי** ({len(file_plan.removed)} שורות, למשל כותרות עמוד): "
                  + ", ".join(f"`{r}`" for r in sorted(set(file_plan.removed))[:8]), ""]
    for source_plan in file_plan.sources:
        entry = source_plan.entry
        lines += [
            f"## {entry.section}. {entry.title}",
            "",
            f"{entry.author or 'ללא מחבר'} · {entry.evidence_level.label} · {source_plan.words} מילים",
            "",
        ]
        for name, drafts in source_plan.chunks.items():
            sizes = [d.words for d in drafts]
            lines += [f"### {name}: {len(drafts)} קטעים, גדלים {sizes}", ""]
            for i, d in enumerate(drafts[:samples]):
                where = d.section or "ללא תווית"
                lines += [f"**קטע {i}** ({d.words} מילים · {where})", "", f"> {d.text}", ""]
    return "\n".join(lines)
