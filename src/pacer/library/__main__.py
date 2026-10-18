"""Command line: the knowledge library, on the database configured in .env.

uv run python -m pacer.library preview data/library/classics.docx --manifest library/classics.json --out data/preview.md
uv run python -m pacer.library import  data/library/classics.docx --manifest library/classics.json
uv run python -m pacer.library index                      # every embedding model, every chunking strategy
uv run python -m pacer.library search "השאלה שלך" --model bge-m3
uv run python -m pacer.library compare library/questions-classics.json --out data/library-report.md

preview needs no database and no model: it shows how the file would be split. index and search need
Ollama running with the models pulled (ollama pull bge-m3, ollama pull qwen3-embedding:0.6b).
Reports are UTF-8 Markdown: open them in VS Code (Ctrl+Shift+V). data/ is outside Git.
"""

import argparse
import json
from pathlib import Path

from pacer.ai.embeddings import EMBEDDING_MODELS, ollama_embedder
from pacer.ai.library import index_library, search_library
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import ScoredChunk, get_engine, repositories

from .chunking import DEFAULT_CHUNKING, STRATEGIES
from .ingest import import_file, plan, preview
from .manifest import load_manifest


def _uow():
    engine = get_engine()
    return lambda: repositories(engine)


def _models(choice: str) -> list[str]:
    return list(EMBEDDING_MODELS) if choice == "all" else [choice]


def _chunkings(choice: str) -> list[str]:
    return list(STRATEGIES) if choice == "all" else [choice]


def _hit_line(rank: int, hit: ScoredChunk) -> str:
    where = " · ".join(p for p in (hit.chunk.section, hit.chunk.pages) if p)
    start = " ".join(hit.chunk.text.split()[:25])
    place = f" ({where})" if where else ""
    return f"{rank}. **{hit.similarity:.3f}** {hit.source.citation}{place}\n   > {start}…"


def compare(questions_file: Path, models: list[str], chunkings: list[str], k: int = 3) -> str:
    """Every question, with every model and chunking: is the expected source at the top?
    hit@1: in first place. hit@3: among the first three. MRR: 1 for first place, 1/2 for second..."""
    questions = json.loads(questions_file.read_text(encoding="utf-8"))["questions"]
    uow = _uow()
    table = ["| מודל | חלוקה | hit@1 | hit@3 | MRR |", "|---|---|---|---|---|"]
    details: list[str] = []
    for model in models:
        embedder = ollama_embedder(model)
        for chunking in chunkings:
            hit1 = hit3 = 0
            reciprocal = 0.0
            details += [f"## {model} · {chunking}", ""]
            for q in questions:
                hits = search_library(uow, embedder, chunking, q["question"], k=max(k, 10))
                titles = list(dict.fromkeys(h.source.title for h in hits))  # sources, best first
                rank = titles.index(q["expected"]) + 1 if q["expected"] in titles else None
                hit1 += rank == 1
                hit3 += rank is not None and rank <= 3
                reciprocal += 1 / rank if rank else 0.0
                mark = "✓" if rank == 1 else (f"מקום {rank}" if rank else "✗ לא נמצא")
                details += [f"**{q['question']}** · צפוי: {q['expected']} · {mark}", ""]
                details += [_hit_line(i, h) for i, h in enumerate(hits[:k], start=1)] + [""]
            n = len(questions)
            table.append(f"| {model} | {chunking} | {hit1}/{n} | {hit3}/{n} | {reciprocal / n:.2f} |")
    return "\n".join(["# השוואת חיפוש: מודלים ושיטות חלוקה", "", *table, "", *details])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.library", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "import"):
        command = commands.add_parser(name)
        command.add_argument("file", type=Path)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--out", type=Path)
    index = commands.add_parser("index")
    index.add_argument("--model", default="all", choices=["all", *EMBEDDING_MODELS])
    index.add_argument("--chunking", default="all", choices=["all", *STRATEGIES])
    search = commands.add_parser("search")
    search.add_argument("question")
    search.add_argument("--model", default="bge-m3", choices=list(EMBEDDING_MODELS))
    search.add_argument("--chunking", default=DEFAULT_CHUNKING, choices=list(STRATEGIES))
    search.add_argument("-k", type=int, default=5)
    comparison = commands.add_parser("compare")
    comparison.add_argument("questions", type=Path)
    comparison.add_argument("--model", default="all", choices=["all", *EMBEDDING_MODELS])
    comparison.add_argument("--chunking", default="all", choices=["all", *STRATEGIES])
    comparison.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    report = None
    if args.command in ("preview", "import"):
        manifest = load_manifest(args.manifest)
        if args.command == "preview":
            report = preview(plan(args.file, manifest), samples=10)
        else:
            file_plan = import_file(_uow(), args.file, manifest)
            counts = {name: sum(len(s.chunks[name]) for s in file_plan.sources) for name in STRATEGIES}
            report = f"Imported {len(file_plan.sources)} sources from {file_plan.file}: chunks {counts}"
    elif args.command == "index":
        for model in _models(args.model):
            embedder = ollama_embedder(model)
            for chunking in _chunkings(args.chunking):
                r = index_library(_uow(), embedder, chunking)
                print(f"{model} · {chunking}: {r.embedded} chunks embedded in {r.seconds}s")
    elif args.command == "search":
        hits = search_library(_uow(), ollama_embedder(args.model), args.chunking, args.question, k=args.k)
        report = "\n".join(_hit_line(i, h) for i, h in enumerate(hits, start=1)) or "No chunks found: run index first"
    elif args.command == "compare":
        report = compare(args.questions, _models(args.model), _chunkings(args.chunking))

    if report is not None:
        print(report)
        if getattr(args, "out", None):
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report + "\n", encoding="utf-8")
            print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
