"""The knowledge library: from a file (Word, PDF, text) to sources and chunks in the database.

May import from: domain, repositories, ai (for indexing). Must not be imported by domain.

    extract.py    a file -> blocks of text (paragraphs), with the page each is on when the format has pages
    clean.py      Hebrew-aware cleaning: invisible characters, broken lines, repeated page headers
    manifest.py   which parts of a file are which sources, with their metadata (a JSON file)
    chunking.py   the chunking strategies, by name ("fixed-200-40", "structure-220")
    ingest.py     all of the above, into the database: importing a file again replaces what came from it
    __main__.py   python -m pacer.library: preview, import, index, search
"""
