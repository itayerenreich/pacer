"""One place that sets up logging for every entry point (API server, demo loader).

Why logging and not print():
* Levels: DEBUG for details while hunting a bug, INFO for normal events, WARNING for
  something suspicious, ERROR for failures. LOG_LEVEL in .env picks what is shown,
  without touching the code.
* Every line says when, how severe, and which module wrote it.
* Modules only call logging.getLogger(__name__); where the lines go is decided here.

Privacy: log ids, never names or personal-file contents. Logs are copied around
(terminals, files, CI), and they must not become a second copy of sensitive data.
"""

import logging
import sys

from pacer.config import LoggingSettings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    use_utf8_console()
    logging.basicConfig(level=(level or LoggingSettings().log_level).upper(), format=FORMAT)


def use_utf8_console() -> None:
    """Print and log in UTF-8 everywhere.

    On Windows, Python writes to a pipe or a file in the system code page (cp1255/cp1252),
    so `python -m pacer.ai.check > out.txt` turns Hebrew into garbage when the file is opened
    as UTF-8. Every entry point calls configure_logging(), so fixing it here fixes it for all.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # real console streams; pytest's capture has no reconfigure
            stream.reconfigure(encoding="utf-8")
