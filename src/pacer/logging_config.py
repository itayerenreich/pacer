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

from pacer.config import LoggingSettings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(level=(level or LoggingSettings().log_level).upper(), format=FORMAT)
