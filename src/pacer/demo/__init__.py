"""Synthetic demo data: an invented Hebrew-speaking academy for development and demos.

May import from: domain, repositories.

    uv run python -m pacer.demo            # fill an empty database
    uv run python -m pacer.demo --reset    # empty every table first, then fill
"""

from .academy import Summary, build_academy

__all__ = ["Summary", "build_academy"]
