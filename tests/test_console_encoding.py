"""Hebrew output survives a Windows-style console: the bug where `> out.txt` produced garbage."""

import os
import subprocess
import sys

SCRIPT = "from pacer.logging_config import configure_logging; configure_logging(); print('שלום מכינה')"


def run(script: str) -> bytes:
    # PYTHONIOENCODING=cp1252 makes Python behave like a Windows console redirected to a file
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, check=False).stdout


def test_without_the_fix_hebrew_cannot_even_be_printed():
    assert run("print('שלום מכינה')") == b""  # cp1252 has no Hebrew letters: the print fails


def test_with_the_fix_the_output_is_utf8():
    assert run(SCRIPT).decode("utf-8").strip() == "שלום מכינה"
