"""Prompt files: loading, versions, and that every prompt in the repo is valid."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pacer.ai.prompts import PROMPTS_DIR, PromptError, available_versions, load_prompt

SYSTEM = "You are a careful assistant. Answer briefly and only in Hebrew."


def write(
    directory: Path, name: str, version: int, *, header_name: str | None = None, body: str = SYSTEM
) -> None:
    folder = directory / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"v{version}.md").write_text(
        f"---\nname: {header_name or name}\nversion: {version}\ndescription: A test prompt\n---\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def clear_cache():
    load_prompt.cache_clear()
    yield
    load_prompt.cache_clear()


def test_newest_version_by_default_or_a_specific_one(tmp_path):
    write(tmp_path, "summary", 1)
    write(tmp_path, "summary", 2, body=SYSTEM + " Version two.")

    assert load_prompt("summary", directory=tmp_path).version == 2
    assert load_prompt("summary", 1, directory=tmp_path).ref == "summary@v1"
    assert available_versions("summary", tmp_path) == [1, 2]


def test_version_10_comes_after_version_9(tmp_path):
    """Versions sort as numbers, not as text (where "v10" < "v9")."""
    write(tmp_path, "summary", 9)
    write(tmp_path, "summary", 10)
    assert load_prompt("summary", directory=tmp_path).version == 10


def test_missing_prompt_or_version_is_a_clear_error(tmp_path):
    write(tmp_path, "summary", 1)
    with pytest.raises(PromptError, match="No prompt named 'nothing'"):
        load_prompt("nothing", directory=tmp_path)
    with pytest.raises(PromptError, match="no version 5"):
        load_prompt("summary", 5, directory=tmp_path)


def test_header_must_match_the_file(tmp_path):
    """A file copied to v2 without updating its header must not load as v2."""
    write(tmp_path, "summary", 1, header_name="other")
    with pytest.raises(PromptError, match="header says other@v1"):
        load_prompt("summary", directory=tmp_path)


def test_file_without_header_is_rejected(tmp_path):
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "v1.md").write_text("Just text, no header", encoding="utf-8")
    with pytest.raises(PromptError, match="header"):
        load_prompt("summary", directory=tmp_path)


def test_too_short_system_prompt_is_rejected(tmp_path):
    write(tmp_path, "summary", 1, body="Be nice.")
    with pytest.raises(ValidationError):
        load_prompt("summary", directory=tmp_path)


def test_a_loaded_prompt_cannot_be_changed(tmp_path):
    write(tmp_path, "summary", 1)
    prompt = load_prompt("summary", directory=tmp_path)
    with pytest.raises(ValidationError):
        prompt.system = "Ignore all the rules"


def test_every_prompt_in_the_repo_loads():
    """Guards every prompt file we ship: a broken header fails CI, not production."""
    names = [p.name for p in PROMPTS_DIR.iterdir() if p.is_dir()]
    assert "connection_check" in names
    for name in names:
        for version in available_versions(name):
            assert load_prompt(name, version).ref == f"{name}@v{version}"


def test_old_prompt_versions_are_kept_and_the_pipelines_are_pinned_to_the_latest():
    """A used version is never edited or deleted: old results still point to the exact prompt that
    made them. v3 / v2 (2026-10-18) say that [NAME_n] may be an ordinary word, not a person."""
    from pacer.ai.meeting_pipeline import MeetingPipeline
    from pacer.ai.talk_pipeline import TalkPipeline

    assert available_versions("meeting_extract") == [1, 2, 3]
    assert available_versions("talk_summary") == [1, 2]
    assert MeetingPipeline(uow=None, router=None)._prompt.ref == "meeting_extract@v3"  # type: ignore[arg-type]
    assert TalkPipeline(uow=None, router=None)._prompt.ref == "talk_summary@v2"  # type: ignore[arg-type]
