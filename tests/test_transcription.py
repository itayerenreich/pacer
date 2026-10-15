"""Recordings on disk, and the Whisper adapter (with a stand-in for faster-whisper: no model needed)."""

import sys
import types
from pathlib import Path

import pytest

from pacer.ai.transcription import RecordingError, RecordingStore, TranscriptionError, WhisperTranscriber


@pytest.fixture
def store(tmp_path) -> RecordingStore:
    return RecordingStore(tmp_path / "recordings")


def files(store: RecordingStore) -> list[str]:
    return sorted(p.name for p in store.directory.iterdir())


# --- RecordingStore -------------------------------------------------------------------------------


def test_a_recording_is_saved_found_and_deleted(store):
    path = store.save(7, iter([b"ab", b"cd"]), ".M4A")
    assert path.name == "meeting-7.m4a" and path.read_bytes() == b"abcd"
    assert store.find(7) == path and store.find(8) is None
    assert store.delete(7) is True
    assert store.find(7) is None and store.delete(7) is False  # deleting twice is fine


def test_a_new_upload_replaces_the_old_one_even_with_another_type(store):
    store.save(7, iter([b"old"]), ".mp3")
    store.save(7, iter([b"new"]), ".wav")
    assert files(store) == ["meeting-7.wav"]


def test_meeting_7_does_not_find_meeting_70(store):
    store.save(70, iter([b"x"]), ".mp3")
    assert store.find(7) is None


@pytest.mark.parametrize("suffix", [".txt", ".exe", "", ".m4a.exe"])
def test_only_audio_types_are_accepted(store, suffix):
    with pytest.raises(RecordingError, match="Unsupported"):
        store.save(7, iter([b"x"]), suffix)


def test_too_large_or_empty_leaves_nothing_behind(store):
    store.save(7, iter([b"earlier"]), ".mp3")
    with pytest.raises(RecordingError, match="larger"):
        store.save(7, iter([b"12345", b"67890"]), ".mp3", max_bytes=8)
    with pytest.raises(RecordingError, match="empty"):
        store.save(7, iter([]), ".mp3")
    assert files(store) == ["meeting-7.mp3"]  # no half-written file, and the earlier one is intact
    assert store.find(7).read_bytes() == b"earlier"


def test_an_upload_that_breaks_midway_leaves_nothing_behind(store):
    def broken():
        yield b"start"
        raise ConnectionError("client went away")

    with pytest.raises(ConnectionError):
        store.save(7, broken(), ".mp3")
    assert files(store) == []


def test_a_missing_folder_means_no_recordings(tmp_path):
    assert RecordingStore(tmp_path / "nowhere").find(1) is None


# --- WhisperTranscriber, with a fake faster_whisper module ------------------------------------------


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    created: list[tuple] = []
    calls: list[dict] = []

    def __init__(self, name, *, device, compute_type):
        _FakeWhisperModel.created.append((name, device, compute_type))

    def transcribe(self, path, **options):
        _FakeWhisperModel.calls.append({"path": path, **options})
        if path.endswith("broken.wav"):
            raise RuntimeError("invalid data found when processing input")
        segments = (_Segment(t) for t in [" רונית: נתחיל. ", "", " איתי: אני על ההסעות. "])
        return segments, types.SimpleNamespace(duration=12.5)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    _FakeWhisperModel.created, _FakeWhisperModel.calls = [], []
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return _FakeWhisperModel


def test_whisper_runs_in_hebrew_on_the_cpu_and_joins_the_segments(fake_faster_whisper, tmp_path):
    transcriber = WhisperTranscriber("some/model")
    text = transcriber.transcribe(tmp_path / "meeting-1.wav")

    assert text == "רונית: נתחיל.\nאיתי: אני על ההסעות."
    assert fake_faster_whisper.created == [("some/model", "cpu", "int8")]
    [call] = fake_faster_whisper.calls
    assert call["language"] == "he" and call["vad_filter"] is True


def test_the_model_is_loaded_once(fake_faster_whisper, tmp_path):
    transcriber = WhisperTranscriber("some/model")
    transcriber.transcribe(tmp_path / "a.wav")
    transcriber.transcribe(tmp_path / "b.wav")
    assert len(fake_faster_whisper.created) == 1


def test_an_unreadable_file_becomes_a_transcription_error(fake_faster_whisper, tmp_path):
    with pytest.raises(TranscriptionError, match="broken.wav: RuntimeError"):
        WhisperTranscriber().transcribe(tmp_path / "broken.wav")


def test_without_faster_whisper_the_error_says_how_to_install_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import fails, as when not installed
    with pytest.raises(TranscriptionError, match="uv sync --extra audio"):
        WhisperTranscriber().transcribe(tmp_path / "a.wav")


def test_whisper_declares_that_it_runs_locally():
    assert WhisperTranscriber.runs_locally is True


def test_availability_is_checked_without_importing(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert WhisperTranscriber.available() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert WhisperTranscriber.available() is True


def test_paths_are_passed_as_strings(fake_faster_whisper, tmp_path):
    WhisperTranscriber().transcribe(Path(tmp_path / "x.wav"))
    assert isinstance(fake_faster_whisper.calls[0]["path"], str)
