"""Speech to text, on this computer: recordings of staff meetings become transcripts.

A recording is the most personal artifact Pacer handles (voices, names, everything said), so:
* It is transcribed ONLY by a model running on this computer. Transcribers declare
  `runs_locally`, and the pipeline refuses one that does not (the same rule as the router).
* It is stored in a local folder outside Git, and deleted as soon as its transcript is saved.

    store = RecordingStore(Path("data/recordings"))
    store.save(meeting_id=7, chunks=upload, suffix=".m4a")
    text = WhisperTranscriber().transcribe(store.find(7))
    ... save the transcript ...
    store.delete(7)

WhisperTranscriber uses faster-whisper, an optional dependency (`uv sync --extra audio`), so the
rest of Pacer, CI and the tests never need it. The default model, ivrit-ai's Whisper large-v3-turbo,
is fine-tuned for Hebrew and converted for faster-whisper. It is downloaded once (about 1.6 GB)
from Hugging Face on first use; after that it runs offline. Downloading a model sends nothing.
"""

import importlib.util
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".wav", ".ogg", ".oga", ".webm", ".flac", ".aac"})
MAX_RECORDING_BYTES = 300 * 1024 * 1024  # about 5 hours of compressed speech


class TranscriptionError(Exception):
    """A recording could not be transcribed (missing dependency, unreadable file...)."""


class RecordingError(Exception):
    """A recording was rejected: wrong type, too large, or empty."""


class Transcriber(Protocol):
    runs_locally: bool  # must be True: recordings never leave this computer

    def transcribe(self, audio: Path) -> str: ...


class WhisperTranscriber:
    """faster-whisper on the CPU (int8). The model is loaded once, on first use.

    One transcription at a time: a large model on a 16 GB laptop has no room for two, and
    running two in parallel on the same cores would not finish sooner anyway.
    """

    runs_locally = True

    def __init__(
        self,
        model: str = DEFAULT_WHISPER_MODEL,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "he",
    ) -> None:
        self.model = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._loaded = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        """Is faster-whisper installed? (Checked without importing it: the import is slow.)"""
        return importlib.util.find_spec("faster_whisper") is not None

    def transcribe(self, audio: Path) -> str:
        with self._lock:
            whisper = self._load()
            try:
                segments, info = whisper.transcribe(
                    str(audio),
                    language=self._language,  # don't let it guess: short Hebrew clips get misdetected
                    vad_filter=True,  # skip silences: faster, and fewer invented words in them
                    beam_size=5,
                )
                lines = [segment.text.strip() for segment in segments]  # a generator: runs here
            except Exception as error:  # an unreadable or corrupt file, among others
                raise TranscriptionError(f"Could not transcribe {audio.name}: {type(error).__name__}") from error
        log.info("Transcribed %s: %.0f seconds of audio", audio.name, info.duration)
        return "\n".join(line for line in lines if line)

    def _load(self):
        if self._loaded is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise TranscriptionError(
                    "Audio transcription is not installed. Run: uv sync --extra audio"
                ) from error
            log.info("Loading Whisper model %s (%s, %s)", self.model, self._device, self._compute_type)
            self._loaded = WhisperModel(self.model, device=self._device, compute_type=self._compute_type)
        return self._loaded


class RecordingStore:
    """Recordings waiting to be transcribed, one per meeting: <directory>/meeting-<id>.<ext>."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(
        self, meeting_id: int, chunks: Iterable[bytes], suffix: str, *, max_bytes: int = MAX_RECORDING_BYTES
    ) -> Path:
        """Stream the upload to disk. Replaces an earlier recording of the same meeting.

        Written to a temporary file first and renamed only when complete, so a failed or too-large
        upload never leaves half a recording where the transcriber would find it.
        """
        suffix = suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise RecordingError(f"Unsupported audio type {suffix or '(none)'}; use one of {sorted(AUDIO_SUFFIXES)}")
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=".upload-")
        written = 0
        try:
            with os.fdopen(handle, "wb") as file:
                for chunk in chunks:
                    written += len(chunk)
                    if written > max_bytes:
                        raise RecordingError(f"The recording is larger than {max_bytes // (1024 * 1024)} MB")
                    file.write(chunk)
            if written == 0:
                raise RecordingError("The recording is empty")
            self.delete(meeting_id)  # an earlier upload with another extension
            target = self.directory / f"meeting-{meeting_id}{suffix}"
            os.replace(temporary, target)  # atomic: the full file appears at once, or nothing
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        log.info("Recording saved for meeting %s (%s bytes)", meeting_id, written)
        return target

    def find(self, meeting_id: int) -> Path | None:
        pattern = re.compile(rf"^meeting-{meeting_id}\.[a-z0-9]+$")
        if not self.directory.is_dir():
            return None
        matches = [p for p in self.directory.iterdir() if pattern.match(p.name)]
        return matches[0] if matches else None

    def delete(self, meeting_id: int) -> bool:
        """Delete the meeting's recording, if there is one. True if a file was deleted."""
        path = self.find(meeting_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        log.info("Recording of meeting %s deleted", meeting_id)
        return True
