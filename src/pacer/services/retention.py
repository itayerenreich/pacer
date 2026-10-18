"""How long notes about a trainee are kept after the trainee's file is closed.

The data model's rule: when a trainee leaves or graduates, their notes (TraineeNote, the
long-term memory) are archived, and deleted after a period the academy sets. "Archived" needs
no extra state: a closed file already takes no new notes, talks or progress, and the date it was
closed is Trainee.closed_at. This module deletes what is past the period. It is run by whoever
operates the server (python -m pacer.maintenance purge-notes), not through the API.

Deleting a note is a real delete. A talk proposal that pointed to it keeps working: its note_id
becomes NULL (ON DELETE SET NULL), the same as when a counselor deletes a note by hand.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pacer.repositories.interfaces import TraineeNoteRepo, TraineeRepo

log = logging.getLogger(__name__)

DEFAULT_KEEP_NOTES_FOR = timedelta(days=365)


@dataclass(frozen=True)
class PurgePlan:
    """What is past the period: trainee id -> the ids of their notes."""

    notes_by_trainee: dict[int, list[int]] = field(default_factory=dict)

    @property
    def note_count(self) -> int:
        return sum(len(ids) for ids in self.notes_by_trainee.values())


def notes_past_retention(
    trainees: TraineeRepo, notes: TraineeNoteRepo, *, now: datetime, keep_for: timedelta = DEFAULT_KEEP_NOTES_FOR
) -> PurgePlan:
    """The notes of every trainee whose file was closed more than keep_for before now."""
    if keep_for < timedelta(0):
        raise ValueError("keep_for cannot be negative")
    cutoff = now - keep_for
    plan: dict[int, list[int]] = {}
    for trainee in trainees.list_all():
        if trainee.closed_at is None or trainee.closed_at >= cutoff:  # exactly the period: still kept
            continue
        ids = [note.id for note in notes.list_by_trainee(trainee.id) if note.id is not None]
        if ids:
            plan[trainee.id] = ids  # type: ignore[index]  # a stored trainee has an id
    return PurgePlan(plan)


def purge_notes(notes: TraineeNoteRepo, plan: PurgePlan) -> int:
    """Delete what the plan lists. Call inside one unit of work: all or nothing."""
    for trainee_id, note_ids in plan.notes_by_trainee.items():
        for note_id in note_ids:
            notes.delete(note_id)
        log.info("Trainee %s: %s notes deleted (retention period over)", trainee_id, len(note_ids))
    return plan.note_count
