"""How long personal records about a trainee are kept after the trainee's file is closed.

The rule: when a trainee leaves or graduates, their file is closed (Trainee.closed_at), and no new
notes, talks or progress can be added. After a period the academy sets, what was written ABOUT
them in private is deleted:
* their notes (TraineeNote, the long-term memory), and
* their personal talks (the counselor's raw notes, what was sent to the model, the summary) with
  every proposal made from them, and
* the tasks about them (2026-10-27): "talk to Nadav about home" is written about him, in private.
Goal progress stays: it is the academy's record of the year, and its reasons were checked for
health details when they came from the AI.

Run by whoever operates the server (python -m pacer.maintenance purge-notes), not through the API.
Deleting is real: a proposal that pointed to a deleted note keeps working (note_id becomes NULL).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pacer.repositories.interfaces import PersonalTalkRepo, TaskRepo, TraineeNoteRepo, TraineeRepo

log = logging.getLogger(__name__)

DEFAULT_KEEP_FOR = timedelta(days=365)
MAX_KEEP_FOR = timedelta(days=36_500)  # a hundred years: anything longer is a typo


@dataclass(frozen=True)
class PurgePlan:
    """What is past the period, by trainee id."""

    notes_by_trainee: dict[int, list[int]] = field(default_factory=dict)
    talks_by_trainee: dict[int, list[int]] = field(default_factory=dict)
    tasks_by_trainee: dict[int, list[int]] = field(default_factory=dict)

    @property
    def note_count(self) -> int:
        return sum(len(ids) for ids in self.notes_by_trainee.values())

    @property
    def talk_count(self) -> int:
        return sum(len(ids) for ids in self.talks_by_trainee.values())

    @property
    def task_count(self) -> int:
        return sum(len(ids) for ids in self.tasks_by_trainee.values())

    @property
    def trainee_ids(self) -> set[int]:
        return set(self.notes_by_trainee) | set(self.talks_by_trainee) | set(self.tasks_by_trainee)


def _cutoff(now: datetime, keep_for: timedelta) -> datetime:
    if not timedelta(0) <= keep_for <= MAX_KEEP_FOR:
        raise ValueError(f"The retention period must be between 0 and {MAX_KEEP_FOR.days} days")
    return now - keep_for


def past_retention(
    trainees: TraineeRepo,
    notes: TraineeNoteRepo,
    talks: PersonalTalkRepo,
    tasks: TaskRepo,
    *,
    now: datetime,
    keep_for: timedelta = DEFAULT_KEEP_FOR,
) -> PurgePlan:
    """The notes, talks and tasks of every trainee whose file was closed more than keep_for before now.
    A file closed exactly keep_for ago is still kept: the period has to be OVER."""
    cutoff = _cutoff(now, keep_for)
    plan = PurgePlan()
    for trainee in trainees.list_all():
        if trainee.closed_at is None or trainee.closed_at >= cutoff:
            continue
        if note_ids := [note.key for note in notes.list_by_trainee(trainee.key)]:
            plan.notes_by_trainee[trainee.key] = note_ids
        if talk_ids := [talk.key for talk in talks.list_by_trainee(trainee.key)]:
            plan.talks_by_trainee[trainee.key] = talk_ids
        if task_ids := [task.key for task in tasks.list_by_trainee(trainee.key)]:
            plan.tasks_by_trainee[trainee.key] = task_ids
    return plan


def purge(
    trainees: TraineeRepo,
    notes: TraineeNoteRepo,
    talks: PersonalTalkRepo,
    tasks: TaskRepo,
    plan: PurgePlan,
    *,
    now: datetime,
    keep_for: timedelta = DEFAULT_KEEP_FOR,
) -> PurgePlan:
    """Delete what the plan lists. Call inside one unit of work: all or nothing.

    Each trainee row is LOCKED and checked again first: a file reopened since the plan was made
    (or at this very moment, by another request) is skipped. Returns what was really deleted."""
    cutoff = _cutoff(now, keep_for)
    done = PurgePlan()
    for trainee_id in sorted(plan.trainee_ids):  # one lock order: no two purges wait for each other
        trainee = trainees.lock(trainee_id)
        if trainee is None or trainee.closed_at is None or trainee.closed_at >= cutoff:
            continue
        for note_id in plan.notes_by_trainee.get(trainee_id, []):
            notes.delete(note_id)
        for talk_id in plan.talks_by_trainee.get(trainee_id, []):
            talks.delete_with_proposals(talk_id)
        for task_id in plan.tasks_by_trainee.get(trainee_id, []):
            tasks.delete(task_id)
        for planned, deleted in (
            (plan.notes_by_trainee, done.notes_by_trainee),
            (plan.talks_by_trainee, done.talks_by_trainee),
            (plan.tasks_by_trainee, done.tasks_by_trainee),
        ):
            if trainee_id in planned:
                deleted[trainee_id] = planned[trainee_id]
        log.info(
            "Trainee %s: %s notes, %s talks and %s tasks deleted (retention period over)",
            trainee_id,
            len(done.notes_by_trainee.get(trainee_id, [])),
            len(done.talks_by_trainee.get(trainee_id, [])),
            len(done.tasks_by_trainee.get(trainee_id, [])),
        )
    return done
