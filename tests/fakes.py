"""In-memory repositories for fast service tests (no database, no Docker).

Each fake has the same methods as its Protocol in repositories/interfaces.py, so a
service cannot tell it apart from the real SQLAlchemy repository. Objects are
stored as copies, like a database would: changing an object you got back does not
change what is stored.
"""

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    ApprovalStatus,
    AttendanceRecord,
    Chunk,
    Cohort,
    Goal,
    GoalUpdate,
    Group,
    KnowledgeSource,
    PersonalFile,
    PersonalTalk,
    ProposalStatus,
    ProposedTask,
    ScheduledSession,
    ScheduleSlot,
    StaffMeeting,
    StaffMember,
    StaffRole,
    TalkProposal,
    Task,
    Trainee,
    TraineeNote,
)
from pacer.repositories import Repos, ScoredChunk


class _Store[M: BaseModel]:
    """add / get / update by id, with ids handed out like a database sequence.

    Every model goes in and out as a DEEP copy: a list or dict field (Group.counselor_ids) changed by a
    caller must never change the stored row. That is also why FakeUnitOfWork may snapshot rows with a
    shallow copy of the dict (independent review, 2026-10-26)."""

    def __init__(self) -> None:
        self.rows: dict[int, M] = {}

    def add(self, item: M) -> M:
        new_id = max(self.rows, default=0) + 1  # like a sequence: never reuses the id of a deleted row
        saved = item.model_copy(update={"id": new_id}, deep=True)
        self.rows[new_id] = saved.model_copy(deep=True)
        return saved

    def get(self, item_id: int) -> M | None:
        item = self.rows.get(item_id)
        return None if item is None else item.model_copy(deep=True)

    def update(self, item: M) -> M:
        if item.id not in self.rows:
            raise LookupError(f"{type(item).__name__} {item.id} does not exist")
        self.rows[item.id] = item.model_copy(deep=True)
        return item

    def where(self, **fields) -> list[M]:
        return [
            item.model_copy(deep=True)
            for item in self.rows.values()
            if all(getattr(item, name) == value for name, value in fields.items())
        ]


class FakeStaffRepo(_Store[StaffMember]):
    def list_all(self) -> list[StaffMember]:
        return self.where()


class FakeCohortRepo(_Store[Cohort]):
    def list_all(self) -> list[Cohort]:
        return self.where()


class FakeGroupRepo(_Store[Group]):
    def list_by_cohort(self, cohort_id: int) -> list[Group]:
        return self.where(cohort_id=cohort_id)


class FakeTraineeRepo(_Store[Trainee]):
    def list_all(self) -> list[Trainee]:
        return self.where()

    def list_by_group(self, group_id: int) -> list[Trainee]:
        return self.where(group_id=group_id)

    def list_by_counselor(self, counselor_id: int) -> list[Trainee]:
        return self.where(primary_counselor_id=counselor_id)

    def lock(self, trainee_id: int) -> Trainee | None:
        return self.get(trainee_id)


class FakePersonalFileRepo(_Store[PersonalFile]):
    def get_by_trainee(self, trainee_id: int) -> PersonalFile | None:
        matches = self.where(trainee_id=trainee_id)
        return matches[0] if matches else None


class FakeGoalRepo(_Store[Goal]):
    def list_by_trainee(self, trainee_id: int) -> list[Goal]:
        return self.where(trainee_id=trainee_id)


class FakeGoalUpdateRepo(_Store[GoalUpdate]):
    def list_by_goal(self, goal_id: int) -> list[GoalUpdate]:
        return sorted(self.where(goal_id=goal_id), key=lambda u: u.created_at)


class FakeApprovalRequestRepo(_Store[ApprovalRequest]):
    def list_pending(self, requested_by_id: int | None = None) -> list[ApprovalRequest]:
        pending = self.where(status=ApprovalStatus.PENDING)
        if requested_by_id is not None:
            pending = [r for r in pending if r.requested_by_id == requested_by_id]
        return sorted(pending, key=lambda r: r.created_at)

    def exists_with_status(
        self, *, kind: ApprovalKind, trainee_id: int, requested_by_id: int, status: ApprovalStatus
    ) -> bool:
        return bool(
            self.where(kind=kind, trainee_id=trainee_id, requested_by_id=requested_by_id, status=status)
        )


class FakeMeetingRepo(_Store[StaffMeeting]):
    def lock(self, meeting_id: int) -> StaffMeeting | None:
        return self.get(meeting_id)  # one thread, no other transactions: nothing to wait for

    def list_all(self) -> list[StaffMeeting]:
        return sorted(self.where(), key=lambda m: m.held_on, reverse=True)


class FakeProposedTaskRepo(_Store[ProposedTask]):
    def lock(self, proposal_id: int) -> ProposedTask | None:
        return self.get(proposal_id)

    def list_by_meeting(self, meeting_id: int) -> list[ProposedTask]:
        return self.where(meeting_id=meeting_id)

    def count_open(self, meeting_id: int) -> int:
        return len(self.where(meeting_id=meeting_id, status=ProposalStatus.PROPOSED))


class FakeTaskRepo(_Store[Task]):
    def list_by_owner(self, owner_id: int) -> list[Task]:
        return self.where(owner_id=owner_id)

    def list_all(self) -> list[Task]:
        return self.where()


class FakeTalkRepo(_Store[PersonalTalk]):
    def __init__(self, proposals: "FakeTalkProposalRepo | None" = None) -> None:
        super().__init__()
        self._proposals = proposals

    def lock(self, talk_id: int) -> PersonalTalk | None:
        return self.get(talk_id)

    def delete_with_proposals(self, talk_id: int) -> None:
        if self._proposals is not None:
            for proposal in self._proposals.where(talk_id=talk_id):
                self._proposals.rows.pop(proposal.key, None)
        self.rows.pop(talk_id, None)

    def list_by_trainee(self, trainee_id: int) -> list[PersonalTalk]:
        return sorted(self.where(trainee_id=trainee_id), key=lambda t: (t.held_on, t.id), reverse=True)


class FakeTalkProposalRepo(_Store[TalkProposal]):
    def lock(self, proposal_id: int) -> TalkProposal | None:
        return self.get(proposal_id)

    def list_by_talk(self, talk_id: int) -> list[TalkProposal]:
        return self.where(talk_id=talk_id)

    def count_open(self, talk_id: int) -> int:
        return len(self.where(talk_id=talk_id, status=ProposalStatus.PROPOSED))


class FakeTraineeNoteRepo(_Store[TraineeNote]):
    def __init__(self, proposals: FakeTalkProposalRepo | None = None) -> None:
        super().__init__()
        self._proposals = proposals  # to do what ON DELETE SET NULL does in the database

    def delete(self, note_id: int) -> None:
        self.rows.pop(note_id, None)
        if self._proposals is not None:
            for proposal in self._proposals.where(note_id=note_id):
                self._proposals.rows[proposal.id] = proposal.model_copy(update={"note_id": None})

    def list_by_trainee(self, trainee_id: int) -> list[TraineeNote]:
        return sorted(self.where(trainee_id=trainee_id), key=lambda n: (n.created_at, n.id))


class FakeUnitOfWork:
    """A fake of repositories(engine): each call opens a "transaction" over the same fakes.

    All or nothing, like the real one: if the block raises, every store goes back to how it was
    when the block started. It also counts open transactions, so a test can check that a model
    is never called while one is open.
    """

    def __init__(self, repos: Repos) -> None:
        self.repos = repos
        self.open = 0
        self.opened = 0

    @contextmanager
    def __call__(self) -> Iterator[Repos]:
        stores = [getattr(self.repos, name) for name in Repos.__dataclass_fields__]
        kept = ("rows", "embeddings", "_next")  # what each fake store holds
        # rows: a copy of the dict is enough. A store keeps its own deep copy of each model, hands out
        # deep copies, and replaces a row on update, never changes it in place: so the old objects ARE
        # the old state. (A deep copy of every row on every transaction made the suite 2.5x slower.)
        snapshot = [
            {
                a: copy.copy(getattr(store, a)) if a == "rows" else copy.deepcopy(getattr(store, a))
                for a in kept
                if hasattr(store, a)
            }
            for store in stores
        ]
        self.open += 1
        self.opened += 1
        try:
            yield self.repos
        except BaseException:
            for store, saved in zip(stores, snapshot, strict=True):
                for attribute, value in saved.items():
                    setattr(store, attribute, value)
            raise
        finally:
            self.open -= 1


class FakeKnowledgeSourceRepo(_Store[KnowledgeSource]):
    def __init__(self, chunks: "FakeChunkRepo | None" = None) -> None:
        super().__init__()
        self._chunks = chunks

    def add(self, item: KnowledgeSource) -> KnowledgeSource:
        new_id = max(self.rows, default=0) + 1  # ids are never reused, even after a delete
        saved = item.model_copy(update={"id": new_id})
        self.rows[new_id] = saved.model_copy()
        return saved

    def list_all(self) -> list[KnowledgeSource]:
        return [self.rows[i].model_copy() for i in sorted(self.rows)]

    def delete(self, source_id: int) -> None:
        self.rows.pop(source_id, None)
        if self._chunks is not None:  # ON DELETE CASCADE
            for chunk in [c for c in self._chunks.rows.values() if c.source_id == source_id]:
                self._chunks.drop(chunk.key)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"different vector dimensions: {len(a)} and {len(b)}")  # pgvector refuses too
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


class FakeChunkRepo:
    """Chunks and their embeddings in memory. Search is the same math as pgvector, in Python."""

    def __init__(self) -> None:
        self.rows: dict[int, Chunk] = {}
        self.embeddings: dict[tuple[int, str], list[float]] = {}
        self.sources: FakeKnowledgeSourceRepo | None = None  # set by empty_repos, for search
        self._next = 1

    def add_many(self, chunks: list[Chunk]) -> list[Chunk]:
        saved = []
        for chunk in chunks:
            if any(
                c.source_id == chunk.source_id and c.chunking == chunk.chunking and c.ordinal == chunk.ordinal
                for c in self.rows.values()
            ):
                raise ValueError("uq_chunks_position: this source already has a chunk at this position")
            stored = chunk.model_copy(update={"id": self._next})
            self.rows[self._next] = stored
            self._next += 1
            saved.append(stored.model_copy())
        return saved

    def drop(self, chunk_id: int) -> None:
        self.rows.pop(chunk_id, None)
        for key in [key for key in self.embeddings if key[0] == chunk_id]:
            del self.embeddings[key]

    def get_many(self, chunk_ids: list[int]) -> list[Chunk]:
        return [self.rows[i].model_copy() for i in chunk_ids if i in self.rows]

    def list_by_source(self, source_id: int, chunking: str) -> list[Chunk]:
        found = [c for c in self.rows.values() if c.source_id == source_id and c.chunking == chunking]
        return [c.model_copy() for c in sorted(found, key=lambda c: c.ordinal)]

    def delete_by_source(self, source_id: int, chunking: str) -> int:
        doomed = [c.key for c in self.rows.values() if c.source_id == source_id and c.chunking == chunking]
        for chunk_id in doomed:
            self.drop(chunk_id)
        return len(doomed)

    def save_embeddings(self, model: str, vectors: dict[int, list[float]]) -> None:
        for chunk_id, vector in vectors.items():
            if chunk_id not in self.rows:
                raise LookupError(f"chunk {chunk_id} does not exist")  # a foreign key in the database
            self.embeddings[(chunk_id, model)] = list(vector)

    def missing_embeddings(self, model: str, chunking: str) -> list[Chunk]:
        found = [c for c in self.rows.values() if c.chunking == chunking and (c.key, model) not in self.embeddings]
        return [c.model_copy() for c in sorted(found, key=lambda c: (c.source_id, c.ordinal))]

    def search(
        self, model: str, chunking: str, query: list[float], *, k: int, cloud_safe_only: bool = False
    ) -> list[ScoredChunk]:
        assert self.sources is not None
        hits = []
        for (chunk_id, used_model), vector in self.embeddings.items():
            chunk = self.rows[chunk_id]
            source = self.sources.get(chunk.source_id)
            if used_model != model or chunk.chunking != chunking or source is None:
                continue
            if cloud_safe_only and source.local_only:
                continue
            hits.append(ScoredChunk(chunk=chunk.model_copy(), source=source, similarity=_cosine(vector, query)))
        return sorted(hits, key=lambda h: (-h.similarity, h.chunk.key))[:k]


class FakeScheduleSlotRepo(_Store[ScheduleSlot]):
    def list_by_cohort(self, cohort_id: int) -> list[ScheduleSlot]:
        return sorted(self.where(cohort_id=cohort_id), key=lambda s: (s.weekday, s.start, s.id))


class FakeScheduledSessionRepo(_Store[ScheduledSession]):
    """Enforces uq_scheduled_sessions_slot_date on add AND update, as the database does."""

    def _check_unique(self, item: ScheduledSession) -> None:
        if item.slot_id is not None and any(
            s.slot_id == item.slot_id and s.on == item.on and s.id != item.id for s in self.rows.values()
        ):
            raise IntegrityError("uq_scheduled_sessions_slot_date", None, Exception("duplicate"))

    def add(self, item: ScheduledSession) -> ScheduledSession:
        self._check_unique(item)
        return super().add(item)

    def update(self, item: ScheduledSession) -> ScheduledSession:
        self._check_unique(item)
        return super().update(item)

    def list_between(self, cohort_id, first, last):
        found = [s for s in self.where(cohort_id=cohort_id) if first <= s.on <= last]
        return sorted(found, key=lambda s: (s.on, s.start, s.id))

    def slot_dates(self, slot_ids, first, last):
        ids = set(slot_ids)
        return {(s.slot_id, s.on) for s in self.rows.values() if s.slot_id in ids and first <= s.on <= last}


class FakeAttendanceRepo(_Store[AttendanceRecord]):
    def __init__(self, sessions: FakeScheduledSessionRepo | None = None) -> None:
        super().__init__()
        self.sessions = sessions  # for the date range, as the real one joins

    def save(self, record: AttendanceRecord) -> AttendanceRecord:
        # A scan, not where(): where() copies every row, and the demo academy saves ~1,700 of them.
        existing = next(
            (r for r in self.rows.values() if r.session_id == record.session_id and r.trainee_id == record.trainee_id),
            None,
        )
        if existing is not None:
            saved = record.model_copy(update={"id": existing.id})
            self.rows[existing.key] = saved.model_copy()
            return saved
        return self.add(record)

    def list_by_session(self, session_id: int) -> list[AttendanceRecord]:
        return [r.model_copy() for r in self.rows.values() if r.session_id == session_id]

    def list_for_trainees(self, trainee_ids, first, last):
        ids = set(trainee_ids)
        assert self.sessions is not None, "FakeAttendanceRepo needs the sessions fake"
        return [
            r.model_copy()
            for r in self.rows.values()
            if r.trainee_id in ids and (s := self.sessions.rows.get(r.session_id)) and first <= s.on <= last
        ]


def empty_repos() -> Repos:
    """A fresh, empty fake for every repository."""
    talk_proposals = FakeTalkProposalRepo()
    chunks = FakeChunkRepo()
    sources = FakeKnowledgeSourceRepo(chunks)
    chunks.sources = sources
    sessions = FakeScheduledSessionRepo()
    return Repos(
        staff=FakeStaffRepo(),
        cohorts=FakeCohortRepo(),
        groups=FakeGroupRepo(),
        trainees=FakeTraineeRepo(),
        personal_files=FakePersonalFileRepo(),
        goals=FakeGoalRepo(),
        goal_updates=FakeGoalUpdateRepo(),
        approvals=FakeApprovalRequestRepo(),
        meetings=FakeMeetingRepo(),
        proposals=FakeProposedTaskRepo(),
        tasks=FakeTaskRepo(),
        talks=FakeTalkRepo(talk_proposals),
        talk_proposals=talk_proposals,
        notes=FakeTraineeNoteRepo(talk_proposals),
        sources=sources,
        chunks=chunks,
        slots=FakeScheduleSlotRepo(),
        sessions=sessions,
        attendance=FakeAttendanceRepo(sessions),
    )


@dataclass
class World:
    """A small academy for service tests.

    Group A is led by Noa and Omer; Yael is in group A and Noa is her primary counselor.
    Group B is led by Dana, who has nothing to do with Yael.
    """

    staff: FakeStaffRepo = field(default_factory=FakeStaffRepo)
    cohorts: FakeCohortRepo = field(default_factory=FakeCohortRepo)
    groups: FakeGroupRepo = field(default_factory=FakeGroupRepo)
    trainees: FakeTraineeRepo = field(default_factory=FakeTraineeRepo)
    personal_files: FakePersonalFileRepo = field(default_factory=FakePersonalFileRepo)
    goals: FakeGoalRepo = field(default_factory=FakeGoalRepo)
    goal_updates: FakeGoalUpdateRepo = field(default_factory=FakeGoalUpdateRepo)
    approvals: FakeApprovalRequestRepo = field(default_factory=FakeApprovalRequestRepo)
    meetings: FakeMeetingRepo = field(default_factory=FakeMeetingRepo)
    proposals: FakeProposedTaskRepo = field(default_factory=FakeProposedTaskRepo)
    tasks: FakeTaskRepo = field(default_factory=FakeTaskRepo)
    talks: FakeTalkRepo = field(default_factory=FakeTalkRepo)
    talk_proposals: FakeTalkProposalRepo = field(default_factory=FakeTalkProposalRepo)
    notes: FakeTraineeNoteRepo = field(default_factory=FakeTraineeNoteRepo)
    chunks: FakeChunkRepo = field(default_factory=FakeChunkRepo)
    sources: FakeKnowledgeSourceRepo = field(default_factory=FakeKnowledgeSourceRepo)
    slots: FakeScheduleSlotRepo = field(default_factory=FakeScheduleSlotRepo)
    sessions: FakeScheduledSessionRepo = field(default_factory=FakeScheduledSessionRepo)
    attendance: FakeAttendanceRepo = field(default_factory=FakeAttendanceRepo)

    @property
    def repos(self) -> Repos:
        return Repos(**{name: getattr(self, name) for name in Repos.__dataclass_fields__})

    def __post_init__(self) -> None:
        self.notes = FakeTraineeNoteRepo(self.talk_proposals)
        self.talks = FakeTalkRepo(self.talk_proposals)
        self.sources = FakeKnowledgeSourceRepo(self.chunks)
        self.chunks.sources = self.sources
        self.attendance = FakeAttendanceRepo(self.sessions)
        self.uow = FakeUnitOfWork(self.repos)
        self.manager = self.staff.add(StaffMember(full_name="Ronit Manager", role=StaffRole.MANAGER))
        self.noa = self.staff.add(StaffMember(full_name="Noa Levi", role=StaffRole.COUNSELOR))
        self.omer = self.staff.add(StaffMember(full_name="Omer Cohen", role=StaffRole.COUNSELOR))
        self.dana = self.staff.add(StaffMember(full_name="Dana Katz", role=StaffRole.COUNSELOR))

        self.cohort = self.cohorts.add(
            Cohort(name="Mahzor 12", start_date=date(2026, 9, 1), end_date=date(2027, 6, 30))
        )
        self.group_a = self.groups.add(
            Group(name="Group A", cohort_id=self.cohort.id, counselor_ids=[self.noa.id, self.omer.id])
        )
        self.group_b = self.groups.add(
            Group(name="Group B", cohort_id=self.cohort.id, counselor_ids=[self.dana.id])
        )

        self.yael = self.trainees.add(
            Trainee(
                first_name="Yael",
                last_name="Mizrahi",
                cohort_id=self.cohort.id,
                group_id=self.group_a.id,
                primary_counselor_id=self.noa.id,
            )
        )
        self.personal_files.add(PersonalFile(trainee_id=self.yael.id, intake_answers={"Hobbies": "Guitar"}))
        self.goal = self.goals.add(Goal(trainee_id=self.yael.id, title="Run 5 km", category="physical"))
