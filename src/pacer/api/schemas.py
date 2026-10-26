"""Request bodies: what a client may send, which is less than the domain objects hold.

A client creating a goal sends a title and a category. It does not choose the id,
created_at, status or progress, and the trainee comes from the URL. So each body
here is a separate small model, and the endpoint builds the domain object from it.

The field rules are not written twice: `Annotated[str, Goal.model_fields["title"]]` reuses the
domain's own rule (3-200 characters), so /docs shows it and it can never drift.
Responses are the domain models themselves (Trainee, Goal, ...), since everything
in them may be shown to a staff member who passed the permission check. The few
responses that combine several objects (the result of an AI pipeline) are at the end.
"""

from datetime import date, time
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pacer.ai.answer import AskResult, Coverage
from pacer.ai.coach import CoachOutcome, CoachResult, InvolveRole
from pacer.ai.issues import Issue, IssueCode
from pacer.ai.meeting_pipeline import ExtractionResult
from pacer.ai.talk_pipeline import TalkResult
from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    AttendanceStatus,
    Goal,
    GoalCategory,
    NoteKind,
    PersonalTalk,
    ProposedTask,
    ScheduledSession,
    ScheduleSlot,
    StaffMeeting,
    TalkProposal,
    Trainee,
    TraineeNote,
    TraineeStatus,
    Weekday,
)
from pacer.services import AttendanceSummary, SheetRow


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown field is a client bug: reject it


class TraineeCreate(_Body):
    first_name: Annotated[str, Trainee.model_fields["first_name"]]
    last_name: Annotated[str, Trainee.model_fields["last_name"]]
    cohort_id: int
    group_id: int
    primary_counselor_id: int


class GoalCreate(_Body):
    title: Annotated[str, Goal.model_fields["title"]]
    description: str | None = None
    category: GoalCategory
    target_date: date | None = None


class ProgressUpdate(_Body):
    # Same range as Goal.progress (written here because the domain field has a default
    # of 0, and here progress is required). The domain checks it again anyway.
    progress: Annotated[int, Field(ge=0, le=100, examples=[60])]
    note: Annotated[str, Field(min_length=1, max_length=1000, examples=["Ran 3 km without stopping"])]


class StatusChange(_Body):
    status: TraineeStatus


class ApprovalCreate(_Body):
    kind: ApprovalKind
    trainee_id: int
    reason: Annotated[str, ApprovalRequest.model_fields["reason"]]
    requested_status: TraineeStatus | None = Field(
        default=None, description="Required for kind=status_change, otherwise leave empty"
    )


class Decision(_Body):
    note: Annotated[str | None, ApprovalRequest.model_fields["decision_note"]]


class MeetingCreate(_Body):
    title: Annotated[str, StaffMeeting.model_fields["title"]]
    held_on: date
    participant_ids: list[int] = Field(min_length=1, description="Staff members who took part")
    transcript: str = Field(
        min_length=20,
        max_length=100_000,
        description="The meeting as text: typed notes, or a transcript made elsewhere. Stays on this server; "
        "only an anonymized version reaches the model.",
    )


class TalkCreate(_Body):
    held_on: date
    notes: str = Field(
        min_length=20,
        max_length=20_000,
        description="The counselor's notes from the talk. Stays on this server; only an anonymized "
        "version reaches the model.",
    )


class NoteCreate(_Body):
    kind: NoteKind
    text: Annotated[str, TraineeNote.model_fields["text"]]


# --- responses ------------------------------------------------------------------------------


class IssueOut(BaseModel):
    code: IssueCode
    item: int | None = Field(description="Position in the model's list (1-based); null for the summary")
    detail: str
    section: str | None = Field(default=None, description="Which list, when the answer has several")


    @classmethod
    def of(cls, issue: Issue) -> "IssueOut":
        return cls(code=issue.code, item=issue.item, detail=issue.detail, section=issue.section)


# A question is stripped first: "      " is not 6 characters of question, it is none (422, not a 500 later).
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]


class LibraryQuestion(_Body):
    question: Question = Field(max_length=1000, examples=["כמה חניכים יכול מדריך אחד ללוות כמו שצריך?"])


class CitationOut(BaseModel):
    passage: str = Field(description="The id the answer cites, e.g. P1")
    source_id: int
    source: str = Field(description="The work: title (author, year)")
    evidence: str = Field(description="What kind of source it is, e.g. 'מחקר' or 'מקור ערכי-מסורתי'")
    section: str | None = Field(description="Where in the work, e.g. 'הלכה ה'")
    pages: str | None
    similarity: float = Field(description="How close the passage is to the question (cosine, 0-1)")
    cited: bool = Field(description="Whether the answer relied on it")


class ClaimOut(BaseModel):
    text: str
    passages: list[str]


class AskOut(BaseModel):
    """An answer from the library only, with where each part of it comes from."""

    question: str
    answer: str
    coverage: Coverage = Field(
        description="direct: the sources answer the question; partial: the answer applies sources about a related "
        "situation; none: the library has no source for it"
    )
    insufficient: bool = Field(description="True when the library has no source that answers the question")
    claims: list[ClaimOut]
    citations: list[CitationOut]
    issues: list[IssueOut] = Field(description="What validation found in the model's answer. Check these")
    prompt: str | None = Field(description="The exact prompt version, or null when no model was asked")
    cost_usd: float | None

    @classmethod
    def of(cls, result: AskResult) -> "AskOut":
        return cls(
            question=result.question,
            answer=result.answer,
            coverage=result.coverage,
            insufficient=result.insufficient,
            claims=[ClaimOut(text=c.text, passages=c.passages) for c in result.claims],
            citations=[CitationOut(**vars(c)) for c in result.citations],
            issues=[IssueOut.of(i) for i in result.issues],
            prompt=result.prompt_ref,
            cost_usd=result.cost_usd,
        )


class ProcessResult(BaseModel):
    """What the meeting pipeline did: the proposals to review, and what validation changed."""

    meeting: StaffMeeting
    proposals: list[ProposedTask]
    issues: list[IssueOut] = Field(description="What validation corrected in the model's answer. Check these")
    prompt: str = Field(description="The exact prompt version used, e.g. meeting_extract@v3")
    attempts: int = Field(description="1, or 2 if the model had to correct its first answer")
    cost_usd: float | None
    removed_sentences: int = Field(
        description="Sentences about health that were removed before sending; the transcript still has them"
    )

    @classmethod
    def of(cls, result: ExtractionResult) -> "ProcessResult":
        return cls(
            meeting=result.meeting,
            proposals=result.proposals,
            issues=[IssueOut.of(i) for i in result.issues],
            prompt=result.prompt_ref,
            attempts=result.attempts,
            cost_usd=result.cost_usd,
            removed_sentences=result.removed_sentences,
        )


class SummarizeResult(BaseModel):
    """What the talk pipeline did: the proposals for the counselor, and what validation changed."""

    talk: PersonalTalk
    proposals: list[TalkProposal]
    issues: list[IssueOut] = Field(description="What validation corrected or dropped. Check these")
    prompt: str
    attempts: int
    cost_usd: float | None
    removed_sentences: int = Field(
        description="Sentences about health that were removed before sending; your notes still have them"
    )

    @classmethod
    def of(cls, result: TalkResult) -> "SummarizeResult":
        return cls(
            talk=result.talk,
            proposals=result.proposals,
            issues=[IssueOut.of(i) for i in result.issues],
            prompt=result.prompt_ref,
            attempts=result.attempts,
            cost_usd=result.cost_usd,
            removed_sentences=result.removed_sentences,
        )


# --- the counselor coach (stage 5) -------------------------------------------------------------


class CoachQuestion(_Body):
    question: Question = Field(max_length=2000, examples=["איך לעזור לו להשתתף יותר בשיחות הקבוצה?"])
    trainee_id: int | None = Field(
        default=None, description="The trainee the question is about; omit for a general question"
    )


class ReferralStepOut(BaseModel):
    action: str
    contact: str | None


class ReferralOut(BaseModel):
    """The fixed procedure for an exceptional incident. The same every time, never written by a model."""

    message: str
    steps: list[ReferralStepOut]


class AdviceStepOut(BaseModel):
    action: str
    example: str | None = Field(description="A sentence the counselor could say to the trainee")
    passages: list[str] = Field(description="The library passages it rests on; empty: general advice")
    grounded: bool = Field(description="True when it rests on a passage of the library, false for general advice")


class InvolveOut(BaseModel):
    role: InvolveRole
    label: str = Field(description="The role in Hebrew")
    why: str


class AdviceOut(BaseModel):
    understanding: str
    questions: list[str] = Field(description="What the coach would want to know to advise better")
    steps: list[AdviceStepOut]
    involve: list[InvolveOut] = Field(description="Who else should be part of this; often empty")
    citations: list[CitationOut]
    issues: list[IssueOut] = Field(description="What validation changed in the model's answer. Check these")


class CoachOut(BaseModel):
    """referral: an exceptional incident, the procedure only. advice: the coach's answer.
    not_sent: the whole question was about health, so nothing was sent to any model."""

    outcome: CoachOutcome
    question: str
    trainee_id: int | None
    referral: ReferralOut | None
    advice: AdviceOut | None
    note: str = Field(description="Shown with every answer: what to do if this may be an exceptional incident")
    prompt: str | None = Field(description="The exact prompt version of the advice, or null")
    cost_usd: float | None
    removed_sentences: int = Field(description="Sentences about health removed before anything was sent")

    @classmethod
    def of(cls, result: CoachResult) -> "CoachOut":
        referral = None
        if result.referral is not None:
            referral = ReferralOut(
                message=result.referral.message,
                steps=[ReferralStepOut(action=s.action, contact=s.contact) for s in result.referral.steps],
            )
        advice = None
        if result.advice is not None:
            a = result.advice
            advice = AdviceOut(
                understanding=a.understanding,
                questions=a.questions,
                steps=[
                    AdviceStepOut(action=s.action, example=s.example, passages=s.passages, grounded=s.grounded)
                    for s in a.steps
                ],
                involve=[InvolveOut(role=role, label=role.label, why=why) for role, why in a.involve],
                citations=[CitationOut(**vars(c)) for c in a.citations],
                issues=[IssueOut.of(i) for i in a.issues],
            )
        return cls(
            outcome=result.outcome,
            question=result.question,
            trainee_id=result.trainee_id,
            referral=referral,
            advice=advice,
            note=result.note,
            prompt=result.prompt_ref,
            cost_usd=result.cost_usd,
            removed_sentences=result.removed_sentences,
        )


# --- the weekly schedule and attendance --------------------------------------------------------


class SlotCreate(_Body):
    cohort_id: int
    group_id: int | None = Field(default=None, description="A group's slot (a group talk); omit for the whole cohort")
    weekday: Weekday = Field(description="Python's numbering: 6 is Sunday, 0 Monday ... 4 Friday, 5 Saturday")
    start: time = Field(examples=["07:00"])
    end: time = Field(examples=["08:00"])
    title: Annotated[str, ScheduleSlot.model_fields["title"]]


class SessionCreate(_Body):
    """A one-off, not part of the template: a trip, a guest lecture."""

    cohort_id: int
    group_id: int | None = None
    on: date
    start: time
    end: time
    title: Annotated[str, ScheduledSession.model_fields["title"]]


class SessionChange(_Body):
    """Only what changes. This week only: the template stays as it is."""

    title: Annotated[str | None, ScheduledSession.model_fields["title"]] = None
    on: date | None = None
    start: time | None = None
    end: time | None = None


class SessionOut(BaseModel):
    id: int
    cohort_id: int
    group_id: int | None
    on: date
    weekday: str = Field(description="The day in Hebrew")
    start: time
    end: time
    title: str
    slot_id: int | None = Field(description="The template slot it came from; null for a one-off")
    cancelled: bool

    @classmethod
    def of(cls, session: ScheduledSession) -> "SessionOut":
        return cls(
            **session.model_dump(exclude={"id"}), id=session.key, weekday=Weekday(session.on.weekday()).label
        )


class SlotOut(BaseModel):
    id: int
    cohort_id: int
    group_id: int | None
    weekday: Weekday
    day: str = Field(description="The day in Hebrew")
    start: time
    end: time
    title: str
    active: bool

    @classmethod
    def of(cls, slot: ScheduleSlot) -> "SlotOut":
        return cls(**slot.model_dump(exclude={"id"}), id=slot.key, day=slot.weekday.label)


class MarkIn(_Body):
    trainee_id: int
    status: AttendanceStatus


class AttendanceMarks(_Body):
    marks: list[MarkIn] = Field(min_length=1, max_length=200)


class SheetRowOut(BaseModel):
    trainee_id: int
    name: str
    status: AttendanceStatus | None = Field(description="null: not marked yet")


class SheetOut(BaseModel):
    session: SessionOut
    trainees: list[SheetRowOut] = Field(
        description="The trainees YOU mark here: your groups, or everyone for the manager"
    )

    @classmethod
    def of(cls, session: ScheduledSession, rows: list[SheetRow]) -> "SheetOut":
        return cls(
            session=SessionOut.of(session),
            trainees=[SheetRowOut(trainee_id=r.trainee.key, name=r.trainee.full_name, status=r.status) for r in rows],
        )


class AttendanceSummaryOut(BaseModel):
    trainee_id: int
    name: str
    group_id: int
    rate: float | None = Field(description="present / (present + absent), 0-1; null when nothing counts yet")
    percent: int | None = Field(description="The rate as a whole percent")
    present: int
    absent: int
    excused: int = Field(description="Not counted against the trainee")
    unmarked: int = Field(description="Sessions that already happened and nobody marked: not counted")
    missed_by_title: dict[str, int] = Field(description="Absences (not excused), by session title, most first")

    @classmethod
    def of(cls, s: AttendanceSummary) -> "AttendanceSummaryOut":
        return cls(
            trainee_id=s.trainee.key,
            name=s.trainee.full_name,
            group_id=s.trainee.group_id,
            rate=None if s.rate is None else round(s.rate, 3),
            percent=None if s.rate is None else round(s.rate * 100),
            present=s.present,
            absent=s.absent,
            excused=s.excused,
            unmarked=s.unmarked,
            missed_by_title=s.missed_by_title,
        )
