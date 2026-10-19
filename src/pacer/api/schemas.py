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

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from pacer.ai.answer import AskResult, Coverage
from pacer.ai.issues import Issue, IssueCode
from pacer.ai.meeting_pipeline import ExtractionResult
from pacer.ai.talk_pipeline import TalkResult
from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    Goal,
    GoalCategory,
    NoteKind,
    PersonalTalk,
    ProposedTask,
    StaffMeeting,
    TalkProposal,
    Trainee,
    TraineeNote,
    TraineeStatus,
)


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


class LibraryQuestion(_Body):
    question: str = Field(min_length=5, max_length=1000, examples=["כמה חניכים יכול מדריך אחד ללוות כמו שצריך?"])


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
