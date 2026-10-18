"""Personal talks, the proposals the AI makes from them, and the notes that become memory.

Who may do what (the same rules as goals, see permissions.py):
    Group counselors and the manager   see the talks, proposals and notes of the trainee.
    Primary counselor and the manager  record talks, run the summary, decide on proposals,
                                       and add or delete notes.
A trainee who left or graduated has a closed file: no new talks, progress or notes.

The AI pipeline (pacer.ai.talk_pipeline) calls submit_proposals() with VALIDATED drafts.
An approved progress proposal goes through GoalService, so it is a regular GoalUpdate with all of
its rules; an approved note becomes a TraineeNote.
"""

import logging
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import (
    NoteKind,
    PersonalTalk,
    ProposalStatus,
    StaffMember,
    TalkProposal,
    TalkProposalKind,
    TalkStatus,
    Trainee,
    TraineeNote,
    present,
)
from pacer.domain.state import require_transition
from pacer.domain.talk import (
    NOTE_MAX_LENGTH,
    REASON_MAX_LENGTH,
    TALK_PROPOSAL_TRANSITIONS,
)
from pacer.repositories.interfaces import (
    GoalRepo,
    GroupRepo,
    PersonalTalkRepo,
    TalkProposalRepo,
    TraineeNoteRepo,
    TraineeRepo,
)

from .errors import RuleViolationError, found
from .goals import GoalService
from .permissions import can_edit_goals, require
from .visibility import visible_trainee

log = logging.getLogger(__name__)


class TalkDraft(BaseModel):
    """One proposal as the pipeline hands it over, before it is stored as a TalkProposal."""

    kind: TalkProposalKind
    goal_id: int | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    note_kind: NoteKind | None = None
    text: str = Field(min_length=5, max_length=REASON_MAX_LENGTH)


class TalkProposalEdits(BaseModel):
    """What the counselor may change before approving. Only the fields that were sent are applied."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=5, max_length=REASON_MAX_LENGTH)
    progress: int | None = Field(default=None, ge=0, le=100)
    note_kind: NoteKind | None = None


class TalkService:
    def __init__(
        self,
        talks: PersonalTalkRepo,
        proposals: TalkProposalRepo,
        notes: TraineeNoteRepo,
        trainees: TraineeRepo,
        groups: GroupRepo,
        goals: GoalRepo,
        goal_service: GoalService,
    ) -> None:
        self._talks = talks
        self._proposals = proposals
        self._notes = notes
        self._trainees = trainees
        self._groups = groups
        self._goals = goals
        self._goal_service = goal_service

    # --- talks ----------------------------------------------------------------------------------

    def record_talk(self, actor: StaffMember, trainee_id: int, *, held_on: date, notes: str) -> PersonalTalk:
        trainee = self._open_trainee(actor, trainee_id)
        require(can_edit_goals(actor, trainee), "record talks with this trainee")
        talk = self._talks.add(
            PersonalTalk(trainee_id=trainee_id, counselor_id=actor.id, held_on=held_on, notes=notes)
        )
        log.info("Talk %s recorded for trainee %s by staff %s", talk.id, trainee_id, actor.id)
        return talk

    def get_talk(self, actor: StaffMember, talk_id: int) -> PersonalTalk:
        talk = found(self._talks.get(talk_id), "Talk", talk_id)
        self._visible(actor, talk.trainee_id, what="Talk", object_id=talk_id)
        return talk

    def list_talks(self, actor: StaffMember, trainee_id: int) -> list[PersonalTalk]:
        self._visible(actor, trainee_id)
        return self._talks.list_by_trainee(trainee_id)

    def talk_to_summarize(self, actor: StaffMember, talk_id: int) -> tuple[PersonalTalk, Trainee]:
        """The talk and its trainee, if this actor may run the summary on it (the pipeline asks)."""
        talk = found(self._talks.get(talk_id), "Talk", talk_id)
        trainee = self._open_trainee(actor, talk.trainee_id, what="Talk", object_id=talk_id)
        require(can_edit_goals(actor, trainee), "summarize talks with this trainee")
        return talk, trainee

    # --- proposals: in from the pipeline, out through the counselor -------------------------------

    def submit_proposals(self, talk_id: int, drafts: list[TalkDraft]) -> list[TalkProposal]:
        """Called by the pipeline with VALIDATED drafts, after it saved the summary."""
        talk = found(self._talks.get(talk_id), "Talk", talk_id)
        if talk.status is not TalkStatus.SUMMARIZED:
            raise RuleViolationError(f"Talk {talk_id} is {talk.status}; proposals come with the summary")
        stored = [self._proposals.add(TalkProposal(talk_id=talk_id, **draft.model_dump())) for draft in drafts]
        log.info("Talk %s: %s proposals waiting for the counselor", talk_id, len(stored))
        self._close_if_done(talk_id)
        return stored

    def list_proposals(self, actor: StaffMember, talk_id: int) -> list[TalkProposal]:
        self.get_talk(actor, talk_id)
        return self._proposals.list_by_talk(talk_id)

    def approve_proposal(
        self, actor: StaffMember, proposal_id: int, edits: TalkProposalEdits | None = None
    ) -> TalkProposal:
        proposal, talk = self._decidable(actor, proposal_id)
        if edits is not None:
            proposal = proposal.edited(**edits.model_dump(exclude_unset=True))

        if proposal.kind is TalkProposalKind.PROGRESS:
            goal_id = present(proposal.goal_id, "a goal on a progress proposal")
            goal = found(self._goals.get(goal_id), "Goal", goal_id)
            if goal.trainee_id != talk.trainee_id:  # checked by the pipeline too; never trust one check
                raise RuleViolationError(f"Goal {goal.id} is not a goal of this talk's trainee")
            progress = present(proposal.progress, "a progress value on a progress proposal")
            text = present(proposal.text, "the text of an open proposal")
            update = self._goal_service.record_progress(actor, goal.key, progress, text)
            approved = proposal.approve(actor.key, goal_update_id=update.key)
        else:
            text = present(proposal.text, "the text of an open proposal")
            if len(text) > NOTE_MAX_LENGTH:
                raise RuleViolationError(f"A note is at most {NOTE_MAX_LENGTH} characters; shorten it first")
            note = self._notes.add(
                TraineeNote(
                    trainee_id=talk.trainee_id,
                    kind=present(proposal.note_kind, "a note kind on a note proposal"),
                    text=text,
                    author_id=actor.key,  # the counselor who approved it owns it, not the model
                    source_talk_id=talk.key,
                )
            )
            approved = proposal.approve(actor.key, note_id=note.key)

        self._proposals.update(approved)
        log.info("Talk proposal %s (%s) approved by staff %s", proposal_id, proposal.kind, actor.id)
        self._close_if_done(talk.key)
        return approved

    def reject_proposal(self, actor: StaffMember, proposal_id: int) -> TalkProposal:
        proposal, talk = self._decidable(actor, proposal_id)
        rejected = self._proposals.update(proposal.reject(actor.key))  # its text is not kept
        log.info("Talk proposal %s rejected by staff %s", proposal_id, actor.id)
        self._close_if_done(talk.key)
        return rejected

    # --- notes: the trainee's long-term memory ------------------------------------------------------

    def list_notes(self, actor: StaffMember, trainee_id: int) -> list[TraineeNote]:
        self._visible(actor, trainee_id)
        return self._notes.list_by_trainee(trainee_id)

    def add_note(self, actor: StaffMember, trainee_id: int, *, kind: NoteKind, text: str) -> TraineeNote:
        trainee = self._open_trainee(actor, trainee_id)
        require(can_edit_goals(actor, trainee), "add notes about this trainee")
        note = self._notes.add(TraineeNote(trainee_id=trainee_id, kind=kind, text=text, author_id=actor.id))
        log.info("Note %s added for trainee %s by staff %s", note.id, trainee_id, actor.id)
        return note

    def delete_note(self, actor: StaffMember, note_id: int) -> None:
        """Allowed even when the trainee's file is closed: removing personal data is always allowed."""
        note = found(self._notes.get(note_id), "Note", note_id)
        trainee = self._visible(actor, note.trainee_id, what="Note", object_id=note_id)
        require(can_edit_goals(actor, trainee), "delete notes about this trainee")
        self._notes.delete(note_id)
        log.info("Note %s about trainee %s deleted by staff %s", note_id, trainee.id, actor.id)

    # --- helpers ----------------------------------------------------------------------------------

    def _decidable(self, actor: StaffMember, proposal_id: int) -> tuple[TalkProposal, PersonalTalk]:
        """The proposal, still open, with its talk LOCKED for the rest of this transaction.

        Talk first, then proposal, the same order everywhere (as in MeetingService): decisions on
        one talk run one after the other, so a double click cannot add a second note or a second
        GoalUpdate, and deciding the last two proposals at once still closes the talk.
        """
        proposal = found(self._proposals.get(proposal_id), "Talk proposal", proposal_id)
        talk = found(self._talks.get(proposal.talk_id), "Talk", proposal.talk_id)
        trainee = self._open_trainee(actor, talk.trainee_id, what="Talk proposal", object_id=proposal_id)
        require(can_edit_goals(actor, trainee), "decide on proposals about this trainee")
        talk = found(self._talks.lock(talk.key), "Talk", talk.key)
        # re-read now that no one else can decide
        proposal = found(self._proposals.lock(proposal_id), "Talk proposal", proposal_id)
        # Still open? Checked BEFORE anything is created.
        require_transition(TALK_PROPOSAL_TRANSITIONS, proposal.status, ProposalStatus.APPROVED, "proposal")
        return proposal, talk

    def _close_if_done(self, talk_id: int) -> None:
        talk = found(self._talks.get(talk_id), "Talk", talk_id)
        if talk.status is TalkStatus.SUMMARIZED and self._proposals.count_open(talk_id) == 0:
            self._talks.update(talk.advance(TalkStatus.REVIEWED))
            log.info("Talk %s: every proposal decided", talk_id)

    def _visible(
        self, actor: StaffMember, trainee_id: int, *, what: str = "Trainee", object_id: int | None = None
    ) -> Trainee:
        return visible_trainee(actor, self._trainees, self._groups, trainee_id, what=what, object_id=object_id)

    def _open_trainee(
        self, actor: StaffMember, trainee_id: int, *, what: str = "Trainee", object_id: int | None = None
    ) -> Trainee:
        """Visible (else 404) and open (else 409), in that order: see services/visibility.py."""
        trainee = self._visible(actor, trainee_id, what=what, object_id=object_id)
        if trainee.is_closed:
            raise RuleViolationError(f"Trainee {trainee_id} is {trainee.status}; the file is closed")
        return trainee
