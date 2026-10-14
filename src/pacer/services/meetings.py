"""Staff meetings, the proposals the AI makes from them, and the real tasks.

Who may do what:
    Manager      creates meetings, reviews proposals (approve, edit then approve, reject),
                 sees every task.
    Staff member sees the tasks they own and marks them done. Never sees proposals or transcripts.

The AI pipeline (task 6) calls submit_proposals() after it has validated the model's answer.
From that moment nothing becomes a real Task without a manager's decision.
"""

import logging
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import (
    MeetingStatus,
    ProposalStatus,
    ProposedTask,
    StaffMeeting,
    StaffMember,
    Task,
)
from pacer.domain.meeting import PROPOSAL_TRANSITIONS
from pacer.domain.state import require_transition
from pacer.repositories.interfaces import MeetingRepo, ProposedTaskRepo, StaffRepo, TaskRepo

from .errors import RuleViolationError, found
from .permissions import is_manager, require

log = logging.getLogger(__name__)


class TaskDraft(BaseModel):
    """One task as the pipeline proposes it, before it is stored as a ProposedTask."""

    description: str = Field(min_length=5, max_length=300)
    owner_id: int | None = None
    due_date: date | None = None


class ProposalEdits(BaseModel):
    """What a manager may change before approving. Only the fields that were sent are applied."""

    model_config = ConfigDict(extra="forbid")  # not status, not task_id: only these three

    description: str | None = Field(default=None, min_length=5, max_length=300)
    owner_id: int | None = None
    due_date: date | None = None


class MeetingService:
    def __init__(
        self, meetings: MeetingRepo, proposals: ProposedTaskRepo, tasks: TaskRepo, staff: StaffRepo
    ) -> None:
        self._meetings = meetings
        self._proposals = proposals
        self._tasks = tasks
        self._staff = staff

    # --- meetings ---------------------------------------------------------------------------

    def create_meeting(
        self,
        actor: StaffMember,
        *,
        title: str,
        held_on: date,
        participant_ids: list[int],
        transcript: str | None = None,
    ) -> StaffMeeting:
        """A meeting from a text transcript starts TRANSCRIBED; one waiting for audio, UPLOADED."""
        require(is_manager(actor), "create staff meetings")
        for staff_id in participant_ids:
            found(self._staff.get(staff_id), "Staff member", staff_id)
        status = MeetingStatus.TRANSCRIBED if transcript is not None else MeetingStatus.UPLOADED
        meeting = self._meetings.add(
            StaffMeeting(
                title=title,
                held_on=held_on,
                created_by_id=actor.id,
                participant_ids=participant_ids,
                status=status,
                transcript=transcript,
            )
        )
        log.info("Meeting %s created by staff %s (%s)", meeting.id, actor.id, status)
        return meeting

    def get_meeting(self, actor: StaffMember, meeting_id: int) -> StaffMeeting:
        require(is_manager(actor), "view staff meeting records")  # transcripts are personal data
        return found(self._meetings.get(meeting_id), "Meeting", meeting_id)

    def list_meetings(self, actor: StaffMember) -> list[StaffMeeting]:
        require(is_manager(actor), "view staff meeting records")
        return self._meetings.list_all()

    # --- proposals: in from the pipeline, out through the manager ---------------------------------

    def submit_proposals(self, meeting_id: int, drafts: list[TaskDraft]) -> list[ProposedTask]:
        """Called by the pipeline with VALIDATED drafts. Moves the meeting to pending approval."""
        meeting = found(self._meetings.get(meeting_id), "Meeting", meeting_id)
        if meeting.status is not MeetingStatus.EXTRACTED:
            raise RuleViolationError(
                f"Meeting {meeting_id} is {meeting.status}; proposals come after extraction"
            )
        stored = [
            self._proposals.add(ProposedTask(meeting_id=meeting_id, **draft.model_dump())) for draft in drafts
        ]
        self._meetings.update(meeting.advance(MeetingStatus.PENDING_APPROVAL))
        log.info("Meeting %s: %s proposals waiting for approval", meeting_id, len(stored))
        self._close_if_done(meeting_id)  # a meeting with no proposals needs no decisions
        return stored

    def list_proposals(self, actor: StaffMember, meeting_id: int) -> list[ProposedTask]:
        require(is_manager(actor), "review proposed tasks")
        found(self._meetings.get(meeting_id), "Meeting", meeting_id)
        return self._proposals.list_by_meeting(meeting_id)

    def approve_proposal(
        self, actor: StaffMember, proposal_id: int, edits: ProposalEdits | None = None
    ) -> Task:
        """Apply the manager's edits, check the result, create the real Task, record the decision."""
        require(is_manager(actor), "approve proposed tasks")
        proposal = found(self._proposals.get(proposal_id), "Proposed task", proposal_id)
        # Check the decision is still open BEFORE creating anything (a second approve must not
        # leave a second Task behind, even outside a database transaction).
        require_transition(PROPOSAL_TRANSITIONS, proposal.status, ProposalStatus.APPROVED, "proposal")
        if edits is not None:
            proposal = proposal.edited(**edits.model_dump(exclude_unset=True))
        meeting = found(self._meetings.get(proposal.meeting_id), "Meeting", proposal.meeting_id)

        if proposal.owner_id is None:
            raise RuleViolationError(
                "Choose an owner before approving (the model named no one we could match)"
            )
        found(self._staff.get(proposal.owner_id), "Staff member", proposal.owner_id)
        if proposal.due_date is not None and proposal.due_date < meeting.held_on:
            raise RuleViolationError("The due date is before the meeting itself")

        task = self._tasks.add(
            Task(
                description=proposal.description,
                owner_id=proposal.owner_id,
                due_date=proposal.due_date,
                created_by_id=actor.id,
                source_meeting_id=meeting.id,
            )
        )
        self._proposals.update(proposal.approve(actor.id, task.id))  # InvalidTransition if decided
        log.info(
            "Proposal %s approved by staff %s -> task %s for staff %s",
            proposal_id,
            actor.id,
            task.id,
            task.owner_id,
        )
        self._close_if_done(meeting.id)
        return task

    def reject_proposal(self, actor: StaffMember, proposal_id: int) -> ProposedTask:
        require(is_manager(actor), "reject proposed tasks")
        proposal = found(self._proposals.get(proposal_id), "Proposed task", proposal_id)
        rejected = self._proposals.update(proposal.reject(actor.id))
        log.info("Proposal %s rejected by staff %s", proposal_id, actor.id)
        self._close_if_done(proposal.meeting_id)
        return rejected

    def _close_if_done(self, meeting_id: int) -> None:
        meeting = self._meetings.get(meeting_id)
        if meeting.status is MeetingStatus.PENDING_APPROVAL and self._proposals.count_open(meeting_id) == 0:
            self._meetings.update(meeting.advance(MeetingStatus.APPROVED))
            log.info("Meeting %s: every proposal decided", meeting_id)


class TaskService:
    def __init__(self, tasks: TaskRepo) -> None:
        self._tasks = tasks

    def my_tasks(self, actor: StaffMember) -> list[Task]:
        return self._tasks.list_by_owner(actor.id)

    def all_tasks(self, actor: StaffMember) -> list[Task]:
        require(is_manager(actor), "view everyone's tasks")
        return self._tasks.list_all()

    def complete(self, actor: StaffMember, task_id: int) -> Task:
        task = found(self._tasks.get(task_id), "Task", task_id)
        require(task.owner_id == actor.id or is_manager(actor), "complete someone else's task")
        done = self._tasks.update(task.complete())  # InvalidTransition if already done
        log.info("Task %s completed by staff %s", task_id, actor.id)
        return done
