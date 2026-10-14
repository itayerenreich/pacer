"""Endpoints for staff meetings and the tasks the AI proposes from them. Manager only."""

from fastapi import APIRouter, status

from pacer.domain import ProposedTask, StaffMeeting, Task
from pacer.services import ProposalEdits

from ..dependencies import Actor, MeetingServiceDep
from ..schemas import MeetingCreate

router = APIRouter(tags=["meetings"])


@router.post(
    "/meetings",
    status_code=status.HTTP_201_CREATED,
    summary="Record a staff meeting (manager)",
    description="With a text transcript the meeting starts as 'transcribed'. Processing it into "
    "proposed tasks is the AI pipeline (stage 3, task 6).",
)
def create_meeting(body: MeetingCreate, actor: Actor, service: MeetingServiceDep) -> StaffMeeting:
    return service.create_meeting(actor, **body.model_dump())


@router.get("/meetings", summary="All staff meetings, newest first (manager)")
def list_meetings(actor: Actor, service: MeetingServiceDep) -> list[StaffMeeting]:
    return service.list_meetings(actor)


@router.get("/meetings/{meeting_id}", summary="One staff meeting, with its transcript and summary (manager)")
def get_meeting(meeting_id: int, actor: Actor, service: MeetingServiceDep) -> StaffMeeting:
    return service.get_meeting(actor, meeting_id)


@router.get("/meetings/{meeting_id}/proposals", summary="The tasks proposed for a meeting (manager)")
def list_proposals(meeting_id: int, actor: Actor, service: MeetingServiceDep) -> list[ProposedTask]:
    return service.list_proposals(actor, meeting_id)


@router.post(
    "/proposals/{proposal_id}/approve",
    summary="Approve a proposed task, optionally edited first (manager)",
    description="Send only the fields to change (description, owner_id, due_date). Creates the real task.",
)
def approve_proposal(
    proposal_id: int, actor: Actor, service: MeetingServiceDep, edits: ProposalEdits | None = None
) -> Task:
    return service.approve_proposal(actor, proposal_id, edits)


@router.post("/proposals/{proposal_id}/reject", summary="Reject a proposed task (manager)")
def reject_proposal(proposal_id: int, actor: Actor, service: MeetingServiceDep) -> ProposedTask:
    return service.reject_proposal(actor, proposal_id)
