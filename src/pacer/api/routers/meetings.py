"""Endpoints for staff meetings and the tasks the AI proposes from them. Manager only.

The pipeline endpoint (process) uses DetachedActor and the pipeline, never SessionDep:
it waits for a model, and must not hold a database transaction open meanwhile.
"""

from fastapi import APIRouter, status

from pacer.domain import ProposedTask, StaffMeeting, Task
from pacer.services import ProposalEdits

from ..dependencies import Actor, DetachedActor, MeetingPipelineDep, MeetingServiceDep
from ..schemas import MeetingCreate, ProcessResult

router = APIRouter(tags=["meetings"])


@router.post(
    "/meetings",
    status_code=status.HTTP_201_CREATED,
    summary="Record a staff meeting (manager)",
    description="The meeting's text (typed notes, or a transcript made elsewhere) is stored on this server. "
    "Then POST /meetings/{id}/process.",
)
def create_meeting(body: MeetingCreate, actor: Actor, service: MeetingServiceDep) -> StaffMeeting:
    return service.create_meeting(actor, **body.model_dump())


@router.get("/meetings", summary="All staff meetings, newest first (manager)")
def list_meetings(actor: Actor, service: MeetingServiceDep) -> list[StaffMeeting]:
    return service.list_meetings(actor)


@router.get("/meetings/{meeting_id}", summary="One staff meeting, with its transcript and summary (manager)")
def get_meeting(meeting_id: int, actor: Actor, service: MeetingServiceDep) -> StaffMeeting:
    return service.get_meeting(actor, meeting_id)


@router.post(
    "/meetings/{meeting_id}/process",
    summary="Run the AI pipeline: anonymize, extract proposed tasks (manager)",
    description="Continues from where the meeting is, so after a failure just call it again. Only the "
    "anonymized transcript is sent to the cloud model. The result lists the proposals to review, and "
    "every correction validation made to the model's answer (issues).",
)
def process_meeting(meeting_id: int, actor: DetachedActor, pipeline: MeetingPipelineDep) -> ProcessResult:
    return ProcessResult.of(pipeline.process(actor, meeting_id))


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
