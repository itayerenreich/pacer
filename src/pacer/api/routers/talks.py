"""Endpoints for personal talks, the proposals the AI makes from them, and trainee notes.

Permissions follow the goals: the trainee's group counselors and the manager see; the primary
counselor and the manager write and decide. The summarize endpoint waits for a model, so it uses
DetachedActor and the pipeline, never SessionDep (no transaction held open meanwhile).
"""

from fastapi import APIRouter, status

from pacer.domain import PersonalTalk, TalkProposal, TraineeNote
from pacer.services import TalkProposalEdits

from ..dependencies import Actor, DetachedActor, TalkPipelineDep, TalkServiceDep
from ..schemas import NoteCreate, SummarizeResult, TalkCreate

router = APIRouter(tags=["talks"])


@router.post(
    "/trainees/{trainee_id}/talks",
    status_code=status.HTTP_201_CREATED,
    summary="Record a personal talk with a trainee (primary counselor or manager)",
)
def record_talk(trainee_id: int, body: TalkCreate, actor: Actor, service: TalkServiceDep) -> PersonalTalk:
    return service.record_talk(actor, trainee_id, held_on=body.held_on, notes=body.notes)


@router.get("/trainees/{trainee_id}/talks", summary="A trainee's talks, newest first")
def list_talks(trainee_id: int, actor: Actor, service: TalkServiceDep) -> list[PersonalTalk]:
    return service.list_talks(actor, trainee_id)


@router.get("/talks/{talk_id}", summary="One talk, with its notes and summary")
def get_talk(talk_id: int, actor: Actor, service: TalkServiceDep) -> PersonalTalk:
    return service.get_talk(actor, talk_id)


@router.post(
    "/talks/{talk_id}/summarize",
    summary="AI: summary, goal progress proposals and notes from the talk (primary counselor or manager)",
    description="Only an anonymized version of the notes reaches the cloud model. Nothing is applied: "
    "every proposal waits for the counselor. The result lists what validation corrected (issues).",
)
def summarize_talk(talk_id: int, actor: DetachedActor, pipeline: TalkPipelineDep) -> SummarizeResult:
    return SummarizeResult.of(pipeline.summarize(actor, talk_id))


@router.get("/talks/{talk_id}/proposals", summary="The proposals made from a talk")
def list_talk_proposals(talk_id: int, actor: Actor, service: TalkServiceDep) -> list[TalkProposal]:
    return service.list_proposals(actor, talk_id)


@router.post(
    "/talk-proposals/{proposal_id}/approve",
    summary="Approve a proposal, optionally edited first",
    description="Send only what to change: text, and progress (progress proposal) or note_kind (note). "
    "A progress proposal becomes a regular goal update; a note becomes part of the trainee's notes.",
)
def approve_talk_proposal(
    proposal_id: int, actor: Actor, service: TalkServiceDep, edits: TalkProposalEdits | None = None
) -> TalkProposal:
    return service.approve_proposal(actor, proposal_id, edits)


@router.post(
    "/talk-proposals/{proposal_id}/reject",
    summary="Reject a proposal (its text is not kept)",
)
def reject_talk_proposal(proposal_id: int, actor: Actor, service: TalkServiceDep) -> TalkProposal:
    return service.reject_proposal(actor, proposal_id)


@router.get("/trainees/{trainee_id}/notes", summary="What the staff approved to remember about a trainee")
def list_notes(trainee_id: int, actor: Actor, service: TalkServiceDep) -> list[TraineeNote]:
    return service.list_notes(actor, trainee_id)


@router.post(
    "/trainees/{trainee_id}/notes",
    status_code=status.HTTP_201_CREATED,
    summary="Add a note by hand (primary counselor or manager)",
    description="Educational and behavioral observations only: no health information or diagnoses.",
)
def add_note(trainee_id: int, body: NoteCreate, actor: Actor, service: TalkServiceDep) -> TraineeNote:
    return service.add_note(actor, trainee_id, kind=body.kind, text=body.text)


@router.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a note for good (primary counselor or manager)",
)
def delete_note(note_id: int, actor: Actor, service: TalkServiceDep) -> None:
    service.delete_note(actor, note_id)
