"""Endpoints under /approvals: counselors ask, the manager decides."""

from fastapi import APIRouter, status

from pacer.domain import ApprovalRequest

from ..dependencies import Actor, ApprovalServiceDep
from ..schemas import ApprovalCreate, Decision

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Ask the manager for approval (counselor)")
def request_approval(body: ApprovalCreate, actor: Actor, service: ApprovalServiceDep) -> ApprovalRequest:
    return service.request(actor, body.kind, body.trainee_id, body.reason, body.requested_status)


@router.get(
    "/pending",
    summary="Pending requests",
    description="The manager sees all of them; a counselor sees only their own.",
)
def list_pending(actor: Actor, service: ApprovalServiceDep) -> list[ApprovalRequest]:
    return service.list_pending(actor)


@router.post(
    "/{request_id}/approve",
    summary="Approve a request (manager)",
    description="For a status change, the trainee's status changes in the same transaction.",
)
def approve(
    request_id: int, actor: Actor, service: ApprovalServiceDep, body: Decision | None = None
) -> ApprovalRequest:
    return service.approve(actor, request_id, body.note if body else None)


@router.post("/{request_id}/reject", summary="Reject a request (manager)")
def reject(
    request_id: int, actor: Actor, service: ApprovalServiceDep, body: Decision | None = None
) -> ApprovalRequest:
    return service.reject(actor, request_id, body.note if body else None)
