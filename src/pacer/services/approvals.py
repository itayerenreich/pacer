"""Approval requests: a counselor asks, the manager approves or rejects.

The state machine itself (PENDING -> APPROVED / REJECTED) lives in the domain
(domain/approval.py). This service adds what the domain object cannot know:
who is asking, whether they may ask, and what happens when a request is approved.
"""

import logging

from pacer.domain import ApprovalKind, ApprovalRequest, ApprovalStatus, StaffMember, TraineeStatus
from pacer.repositories.interfaces import ApprovalRequestRepo, GroupRepo, TraineeRepo

from .errors import RuleViolationError, found
from .permissions import is_manager, leads_group, require

log = logging.getLogger(__name__)


class ApprovalService:
    def __init__(self, approvals: ApprovalRequestRepo, trainees: TraineeRepo, groups: GroupRepo) -> None:
        self._approvals = approvals
        self._trainees = trainees
        self._groups = groups

    def request(
        self,
        actor: StaffMember,
        kind: ApprovalKind,
        trainee_id: int,
        reason: str,
        requested_status: TraineeStatus | None = None,
    ) -> ApprovalRequest:
        if is_manager(actor):
            raise RuleViolationError("The manager does not need approval; act directly")
        trainee = found(self._trainees.get(trainee_id), "Trainee", trainee_id)
        group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
        require(leads_group(actor, group), "ask for approvals about a trainee outside your groups")

        same = {"kind": kind, "trainee_id": trainee_id, "requested_by_id": actor.id}
        if self._approvals.exists_with_status(**same, status=ApprovalStatus.PENDING):
            raise RuleViolationError("You already have a pending request of this kind for this trainee")
        if kind == ApprovalKind.VIEW_PERSONAL_FILE and self._approvals.exists_with_status(
            **same, status=ApprovalStatus.APPROVED
        ):
            raise RuleViolationError("You already have access to this personal file")
        if kind == ApprovalKind.STATUS_CHANGE and requested_status == trainee.status:
            raise RuleViolationError(f"Trainee {trainee_id} is already {trainee.status}")

        request = self._approvals.add(
            ApprovalRequest(
                kind=kind,
                trainee_id=trainee_id,
                requested_by_id=actor.id,
                reason=reason,
                requested_status=requested_status,
            )
        )
        log.info(
            "Approval request %s (%s) for trainee %s by staff %s", request.id, kind, trainee_id, actor.id
        )
        return request

    def approve(self, actor: StaffMember, request_id: int, note: str | None = None) -> ApprovalRequest:
        """Approve, and carry out what was approved, in the same unit of work.

        For a STATUS_CHANGE the trainee's status changes here, together with the
        request. If anything fails, the caller's unit of work rolls back both, so a
        request can never be APPROVED while its change was not made.
        """
        require(is_manager(actor), "approve requests")
        request = found(self._approvals.get(request_id), "Approval request", request_id)
        approved = request.approve(actor.id, note)  # InvalidTransitionError unless PENDING
        self._approvals.update(approved)
        log.info("Approval request %s (%s) approved by staff %s", request_id, approved.kind, actor.id)

        if approved.kind == ApprovalKind.STATUS_CHANGE:
            trainee = found(self._trainees.get(approved.trainee_id), "Trainee", approved.trainee_id)
            self._trainees.update(trainee.with_status(approved.requested_status))
            log.info(
                "Trainee %s: status %s -> %s (request %s)",
                trainee.id,
                trainee.status,
                approved.requested_status,
                request_id,
            )
        return approved

    def reject(self, actor: StaffMember, request_id: int, note: str | None = None) -> ApprovalRequest:
        require(is_manager(actor), "reject requests")
        request = found(self._approvals.get(request_id), "Approval request", request_id)
        rejected = self._approvals.update(request.reject(actor.id, note))
        log.info("Approval request %s (%s) rejected by staff %s", request_id, rejected.kind, actor.id)
        return rejected

    def list_pending(self, actor: StaffMember) -> list[ApprovalRequest]:
        """The manager sees every pending request; a counselor sees only their own."""
        return self._approvals.list_pending(None if is_manager(actor) else actor.id)
