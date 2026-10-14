"""Persistence for approval requests."""

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from pacer.domain import ApprovalKind, ApprovalRequest, ApprovalStatus

from ._mapping import row_values
from .tables import ApprovalRequestRow


class ApprovalRequestRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, request: ApprovalRequest) -> ApprovalRequest:
        row = ApprovalRequestRow(**row_values(request))
        self._session.add(row)
        self._session.flush()
        return request.model_copy(update={"id": row.id})

    def get(self, request_id: int) -> ApprovalRequest | None:
        row = self._session.get(ApprovalRequestRow, request_id)
        return None if row is None else _to_domain(row)

    def update(self, request: ApprovalRequest) -> ApprovalRequest:
        row = self._session.get(ApprovalRequestRow, request.id)
        if row is None:
            raise LookupError(f"Approval request {request.id} does not exist")
        for column, value in row_values(request).items():
            setattr(row, column, value)
        self._session.flush()
        return request

    def list_pending(self, requested_by_id: int | None = None) -> list[ApprovalRequest]:
        """All pending requests, oldest first. With requested_by_id: only that counselor's."""
        query = select(ApprovalRequestRow).where(ApprovalRequestRow.status == ApprovalStatus.PENDING.value)
        if requested_by_id is not None:
            query = query.where(ApprovalRequestRow.requested_by_id == requested_by_id)
        rows = self._session.scalars(query.order_by(ApprovalRequestRow.created_at))
        return [_to_domain(row) for row in rows]

    def exists_with_status(
        self, *, kind: ApprovalKind, trainee_id: int, requested_by_id: int, status: ApprovalStatus
    ) -> bool:
        """Is there a request of this kind, by this counselor, for this trainee, in this status?"""
        query = select(
            exists().where(
                ApprovalRequestRow.kind == kind.value,
                ApprovalRequestRow.trainee_id == trainee_id,
                ApprovalRequestRow.requested_by_id == requested_by_id,
                ApprovalRequestRow.status == status.value,
            )
        )
        return bool(self._session.scalar(query))


def _to_domain(row: ApprovalRequestRow) -> ApprovalRequest:
    return ApprovalRequest.model_validate(row, from_attributes=True)
