"""Trainees: enrolling, viewing, the sensitive personal file, and status changes."""

from pacer.domain import (
    ApprovalKind,
    ApprovalStatus,
    PersonalFile,
    StaffMember,
    Trainee,
    TraineeStatus,
)
from pacer.repositories.interfaces import ApprovalRequestRepo, GroupRepo, PersonalFileRepo, TraineeRepo

from .errors import RuleViolationError, found
from .permissions import can_view_trainee, is_manager, leads_group, require


class TraineeService:
    def __init__(
        self,
        trainees: TraineeRepo,
        groups: GroupRepo,
        personal_files: PersonalFileRepo,
        approvals: ApprovalRequestRepo,
    ) -> None:
        self._trainees = trainees
        self._groups = groups
        self._personal_files = personal_files
        self._approvals = approvals

    def enroll(self, actor: StaffMember, trainee: Trainee) -> Trainee:
        """Add a trainee to a group. Rules that need more than the Trainee object itself:
        the group is in the same cohort, and the primary counselor leads that group.
        """
        require(is_manager(actor), "enroll trainees")
        group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
        if group.cohort_id != trainee.cohort_id:
            raise RuleViolationError(
                f"Group {group.id} belongs to cohort {group.cohort_id}, not {trainee.cohort_id}"
            )
        if trainee.primary_counselor_id not in group.counselor_ids:
            raise RuleViolationError(
                f"Counselor {trainee.primary_counselor_id} does not lead group {group.id}"
            )
        return self._trainees.add(trainee)

    def get(self, actor: StaffMember, trainee_id: int) -> Trainee:
        trainee = found(self._trainees.get(trainee_id), "Trainee", trainee_id)
        group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
        require(can_view_trainee(actor, group), "view this trainee")
        return trainee

    def get_personal_file(self, actor: StaffMember, trainee_id: int) -> PersonalFile:
        """Manager: always. Counselor: only one who leads the group AND has an approved request."""
        trainee = found(self._trainees.get(trainee_id), "Trainee", trainee_id)
        if not is_manager(actor):
            group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
            approved = self._approvals.exists_with_status(
                kind=ApprovalKind.VIEW_PERSONAL_FILE,
                trainee_id=trainee_id,
                requested_by_id=actor.id,
                status=ApprovalStatus.APPROVED,
            )
            require(
                leads_group(actor, group) and approved, "view this personal file without manager approval"
            )
        return found(self._personal_files.get_by_trainee(trainee_id), "Personal file of trainee", trainee_id)

    def change_status(self, actor: StaffMember, trainee_id: int, status: TraineeStatus) -> Trainee:
        """Managers change status directly. Counselors ask through ApprovalService.request()."""
        require(is_manager(actor), "change a trainee's status (ask the manager for approval)")
        trainee = found(self._trainees.get(trainee_id), "Trainee", trainee_id)
        return self._trainees.update(trainee.with_status(status))
