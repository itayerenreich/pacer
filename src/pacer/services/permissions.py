"""Who may do what: role-based access control (RBAC) as small, pure functions.

The rules, in one place:
    Manager            everything, including personal files and status changes.
    Group counselor    sees the trainees, goals and history of the groups they lead,
                       and may ask the manager for approvals about those trainees.
    Primary counselor  (the one counselor a trainee is assigned to) also edits that
                       trainee's goals and progress.
    Personal file      manager always; a counselor only with an APPROVED request
                       (checked in TraineeService, because it needs the database).

The functions take domain objects and return bool, so they need no database and
are trivial to test. Services call require() to turn a False into an error (403), and
visibility.visible_trainee() when the actor may not even SEE the record (404).
"""

from pacer.domain import Group, StaffMember, StaffRole, Trainee

from .errors import PermissionDeniedError


def is_manager(actor: StaffMember) -> bool:
    return actor.role == StaffRole.MANAGER


def is_primary_counselor(actor: StaffMember, trainee: Trainee) -> bool:
    return actor.id == trainee.primary_counselor_id


def leads_group(actor: StaffMember, group: Group) -> bool:
    return actor.id in group.counselor_ids


def can_view_trainee(actor: StaffMember, group: Group) -> bool:
    """Trainee details, goals and progress history (not the personal file)."""
    return is_manager(actor) or leads_group(actor, group)


def can_edit_goals(actor: StaffMember, trainee: Trainee) -> bool:
    return is_manager(actor) or is_primary_counselor(actor, trainee)


def require(allowed: bool, action: str) -> None:
    """Raise PermissionDeniedError unless allowed. action completes "Not allowed to ..."."""
    if not allowed:
        raise PermissionDeniedError(f"Not allowed to {action}")
