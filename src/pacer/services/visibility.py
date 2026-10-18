"""Who may not SEE a record gets 404, as if it did not exist (decision 2026-10-18).

With 403 a counselor from another group could probe ids and learn which trainees, talks or
notes exist ("403: there is one, you just may not see it"). So:

    may not see it                     -> 404, the same message as a record that does not exist
    may see it, but not do this action -> 403 (e.g. the second counselor of the group)
    manager-only actions               -> 403, checked BEFORE the lookup, so it reveals nothing

Order matters: visibility is checked before anything that depends on the record, such as
"the file is closed" (409), which would otherwise reveal the trainee's status.
"""

from pacer.domain import StaffMember, Trainee
from pacer.repositories.interfaces import GroupRepo, TraineeRepo

from .errors import NotFoundError, found
from .permissions import can_view_trainee


def hidden(what: str, object_id: int) -> NotFoundError:
    """The error for a record this actor may not see: identical to one that does not exist."""
    return NotFoundError(f"{what} {object_id} does not exist")


def visible_trainee(
    actor: StaffMember,
    trainees: TraineeRepo,
    groups: GroupRepo,
    trainee_id: int,
    *,
    what: str = "Trainee",
    object_id: int | None = None,
) -> Trainee:
    """The trainee, if this actor may see them. Otherwise 404 about what was asked for:
    the trainee itself, or the goal / talk / note of theirs (what, object_id)."""
    trainee = found(trainees.get(trainee_id), "Trainee", trainee_id)
    group = found(groups.get(trainee.group_id), "Group", trainee.group_id)
    if not can_view_trainee(actor, group):
        raise hidden(what, trainee_id if object_id is None else object_id)
    return trainee
