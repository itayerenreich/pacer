"""Fixed sets of values used across the domain."""

from enum import StrEnum


class StaffRole(StrEnum):
    COUNSELOR = "counselor"
    MANAGER = "manager"


class ResponsibilityArea(StrEnum):
    VOLUNTEERING = "volunteering"
    LOGISTICS = "logistics"
    KITCHEN = "kitchen"
    CONTENT = "content"
    ARMY_PREP = "army_prep"


class TraineeStatus(StrEnum):
    ACTIVE = "active"
    ON_LEAVE = "on_leave"
    LEFT = "left"
    GRADUATED = "graduated"


class GoalCategory(StrEnum):
    PERSONAL = "personal"
    SOCIAL = "social"
    VALUES = "values"
    PHYSICAL = "physical"
    LEADERSHIP = "leadership"
    ARMY_PREP = "army_prep"


class GoalStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    DROPPED = "dropped"


class ApprovalKind(StrEnum):
    VIEW_PERSONAL_FILE = "view_personal_file"
    STATUS_CHANGE = "status_change"  # includes sending a trainee home (-> ON_LEAVE)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MeetingStatus(StrEnum):
    """Where a staff meeting is in the AI pipeline (task 6 moves it along)."""

    TRANSCRIBED = "transcribed"  # we have the text of the meeting (typed or pasted by the manager)
    ANONYMIZED = "anonymized"  # names replaced by tokens
    EXTRACTED = "extracted"  # the model proposed a summary and tasks
    PENDING_APPROVAL = "pending_approval"  # proposals validated, waiting for the manager
    APPROVED = "approved"  # every proposal was approved or rejected


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskStatus(StrEnum):
    OPEN = "open"
    DONE = "done"


class TalkStatus(StrEnum):
    """Where a personal talk is in the AI flow (task 7)."""

    RECORDED = "recorded"  # the counselor's notes are saved
    SUMMARIZED = "summarized"  # the model proposed a summary, progress updates and notes
    REVIEWED = "reviewed"  # every proposal was approved or rejected


class TalkProposalKind(StrEnum):
    PROGRESS = "progress"  # a new progress value for one of the trainee's goals
    NOTE = "note"  # something worth remembering about the trainee (long-term memory)


class NoteKind(StrEnum):
    OBSERVATION = "observation"  # what was seen or said
    WORKED = "worked"  # an approach that helped this trainee
    DID_NOT_WORK = "did_not_work"  # an approach that did not
    FOLLOW_UP = "follow_up"  # something open, to come back to
