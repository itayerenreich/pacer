"""What content validation changed in a model's answer, reported to the person who decides.

Every pipeline validates the model's answer against the world, and never fixes something
silently: each correction becomes an Issue that is returned with the proposals (and counted in
the logs). Details hold tokens, ids and dates, never names: issues go to logs and to the API.
"""

from dataclasses import dataclass
from enum import StrEnum


class IssueCode(StrEnum):
    # any pipeline
    UNKNOWN_TOKEN = "unknown_token"  # a token that was not in what the model received
    DUPLICATE = "duplicate"
    # staff meetings (task 6)
    OWNER_INVALID = "owner_invalid"  # not a token at all
    OWNER_AMBIGUOUS = "owner_ambiguous"  # [NAME_n]: the name fits more than one person
    OWNER_NOT_STAFF = "owner_not_staff"  # a trainee cannot own a staff task
    OWNER_UNKNOWN = "owner_unknown"  # a staff token that is not in the text the model received: invented
    OWNER_ABSENT = "owner_absent"  # kept: named in the meeting but not there. The manager should check
    DUE_BEFORE_MEETING = "due_before_meeting"
    DUE_TOO_FAR = "due_too_far"
    DESCRIPTION_SHORTENED = "description_shortened"
    # personal talks (task 7)
    GOAL_NOT_FOUND = "goal_not_found"  # not one of this trainee's goals
    GOAL_CLOSED = "goal_closed"  # dropped: it takes no more progress
    NO_CHANGE = "no_change"  # the proposed progress is the current progress
    SENSITIVE_TEXT = "sensitive_text"  # health or a diagnosis: never stored as a note or a progress reason
    # library answers (stage 4)
    UNKNOWN_PASSAGE = "unknown_passage"  # a cited passage id that was never sent
    UNSUPPORTED_CLAIM = "unsupported_claim"  # a claim with no passage behind it
    NO_CITATIONS = "no_citations"  # an answer that cites nothing
    CONTRADICTORY_COVERAGE = "contradictory_coverage"  # "the library has nothing", with claims that cite it
    # the counselor coach (stage 5)
    NO_ADVICE = "no_advice"  # every step was removed by validation
    # the agent (stage 6)
    TRUNCATED = "truncated"  # the answer hit the token limit and was cut off


@dataclass(frozen=True)
class Issue:
    code: IssueCode
    item: int | None  # 1-based position in its list of the model's answer; None for the summary
    detail: str
    section: str | None = None  # which list, when an answer has several ("progress", "notes")
