"""Content validation of the model's answer: what Pydantic cannot know, checked against the world.

Every rule has a test, and every test checks two things: what was corrected, and that the issue
says so (a correction the manager cannot see would be a silent guess).
"""

from datetime import date

import pytest

from pacer.ai.anonymization import Anonymizer, Person, PersonKind, Roster
from pacer.ai.issues import IssueCode
from pacer.ai.meeting_pipeline import (
    DUE_DATE_HORIZON,
    ExtractedTask,
    MeetingExtraction,
    validate_extraction,
)
from pacer.domain.meeting import DESCRIPTION_MAX_LENGTH

S, T = PersonKind.STAFF, PersonKind.TRAINEE
ROSTER = Roster(
    [
        Person(S, 1, "רונית", "שמעוני"),
        Person(S, 2, "נועה", "לוי"),
        Person(S, 3, "עומר", "כהן"),
        Person(S, 4, "שירה", "אברהם"),
        Person(T, 16, "אדל", "אזולאי"),
        Person(T, 24, "זיו", "וקנין"),
        Person(T, 6, "זיו", "רוזן"),  # two trainees named זיו: "זיו" alone is ambiguous
    ]
)
HELD_ON = date(2026, 10, 12)
PARTICIPANTS = [1, 2, 3]  # שירה (4) did not attend
TRANSCRIPT = "רונית: נועה, תתאמי. עומר ידבר עם אדל. זיו ביקש לצאת. שירה לא כאן."
SOURCE = Anonymizer(ROSTER).anonymize(TRANSCRIPT, known=[ROSTER.find(S, i) for i in PARTICIPANTS])


def task(description: str = "לתאם הסעה לטיול השנתי", owner: str | None = "[STAFF_2]", due: date | None = None):
    return ExtractedTask(description=description, owner=owner, due_date=due)


def validate(*tasks: ExtractedTask, summary: str = "[STAFF_1] פתחה את הישיבה וסיכמה משימות."):
    extraction = MeetingExtraction(summary=summary, tasks=list(tasks))
    return validate_extraction(extraction, SOURCE, held_on=HELD_ON, participant_ids=PARTICIPANTS)


def codes(result) -> list[tuple[IssueCode, int | None]]:
    return [(issue.code, issue.item) for issue in result.issues]


def test_a_clean_answer_passes_unchanged_with_names_restored():
    result = validate(task("לשוחח עם [TRAINEE_16] על הריצות", owner="[STAFF_3]", due=date(2026, 10, 15)))
    [draft] = result.drafts
    assert draft.description == "לשוחח עם אדל אזולאי על הריצות"
    assert (draft.owner_id, draft.due_date) == (3, date(2026, 10, 15))
    assert result.summary == "רונית שמעוני פתחה את הישיבה וסיכמה משימות."
    assert result.issues == []


# --- owners -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("owner", "code"),
    [
        ("[STAFF_4]", IssueCode.OWNER_NOT_PARTICIPANT),  # שירה was not there
        ("[STAFF_9]", IssueCode.OWNER_NOT_PARTICIPANT),  # invented: no such participant
        ("[TRAINEE_16]", IssueCode.OWNER_NOT_STAFF),
        ("[NAME_1]", IssueCode.OWNER_AMBIGUOUS),
        ("נועה", IssueCode.OWNER_INVALID),  # not a token
        ("STAFF_2", IssueCode.OWNER_INVALID),
        ("[STAFF_2] ו[STAFF_3]", IssueCode.OWNER_INVALID),  # two owners in one field
    ],
)
def test_an_owner_who_cannot_own_the_task_is_cleared_and_reported(owner, code):
    result = validate(task(owner=owner))
    assert result.drafts[0].owner_id is None  # the manager chooses, before approving
    assert codes(result) == [(code, 1)]


def test_no_owner_is_fine_and_not_an_issue():
    result = validate(task(owner=None))
    assert result.drafts[0].owner_id is None
    assert result.issues == []


def test_whitespace_around_a_good_owner_token_is_forgiven():
    assert validate(task(owner=" [STAFF_2] ")).drafts[0].owner_id == 2


# --- dates ---------------------------------------------------------------------------------------


def test_a_due_date_on_the_meeting_day_is_fine():
    assert validate(task(due=HELD_ON)).drafts[0].due_date == HELD_ON


def test_a_due_date_before_the_meeting_is_cleared():
    result = validate(task(due=date(2026, 10, 11)))
    assert result.drafts[0].due_date is None
    assert codes(result) == [(IssueCode.DUE_BEFORE_MEETING, 1)]


def test_a_due_date_over_a_year_away_is_a_misread_date():
    ok = validate(task(due=HELD_ON + DUE_DATE_HORIZON))
    too_far = validate(task(due=HELD_ON + DUE_DATE_HORIZON + (date(2026, 1, 2) - date(2026, 1, 1))))
    assert ok.drafts[0].due_date is not None and ok.issues == []
    assert too_far.drafts[0].due_date is None
    assert codes(too_far) == [(IssueCode.DUE_TOO_FAR, 1)]


# --- duplicates ------------------------------------------------------------------------------------


def test_the_same_task_for_the_same_owner_is_listed_once():
    result = validate(task(), task("לתאם   הסעה לטיול השנתי"), task(owner="[STAFF_3]"))
    assert [d.owner_id for d in result.drafts] == [2, 3]  # same words for someone else: a separate task
    assert codes(result) == [(IssueCode.DUPLICATE, 2)]


def test_a_duplicate_after_owner_clearing_is_still_caught():
    """[STAFF_4] is cleared to no owner, and then equals the earlier task without an owner."""
    result = validate(task(owner=None), task(owner="[STAFF_4]"))
    assert len(result.drafts) == 1
    assert codes(result) == [(IssueCode.OWNER_NOT_PARTICIPANT, 2), (IssueCode.DUPLICATE, 2)]


# --- tokens the model made up, and lengths -----------------------------------------------------------


def test_an_invented_token_is_left_visible_and_reported():
    result = validate(task("לדבר עם [TRAINEE_99] על הלו״ז"), summary="[STAFF_1] ו[STAFF_7] דיברו על הטיול.")
    assert result.drafts[0].description == "לדבר עם [TRAINEE_99] על הלו״ז"  # never replaced by a guess
    assert "[STAFF_7]" in result.summary
    assert codes(result) == [(IssueCode.UNKNOWN_TOKEN, None), (IssueCode.UNKNOWN_TOKEN, 1)]
    assert "[STAFF_7]" in result.issues[0].detail


def test_an_ambiguous_name_goes_back_to_the_word_that_was_said():
    result = validate(task("לבדוק עם [NAME_1] את היציאה הביתה", owner="[STAFF_3]"))
    assert result.drafts[0].description == "לבדוק עם זיו את היציאה הביתה"
    assert result.issues == []


def test_a_description_that_grew_too_long_with_names_is_shortened():
    long = "לתאם " + "[STAFF_1] " * 24  # 250 characters as tokens, over 300 once names are restored
    result = validate(task(long.strip()))
    assert len(result.drafts[0].description) == DESCRIPTION_MAX_LENGTH
    assert result.drafts[0].description.endswith("…")
    assert codes(result) == [(IssueCode.DESCRIPTION_SHORTENED, 1)]


def test_issues_never_carry_names():
    result = validate(
        task(owner="[STAFF_4]", due=date(2020, 1, 1)),
        task(owner="[STAFF_4]", due=date(2020, 1, 1)),
        task("לדבר עם [TRAINEE_99] היום", owner="נועה"),
    )
    assert result.issues
    for issue in result.issues:
        for person in ROSTER.people:
            assert person.first_name not in issue.detail and person.last_name not in issue.detail


def test_the_extraction_schema_rejects_extra_fields_and_too_many_tasks():
    with pytest.raises(ValueError):
        MeetingExtraction.model_validate({"summary": "סיכום ארוך מספיק לבדיקה", "tasks": [], "notes": "x"})
    with pytest.raises(ValueError):
        MeetingExtraction(summary="סיכום ארוך מספיק לבדיקה", tasks=[task()] * 31)
