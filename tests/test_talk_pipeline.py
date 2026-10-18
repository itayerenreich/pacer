"""The personal talk pipeline, on the demo academy in memory, with fake models.

מאיה גולן is trainee 22; her counselor is איתי ביטון (5); her goals are 62-65 (see demo/talks.py).
"""

import json

import pytest
from fake_llm import FakeLLMClient, fake_router
from fakes import FakeUnitOfWork, empty_repos

from pacer.ai.anonymization import AnonymizationError, Anonymizer, load_roster
from pacer.ai.issues import IssueCode
from pacer.ai.structured import StructuredOutputError
from pacer.ai.talk_pipeline import (
    TalkExtraction,
    TalkPipeline,
    is_sensitive,
    remove_sensitive_sentences,
    talk_service,
    validate_talk,
)
from pacer.demo import build_academy
from pacer.demo.talks import MAYA
from pacer.domain import Goal, GoalStatus, TalkProposalKind, TalkStatus
from pacer.services import PermissionDeniedError, RuleViolationError

MAYA_ANSWER = {
    "summary": "[TRAINEE_22] רצה 3 ק״מ ברצף פעמיים השבוע. ביומן היא כותבת פעם בשבוע, וסוכם לנסות "
    "לכתוב בבוקר. [TRAINEE_22] מתלבטת אם להוביל את היום הקבוצתי עם [NAME_1].",
    "progress": [{"goal_id": 63, "progress": 60, "reason": "רצה 3 ק״מ ברצף, פעמיים השבוע"}],
    "notes": [
        {"kind": "worked", "text": "יעד קטן ומדיד עוזר ל[TRAINEE_22] להתמיד"},
        {"kind": "did_not_work", "text": "תזכורות בקבוצת הווטסאפ לא עובדות, היא מתעלמת מהן"},
        {"kind": "follow_up", "text": "לחזור אליה בשבוע הבא לגבי הובלת היום הקבוצתי עם [NAME_1]"},
    ],
}


def reply(**changes) -> str:
    return json.dumps({**MAYA_ANSWER, **changes}, ensure_ascii=False)


@pytest.fixture
def repos():
    repos = empty_repos()
    build_academy(repos)
    return repos


@pytest.fixture
def uow(repos) -> FakeUnitOfWork:
    return FakeUnitOfWork(repos)


@pytest.fixture
def itai(repos):
    return repos.staff.get(5)


@pytest.fixture
def cloud() -> FakeLLMClient:
    return FakeLLMClient(answers=[reply()], runs_locally=False)


def record(uow, actor, *, trainee_id: int = MAYA.trainee_id, notes: str = MAYA.notes) -> int:
    with uow() as repos:
        return talk_service(repos).record_talk(actor, trainee_id, held_on=MAYA.held_on, notes=notes).id


def summarize(uow, cloud, actor, talk_id):
    return TalkPipeline(uow, fake_router(cloud)).summarize(actor, talk_id)


# --- the happy path --------------------------------------------------------------------------------


def test_notes_become_a_summary_and_proposals_for_the_counselor(uow, repos, itai, cloud):
    result = summarize(uow, cloud, itai, record(uow, itai))

    assert result.talk.status is TalkStatus.SUMMARIZED
    progress, *notes = result.proposals
    assert (progress.kind, progress.goal_id, progress.progress) == (TalkProposalKind.PROGRESS, 63, 60)
    assert [n.note_kind for n in notes] == ["worked", "did_not_work", "follow_up"]
    assert notes[0].text == "יעד קטן ומדיד עוזר למאיה גולן להתמיד"  # token -> name, here
    assert notes[2].text.endswith("עם רותם")  # an ambiguous name goes back to the word that was said
    assert "מאיה גולן" in result.talk.summary and "[TRAINEE_" not in result.talk.summary
    assert result.issues == [] and result.prompt_ref == "talk_summary@v2"
    assert repos.goals.get(63).progress == 25  # nothing applied: proposals only


def test_the_model_gets_the_pinned_prompt_the_schema_and_the_route_budget(uow, itai, cloud):
    summarize(uow, cloud, itai, record(uow, itai))
    [call] = cloud.calls
    assert call.prompt.ref == "talk_summary@v2"
    assert call.response_schema == TalkExtraction.model_json_schema()
    assert call.max_tokens == 4000  # ROUTES[SUMMARIZE_TALK]: room for long Hebrew answers


def test_the_message_has_the_goals_by_id_and_who_the_talk_was_with(uow, itai, cloud):
    summarize(uow, cloud, itai, record(uow, itai))
    message = cloud.calls[0].user_message
    assert "Trainee: [TRAINEE_22]\nCounselor: [STAFF_5]" in message
    assert "- goal_id 63: לרוץ 5 ק״מ ברצף (category: physical, progress: 25%, status: in_progress)" in message


# --- privacy ------------------------------------------------------------------------------------------


def test_the_cloud_model_never_sees_a_known_name(uow, repos, itai, cloud):
    talk_id = record(uow, itai)
    summarize(uow, cloud, itai, talk_id)

    message = cloud.calls[0].user_message
    assert not Anonymizer(load_roster(repos.staff, repos.trainees)).contains_known_name(message)
    assert repos.talks.get(talk_id).anonymized_notes == message.split("\n\n---")[0]  # stored as sent


def test_a_name_inside_a_goal_title_is_anonymized_too(uow, repos, itai, cloud):
    repos.goals.add(Goal(trainee_id=22, title="לעזור לאדל אזולאי בשיעורי הבית", category="social"))
    summarize(uow, cloud, itai, record(uow, itai))
    message = cloud.calls[0].user_message
    assert "לעזור ל[TRAINEE_16] בשיעורי הבית" in message and "אדל" not in message


def test_in_a_talk_with_ziv_rozen_ziv_is_him(uow, repos):
    """Two trainees are named זיו (6 and 24). In a talk with זיו רוזן (6), "זיו" alone is him."""
    omer = repos.staff.get(3)  # זיו רוזן's primary counselor
    cloud = FakeLLMClient(answers=[reply(progress=[], notes=[])], runs_locally=False)
    summarize(uow, cloud, omer, record(uow, omer, trainee_id=6, notes="זיו סיפר שהשבוע היה לו קשה לקום בבוקר."))
    assert "[TRAINEE_6] סיפר" in cloud.calls[0].user_message


def test_a_name_in_the_final_message_stops_the_call(uow, itai, cloud, monkeypatch):
    import pacer.ai.talk_pipeline as module

    monkeypatch.setattr(module.Anonymizer, "anonymize", lambda self, text, **kw: _Leaky(text))
    with pytest.raises(AnonymizationError):
        summarize(uow, cloud, itai, record(uow, itai))
    assert cloud.calls == []


class _Leaky:
    """What a broken anonymizer would return: the text as it was, names included."""

    def __init__(self, text: str) -> None:
        self.text, self.people, self.unresolved = text, {}, {}


# --- transactions and failures ---------------------------------------------------------------------------


def test_no_transaction_is_open_while_the_model_works(uow, itai):
    seen: list[int] = []

    def model(message: str) -> str:
        seen.append(uow.open)
        return reply()

    summarize(uow, FakeLLMClient(answers=model, runs_locally=False), itai, record(uow, itai))
    assert seen == [0]


def test_an_answer_that_arrives_after_another_summary_was_saved_is_discarded(uow, repos, itai):
    talk_id = record(uow, itai)

    def model(message: str) -> str:
        with uow() as other:
            talk = other.talks.get(talk_id)
            other.talks.update(talk.advance(TalkStatus.SUMMARIZED, summary="אחר", anonymized_notes="אחר"))
        return reply()

    with pytest.raises(RuleViolationError, match="another request"):
        summarize(uow, FakeLLMClient(answers=model, runs_locally=False), itai, talk_id)
    assert repos.talk_proposals.list_by_talk(talk_id) == []


def test_an_invalid_answer_twice_leaves_the_talk_recorded(uow, repos, itai):
    talk_id = record(uow, itai)
    with pytest.raises(StructuredOutputError):
        summarize(uow, FakeLLMClient(answers=['{"summary": "קצר"}'], runs_locally=False), itai, talk_id)
    assert repos.talks.get(talk_id).status is TalkStatus.RECORDED
    assert summarize(uow, FakeLLMClient(answers=[reply()], runs_locally=False), itai, talk_id).proposals


def test_a_talk_is_summarized_once(uow, itai, cloud):
    talk_id = record(uow, itai)
    summarize(uow, cloud, itai, talk_id)
    with pytest.raises(RuleViolationError, match="summarized only when recorded"):
        summarize(uow, cloud, itai, talk_id)
    assert len(cloud.calls) == 1


def test_only_the_primary_counselor_or_the_manager_runs_it(uow, repos, itai, cloud):
    talk_id = record(uow, itai)
    with pytest.raises(PermissionDeniedError):
        summarize(uow, cloud, repos.staff.get(4), talk_id)  # שירה leads the group, not her counselor
    assert cloud.calls == []
    assert summarize(uow, cloud, repos.staff.get(1), talk_id).talk.status is TalkStatus.SUMMARIZED


def test_approving_the_proposal_moves_the_goal(uow, repos, itai, cloud):
    result = summarize(uow, cloud, itai, record(uow, itai))
    with uow() as r:
        talk_service(r).approve_proposal(itai, result.proposals[0].id)
    assert repos.goals.get(63).progress == 60
    assert repos.goal_updates.list_by_goal(63)[-1].note == "רצה 3 ק״מ ברצף, פעמיים השבוע"


# --- content validation ----------------------------------------------------------------------------------


@pytest.fixture
def source(repos):
    return Anonymizer(load_roster(repos.staff, repos.trainees)).anonymize(MAYA.notes)


@pytest.fixture
def goals(repos) -> list[Goal]:
    return repos.goals.list_by_trainee(22)


def validate(source, goals, **changes):
    return validate_talk(TalkExtraction.model_validate_json(reply(**changes)), source, goals=goals)


def codes(result):
    return [(i.code, i.section, i.item) for i in result.issues]


def test_a_goal_that_is_not_hers_is_dropped(source, goals):
    result = validate(source, goals, progress=[{"goal_id": 13, "progress": 70, "reason": "יואב התקדם"}], notes=[])
    assert result.drafts == []
    assert codes(result) == [(IssueCode.GOAL_NOT_FOUND, "progress", 1)]


def test_a_dropped_goal_takes_no_progress(source, goals):
    goals = [g.model_copy(update={"status": GoalStatus.DROPPED}) if g.id == 63 else g for g in goals]
    result = validate(source, goals, notes=[])
    assert codes(result) == [(IssueCode.GOAL_CLOSED, "progress", 1)]


def test_the_same_progress_again_is_no_change(source, goals):
    result = validate(source, goals, progress=[{"goal_id": 63, "progress": 25, "reason": "אותו דבר"}], notes=[])
    assert codes(result) == [(IssueCode.NO_CHANGE, "progress", 1)]


def test_one_update_per_goal(source, goals):
    twice = [{"goal_id": 63, "progress": 60, "reason": "הסבר ראשון"}, {"goal_id": 63, "progress": 70, "reason": "הסבר שני"}]
    result = validate(source, goals, progress=twice, notes=[])
    assert [d.progress for d in result.drafts] == [60]
    assert codes(result) == [(IssueCode.DUPLICATE, "progress", 2)]


def test_progress_can_go_down_after_a_setback(source, goals):
    result = validate(source, goals, progress=[{"goal_id": 63, "progress": 10, "reason": "הפסיקה לרוץ השבוע"}], notes=[])
    assert result.drafts[0].progress == 10 and result.issues == []


def test_a_note_about_health_never_becomes_memory(source, goals):
    notes = [
        {"kind": "observation", "text": "יש לה כאבי ראש והיא הולכת לרופא"},
        {"kind": "worked", "text": "יעד קטן ומדיד עוזר לה"},
    ]
    result = validate(source, goals, progress=[], notes=notes)
    assert [d.text for d in result.drafts] == ["יעד קטן ומדיד עוזר לה"]
    assert codes(result) == [(IssueCode.SENSITIVE_TEXT, "notes", 1)]
    assert "ראש" not in result.issues[0].detail and "רופא" not in result.issues[0].detail


def test_the_same_note_twice_is_kept_once(source, goals):
    same = {"kind": "worked", "text": "יעד קטן ומדיד עוזר לה"}
    result = validate(source, goals, progress=[], notes=[same, same])
    assert len(result.drafts) == 1 and codes(result) == [(IssueCode.DUPLICATE, "notes", 2)]


def test_an_invented_token_is_reported_where_it_is(source, goals):
    result = validate(source, goals, progress=[], notes=[{"kind": "observation", "text": "עבדה יפה עם [TRAINEE_99]"}])
    assert codes(result) == [(IssueCode.UNKNOWN_TOKEN, "notes", 1)]
    assert result.drafts[0].text == "עבדה יפה עם [TRAINEE_99]"


@pytest.mark.parametrize(
    "text",
    [
        "הלכה לרופא",
        "וכאבי ראש",
        "יש לו ADHD",
        "מאובחן עם הפרעת קשב",
        "אובחנה השנה",
        "מקבלת תרופות",
        "טיפול תרופתי",
        "בגלל הפציעה",
        "חרדה לפני מבחנים",
    ],
)
def test_sensitive_words_are_found_with_prefixes(text):
    assert is_sensitive(text)


@pytest.mark.parametrize(
    "text", ["משחקת כדורגל", "חולצה לבנה", "דיברה כאבא לקבוצה", "יעד קטן ומדיד", "ריפאה את הקשר עם החברים"]
)
def test_everyday_words_are_not_sensitive(text):
    assert not is_sensitive(text)


def test_a_progress_reason_about_health_is_not_kept_either(source, goals):
    """An approved reason becomes the GoalUpdate's note, forever: the same rule as notes."""
    hurt = [{"goal_id": 63, "progress": 10, "reason": "נפצעה בברך ולא רצה השבוע"}]
    result = validate(source, goals, progress=hurt, notes=[])
    assert result.drafts == []
    assert codes(result) == [(IssueCode.SENSITIVE_TEXT, "progress", 1)]
    assert "ברך" not in result.issues[0].detail


# --- sensitive sentences never leave the computer (decision 2026-10-18) ------------------------------


def test_the_sentence_about_the_doctor_is_removed_before_sending(uow, repos, itai, cloud):
    talk_id = record(uow, itai)
    result = summarize(uow, cloud, itai, talk_id)
    sent = cloud.calls[-1].user_message
    assert "רופא" not in sent and "כאבי" not in sent
    assert "3 ק\"מ ברצף" in sent  # the rest of the notes is all there
    assert result.removed_sentences == 1
    assert sent.startswith(repos.talks.get(talk_id).anonymized_notes)  # stored = sent (+ the schema)
    assert "לרופא" in repos.talks.get(talk_id).notes  # the counselor's own notes are untouched


def test_removing_sentences_keeps_the_others_and_the_lines():
    notes = "רצה 3 ק\"מ. יש לה כאבים בברך! היא מרוצה.\nאובחנה עם משהו\nהיומן התקדם"
    kept, removed = remove_sensitive_sentences(notes)
    assert removed == 2
    assert kept == "רצה 3 ק\"מ. היא מרוצה.\n\nהיומן התקדם"


def test_notes_without_sensitive_sentences_are_unchanged():
    notes = "רצה 3 ק\"מ ברצף. היומן התקדם.\nסיכמנו להיפגש שוב."
    assert remove_sensitive_sentences(notes) == (notes, 0)


@pytest.mark.parametrize(
    "text",
    [
        "חשבה לפגוע בעצמה",
        "מחשבות אובדניות",
        "קושי נפשי בתקופה האחרונה",
        "יש לה מחלת נפש במשפחה",
        "מתחיל טיפול רגשי",
        'קבע תור לד"ר כהן',
        "רופאת המשפחה אמרה",
        "לוקחת כדורים",
        "יש לו OCD",
        "חולת סוכרת",
        "הלך לפסיכולוגים",
        "שניהם מאובחנים",
    ],
)
def test_the_review_cases_are_sensitive_too(text):
    """Review 2026-10-18: each of these slipped through the first word list."""
    assert is_sensitive(text)


def test_a_sentence_that_only_continues_a_removed_one_goes_with_it():
    kept, removed = remove_sensitive_sentences("יש לו ADHD. לכן הוא לוקח משהו כל בוקר. היומן התקדם.")
    assert kept == " היומן התקדם."
    assert removed == 2


def test_a_list_under_a_removed_heading_goes_with_it():
    notes = "בקשר לריטלין:\n- מקבל כל בוקר\n- הצוות יודע\nהיומן התקדם\n"
    kept, removed = remove_sensitive_sentences(notes)
    assert "בוקר" not in kept and "הצוות" not in kept and "היומן התקדם" in kept
    assert removed == 3


def test_a_goal_title_about_health_is_withheld(repos):
    from pacer.ai.talk_pipeline import WITHHELD_TITLE, prepare_talk_message

    trainee, counselor = repos.trainees.get(MAYA.trainee_id), repos.staff.get(MAYA.counselor_id)
    goals = [*repos.goals.list_by_trainee(trainee.id), Goal(id=999, trainee_id=trainee.id, title="להגיע לכל פגישה עם הפסיכולוגית", category="personal")]
    safe, removed = prepare_talk_message(load_roster(repos.staff, repos.trainees), trainee, counselor, MAYA.held_on, goals, "שיחה קצרה.")
    assert "פסיכולוג" not in safe.text and f"goal_id 999: {WITHHELD_TITLE}" in safe.text
    assert removed == 1
