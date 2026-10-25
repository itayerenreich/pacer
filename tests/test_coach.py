"""The counselor coach: the order (permissions, safety, context, library, model, validation), what is
sent and what never is, and how validation decides whether a step is grounded or general."""

import json
from datetime import UTC, datetime

import pytest
from fake_llm import FakeLLMClient, fake_router
from fakes import FakeUnitOfWork, empty_repos
from test_library_answer import CHUNKING, VectorEmbedder, add

from pacer.ai.anonymization import Anonymizer, load_roster
from pacer.ai.coach import (
    NOT_SENT_NOTE,
    PROCEDURE_NOTE,
    Coach,
    CoachAnswer,
    CoachOutcome,
    InvolveRole,
    validate_coach,
)
from pacer.ai.issues import IssueCode
from pacer.ai.safety import REFERRAL
from pacer.domain import (
    Cohort,
    EvidenceLevel,
    Goal,
    GoalUpdate,
    Group,
    NoteKind,
    PersonalFile,
    StaffMember,
    StaffRole,
    Trainee,
    TraineeNote,
)
from pacer.services import NotFoundError

QUESTION = "איך לעזור לנדב להשתתף יותר בשיחות הקבוצה?"


@pytest.fixture
def repos():
    repos = empty_repos()
    repos.staff.add(StaffMember(full_name="רונית שמעוני", role=StaffRole.MANAGER))  # 1
    repos.staff.add(StaffMember(full_name="נועה לוי", role=StaffRole.COUNSELOR))  # 2
    repos.staff.add(StaffMember(full_name="דנה כץ", role=StaffRole.COUNSELOR))  # 3, another group
    cohort = repos.cohorts.add(Cohort(name="מחזור", start_date="2026-09-01", end_date="2027-06-30"))
    ours = repos.groups.add(Group(name="ארז", cohort_id=cohort.key, counselor_ids=[2]))
    repos.groups.add(Group(name="אלון", cohort_id=cohort.key, counselor_ids=[3]))
    nadav = repos.trainees.add(
        Trainee(first_name="נדב", last_name="הררי", cohort_id=cohort.key, group_id=ours.key, primary_counselor_id=2)
    )
    repos.personal_files.add(PersonalFile(trainee_id=nadav.key, intake_answers={"תחביבים": "גיטרה חשמלית"}))
    goal = repos.goals.add(Goal(trainee_id=nadav.key, title="להשתתף באופן פעיל בשיחות הקבוצה", category="social"))
    repos.goal_updates.add(
        GoalUpdate(goal_id=goal.key, author_id=2, progress=30, note="השתתף פעמיים", created_at=datetime(2026, 10, 1, tzinfo=UTC))
    )
    for kind, text in [
        (NoteKind.WORKED, "כשהכין מראש נקודה אחת לשיחה, הוא אמר אותה"),
        (NoteKind.DID_NOT_WORK, "פנייה בשמו מול כל הקבוצה גרמה לו להסתגר"),
    ]:
        repos.notes.add(TraineeNote(trainee_id=nadav.key, kind=kind, text=text, author_id=2))
    add(repos, "משנה תורה, הלכות תלמוד תורה", [("מלמד שאין לו לומדים שואלים, יחלק להם את הלימוד לפי כוחם", [1.0, 0.0])])
    return repos


@pytest.fixture
def uow(repos):
    return FakeUnitOfWork(repos)


def screen(risk: bool = False, kinds: list[str] | None = None) -> str:
    return json.dumps({"risk": risk, "kinds": kinds or [], "reason": "a signal" if risk else "nothing"})


def advice(**changes) -> str:
    answer = {
        "understanding": "[TRAINEE_1] פתוח בשיחות אישיות ושותק בקבוצה. ידוע מה עבד איתו.",
        "questions": [],
        "steps": [
            {"action": "לבקש מ-[TRAINEE_1] להכין מראש נקודה אחת", "example": "תכין משהו אחד למחר?", "passages": ["P1"]},
            {"action": "לא לפנות אליו בשמו מול כולם", "example": None, "passages": []},
        ],
        "involve": [],
    }
    answer.update(changes)
    return json.dumps(answer, ensure_ascii=False)


def coach_with(uow, *answers, embedder=None):
    cloud = FakeLLMClient(answers=list(answers), runs_locally=False)
    coach = Coach(uow, fake_router(cloud), embedder or VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING)
    return coach, cloud


def actor(repos, staff_id: int) -> StaffMember:
    return repos.staff.get(staff_id)


# --- the whole way -------------------------------------------------------------------------------


def test_advice_for_a_trainee_grounded_and_general_steps_names_restored(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    result = coach.ask(actor(repos, 2), QUESTION, trainee_id=1)
    assert result.outcome is CoachOutcome.ADVICE and result.note == PROCEDURE_NOTE
    grounded, general = result.advice.steps
    assert grounded.passages == ["P1"] and grounded.grounded and "נדב" in grounded.action
    assert general.passages == [] and not general.grounded
    assert result.advice.citations[0].cited and result.prompt_ref == "coach_advice@v1"
    assert [call.prompt.ref for call in cloud.calls] == ["risk_screen@v1", "coach_advice@v1"]


def test_what_the_model_reads_the_trainees_record_without_names(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    coach.ask(actor(repos, 2), QUESTION, trainee_id=1)
    sent = cloud.calls[1].user_message
    assert "נדב" not in sent and "הררי" not in sent and "נועה" not in sent
    assert "[TRAINEE_1]" in sent
    assert "[worked]" in sent and "הכין מראש נקודה אחת" in sent  # what worked with him
    assert "30%" in sent and "השתתף פעמיים" in sent  # his goal and its last update
    assert "[P1]" in sent and "לפי כוחם" in sent  # the passage


def test_the_personal_file_never_reaches_any_model(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    coach.ask(actor(repos, 1), QUESTION, trainee_id=1)  # even the manager, who may open it
    assert all("גיטרה" not in call.user_message for call in cloud.calls)


def test_health_sentences_in_the_question_and_in_a_note_never_reach_any_model(uow, repos):
    repos.notes.add(TraineeNote(trainee_id=1, kind=NoteKind.OBSERVATION, text="נפגש עם פסיכולוג פעם בשבוע", author_id=2))
    coach, cloud = coach_with(uow, screen(), advice())
    result = coach.ask(actor(repos, 2), "הוא לוקח ריטלין. " + QUESTION, trainee_id=1)
    assert all("ריטלין" not in c.user_message and "פסיכולוג" not in c.user_message for c in cloud.calls)
    assert result.removed_sentences == 2


def test_retrieval_searches_with_the_anonymized_question(uow, repos):
    embedder = VectorEmbedder([1.0, 0.0], floor=0.4)
    coach, _ = coach_with(uow, screen(), advice(), embedder=embedder)
    coach.ask(actor(repos, 2), QUESTION, trainee_id=1)
    assert embedder.queries and all("נדב" not in q for q in embedder.queries)


def test_no_transaction_is_open_while_a_model_works(uow, repos):
    seen = []

    def answer(message: str) -> str:
        seen.append(uow.open)
        return screen() if len(seen) == 1 else advice()

    cloud = FakeLLMClient(answers=answer, runs_locally=False)
    Coach(uow, fake_router(cloud), VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING).ask(actor(repos, 2), QUESTION, 1)
    assert seen == [0, 0]


def test_a_general_question_needs_no_trainee(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice(understanding="שאלה כללית על קבוצה שמאחרת."))
    result = coach.ask(actor(repos, 3), "איך מתמודדים עם קבוצה שמאחרת לפעילויות בוקר?")
    assert result.outcome is CoachOutcome.ADVICE and result.trainee_id is None
    assert "Trainee: none" in cloud.calls[1].user_message


def test_no_passage_still_gets_advice_marked_general(uow, repos):
    far = VectorEmbedder([0.0, 1.0], floor=0.4)  # nothing in the library is close
    coach, cloud = coach_with(uow, screen(), advice(steps=[{"action": "שיחה אישית קצרה", "example": None, "passages": []}]), embedder=far)
    result = coach.ask(actor(repos, 2), QUESTION, trainee_id=1)
    assert result.outcome is CoachOutcome.ADVICE and result.advice.citations == []
    assert "Passages: none" in cloud.calls[1].user_message


# --- the order: permissions first, then safety ---------------------------------------------------


def test_a_counselor_of_another_group_gets_404_before_any_model(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    with pytest.raises(NotFoundError):
        coach.ask(actor(repos, 3), QUESTION, trainee_id=1)
    assert cloud.calls == []


def test_a_risk_by_phrases_gets_the_procedure_and_no_model_at_all(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    result = coach.ask(actor(repos, 2), "נדב אמר שהוא רוצה למות", trainee_id=1)
    assert result.outcome is CoachOutcome.REFERRAL and result.referral is REFERRAL and result.advice is None
    assert cloud.calls == []


def test_a_risk_by_the_screen_gets_the_procedure_and_no_advice_model(uow, repos):
    coach, cloud = coach_with(uow, screen(True, ["self_harm"]), advice())
    result = coach.ask(actor(repos, 2), "חבר סיפר שנדב עמד על הגג בלילה", trainee_id=1)
    assert result.outcome is CoachOutcome.REFERRAL
    assert [call.prompt.ref for call in cloud.calls] == ["risk_screen@v1"]
    assert "[worked]" not in cloud.calls[0].user_message  # the screen sees the question only


def test_an_approved_note_about_a_fight_does_not_make_every_question_a_referral(uow, repos):
    """Decision 2026-10-25: the gate checks the question, not the trainee's record."""
    repos.notes.add(TraineeNote(trainee_id=1, kind=NoteKind.WORKED, text="אחרי מכות בחדר, הפרדה ליום אחד עזרה", author_id=2))
    coach, _ = coach_with(uow, screen(), advice())
    assert coach.ask(actor(repos, 2), QUESTION, trainee_id=1).outcome is CoachOutcome.ADVICE


def test_a_question_only_about_health_is_not_sent_anywhere(uow, repos):
    coach, cloud = coach_with(uow, screen(), advice())
    result = coach.ask(actor(repos, 2), "הוא התחיל לקחת ריטלין.", trainee_id=1)
    assert result.outcome is CoachOutcome.NOT_SENT and result.note == NOT_SENT_NOTE
    assert cloud.calls == []


def test_an_empty_question_is_refused(uow, repos):
    coach, _ = coach_with(uow)
    with pytest.raises(ValueError):
        coach.ask(actor(repos, 2), "   ", trainee_id=1)


# --- validation ----------------------------------------------------------------------------------


@pytest.fixture
def document(repos):
    anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))
    return anonymizer.anonymize("Question: על נדב הררי")


def validate(raw: str, hits, document):
    return validate_coach(CoachAnswer.model_validate_json(raw), hits, document)


def one_hit(repos):
    from pacer.ai.library import retrieve

    return retrieve(FakeUnitOfWork(repos), VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING, "שאלה")


def test_a_citation_never_sent_is_removed_and_the_step_becomes_general(repos, document):
    raw = advice(steps=[{"action": "צעד ראשון וברור", "example": None, "passages": ["P1", "P7"]},
                        {"action": "צעד שני וברור", "example": None, "passages": ["P9"]}])
    result = validate(raw, one_hit(repos), document)
    assert [s.passages for s in result.steps] == [["P1"], []]
    assert [i.code for i in result.issues] == [IssueCode.UNKNOWN_PASSAGE, IssueCode.UNKNOWN_PASSAGE]


def test_a_step_about_health_is_never_shown(repos, document):
    raw = advice(steps=[{"action": "לבדוק אם יש לו קשב וריכוז", "example": None, "passages": []},
                        {"action": "שיחה אישית קצרה", "example": "נדבר רגע?", "passages": []}])
    result = validate(raw, [], document)
    assert [s.action for s in result.steps] == ["שיחה אישית קצרה"]
    assert result.issues[0].code is IssueCode.SENSITIVE_TEXT and "קשב" not in result.issues[0].detail


def test_understanding_about_health_is_withheld(repos, document):
    result = validate(advice(understanding="נראה שיש לו חרדה חברתית, כדאי להתייחס לזה."), [], document)
    assert "חרדה" not in result.understanding
    assert any(i.code is IssueCode.SENSITIVE_TEXT and i.section == "understanding" for i in result.issues)


def test_no_step_left_is_reported(repos, document):
    raw = advice(steps=[{"action": "לברר אם הוא מאובחן", "example": None, "passages": []}])
    result = validate(raw, [], document)
    assert result.steps == [] and IssueCode.NO_ADVICE in [i.code for i in result.issues]


def test_who_to_involve_once_per_role_and_only_from_the_list(repos, document):
    raw = advice(involve=[{"role": "manager", "why": "החלטה שמעבר למדריך"}, {"role": "manager", "why": "שוב"}])
    result = validate(raw, [], document)
    assert result.involve == [(InvolveRole.MANAGER, "החלטה שמעבר למדריך")]
    assert IssueCode.DUPLICATE in [i.code for i in result.issues]
    with pytest.raises(ValueError):
        CoachAnswer.model_validate_json(advice(involve=[{"role": "police", "why": "x" * 5}]))


def test_an_invented_token_is_reported_and_left_as_written(repos, document):
    result = validate(advice(understanding="[TRAINEE_99] ו-[TRAINEE_1] בקבוצה אחת."), [], document)
    assert "[TRAINEE_99]" in result.understanding and "נדב" in result.understanding
    assert result.issues[0].code is IssueCode.UNKNOWN_TOKEN


def test_evidence_level_reaches_the_citation(repos, document):
    citation = validate(advice(), one_hit(repos), document).citations[0]
    assert citation.evidence == EvidenceLevel.TRADITION.label and citation.cited


# --- the API (task 4) ----------------------------------------------------------------------------


@pytest.fixture
def api(repos):
    from fastapi.testclient import TestClient

    from pacer.api.app import create_app
    from pacer.api.dependencies import get_ai_router, get_db_engine, get_embedder, get_unit_of_work

    cloud = FakeLLMClient(answers=[screen(), advice()], runs_locally=False)
    app = create_app()
    app.dependency_overrides = {
        get_db_engine: lambda: (_ for _ in ()).throw(AssertionError("no database in these tests")),
        get_unit_of_work: lambda: FakeUnitOfWork(repos),
        get_ai_router: lambda: fake_router(cloud),
        get_embedder: lambda: VectorEmbedder([1.0, 0.0], floor=0.4),
    }
    with TestClient(app) as client:
        yield client, cloud


def post(client, staff: int | None, **body):
    headers = {} if staff is None else {"X-Staff-Id": str(staff)}
    return client.post("/coach/ask", json=body, headers=headers)


def test_the_api_gives_advice_with_grounding_and_the_procedure_line(api):
    client, _ = api
    response = post(client, 2, question=QUESTION, trainee_id=1)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "advice" and body["referral"] is None and body["note"] == PROCEDURE_NOTE
    assert [s["grounded"] for s in body["advice"]["steps"]] == [True, False]
    assert body["advice"]["citations"][0]["evidence"] == "מקור ערכי-מסורתי"
    assert body["prompt"] == "coach_advice@v1"


def test_the_api_gives_the_procedure_for_an_exceptional_incident(api):
    client, cloud = api
    body = post(client, 2, question="נדב אמר שהוא רוצה למות", trainee_id=1).json()
    assert body["outcome"] == "referral" and body["advice"] is None
    assert body["referral"]["message"] == REFERRAL.message
    assert body["referral"]["steps"][0]["contact"] == REFERRAL.steps[0].contact
    assert cloud.calls == []


def test_the_api_hides_a_trainee_of_another_group_and_needs_a_real_question(api):
    client, cloud = api
    assert post(client, 3, question=QUESTION, trainee_id=1).status_code == 404
    assert post(client, 2, question=QUESTION, trainee_id=999).status_code == 404
    assert post(client, None, question=QUESTION).status_code == 401
    assert post(client, 2, question="?").status_code == 422
    assert post(client, 2, question=QUESTION, trainee_id=1, extra="x").status_code == 422
    assert cloud.calls == []


def test_the_api_without_a_trainee_is_a_general_question(api):
    client, _ = api
    body = post(client, 3, question="איך מתמודדים עם קבוצה שמאחרת?").json()
    assert body["outcome"] == "advice" and body["trainee_id"] is None


def test_a_model_that_is_down_is_503_and_no_advice(repos):
    from fastapi.testclient import TestClient

    from pacer.ai import LLMUnavailableError
    from pacer.api.app import create_app
    from pacer.api.dependencies import get_ai_router, get_db_engine, get_embedder, get_unit_of_work

    def down(_message: str) -> str:
        raise LLMUnavailableError("down")

    app = create_app()
    app.dependency_overrides = {
        get_db_engine: lambda: None,
        get_unit_of_work: lambda: FakeUnitOfWork(repos),
        get_ai_router: lambda: fake_router(FakeLLMClient(answers=down, runs_locally=False)),
        get_embedder: lambda: VectorEmbedder([1.0, 0.0], floor=0.4),
    }
    with TestClient(app) as client:
        response = post(client, 2, question=QUESTION, trainee_id=1)
    assert response.status_code == 503 and "advice" not in response.json()
