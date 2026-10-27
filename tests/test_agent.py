"""The agent: the loop our code runs (limits, errors, the order of the checks), and above all what the
model may read and what the counselor gets back. The model is a script: each test says which tools it
asks for, and checks what our code did with that."""

import json
from datetime import UTC, date, datetime, time

import pytest
from fake_llm import FakeLLMClient, fake_router, say, use
from fakes import FakeUnitOfWork, empty_repos
from test_library_answer import CHUNKING, VectorEmbedder, add

from pacer.ai.agent import (
    EMPTY_LEFT,
    NOT_SENT_NOTE,
    STOPPED_ANSWER,
    Agent,
    AgentOutcome,
    Turn,
)
from pacer.ai.coach import PROCEDURE_NOTE, Coach
from pacer.ai.issues import IssueCode
from pacer.ai.safety import REFERRAL
from pacer.ai.tools import AgentTools
from pacer.domain import (
    AttendanceRecord,
    AttendanceStatus,
    Cohort,
    Goal,
    Group,
    NoteKind,
    PersonalFile,
    ScheduledSession,
    StaffMember,
    StaffRole,
    Trainee,
    TraineeNote,
)

TODAY = date(2026, 10, 27)  # a Tuesday
NO_RISK = json.dumps({"risk": False, "kinds": [], "reason": "nothing"})


@pytest.fixture
def repos():
    """Noa leads group ארז: Dani Avraham, Nadav Harari, Yael Bar. Dana leads אלון: Yael Cohen.
    Two trainees named יעל: in Noa's eyes only one of them exists."""
    repos = empty_repos()
    repos.staff.add(StaffMember(full_name="רונית שמעוני", role=StaffRole.MANAGER))  # 1
    repos.staff.add(StaffMember(full_name="נועה לוי", role=StaffRole.COUNSELOR))  # 2
    repos.staff.add(StaffMember(full_name="דנה כץ", role=StaffRole.COUNSELOR))  # 3
    cohort = repos.cohorts.add(Cohort(name="מחזור", start_date="2026-09-01", end_date="2027-06-30"))
    ours = repos.groups.add(Group(name="ארז", cohort_id=cohort.key, counselor_ids=[2]))
    theirs = repos.groups.add(Group(name="אלון", cohort_id=cohort.key, counselor_ids=[3]))

    def trainee(first, last, group):
        counselor = 2 if group is ours else 3
        return repos.trainees.add(
            Trainee(
                first_name=first,
                last_name=last,
                cohort_id=cohort.key,
                group_id=group.key,
                primary_counselor_id=counselor,
            )
        )

    dani = trainee("דני", "אברהם", ours)  # 1
    trainee("נדב", "הררי", ours)  # 2
    trainee("יעל", "בר", ours)  # 3
    trainee("יעל", "כהן", theirs)  # 4
    repos.personal_files.add(PersonalFile(trainee_id=dani.key, intake_answers={"תחביבים": "גיטרה חשמלית"}))
    goal = repos.goals.add(Goal(trainee_id=dani.key, title="להגיע בזמן לספורט בוקר", category="personal", progress=20))
    assert goal.key == 1
    repos.notes.add(
        TraineeNote(
            trainee_id=dani.key,
            kind=NoteKind.WORKED,
            text="כשקבענו יחד שעון מעורר הוא הגיע",
            author_id=2,
            created_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
    )
    add(repos, "משנה תורה, הלכות תלמוד תורה", [("מלמד שאין לו לומדים שואלים, יחלק להם את הלימוד לפי כוחם", [1.0, 0.0])])
    return repos


@pytest.fixture
def uow(repos):
    return FakeUnitOfWork(repos)


def build(uow, *replies, answers=(NO_RISK,), **limits):
    cloud = FakeLLMClient(answers=list(answers), runs_locally=False, replies=list(replies))
    cloud.before_chat = lambda: _no_transaction(uow)
    router = fake_router(cloud)
    coach = Coach(uow, router, VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING, today=lambda: TODAY)
    tools = AgentTools(uow, coach, VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING).registry()
    return Agent(uow, router, tools, today=lambda: TODAY, **limits), cloud


def _no_transaction(uow):
    assert uow.open == 0, "a model was called while a transaction was open"


def staff(repos, staff_id):
    return repos.staff.get(staff_id)


def coach_answer() -> str:
    return json.dumps(
        {
            "risk": False,
            "understanding": "[TRAINEE_1] מאחר לספורט בוקר. שעון מעורר משותף עבד.",
            "questions": [],
            "steps": [
                {"action": "לקבוע עם [TRAINEE_1] שוב שעון מעורר", "example": None, "passages": []},
                {"action": "לחלק את היעד לפי כוחו: פעמיים בשבוע קודם", "example": None, "passages": ["P1"]},
            ],
            "involve": [],
        },
        ensure_ascii=False,
    )


# --- the whole way: the definition of done of stage 6 --------------------------------------------


def test_how_to_deal_with_dani_finds_him_reads_the_record_asks_the_coach_and_answers(uow, repos):
    agent, cloud = build(
        uow,
        use(("find_trainees", {"name": "[TRAINEE_1]"})),
        use(
            ("attendance", {"trainee_id": 1}),
            ("ask_coach", {"trainee_id": 1, "question": "איך להתמודד עם [TRAINEE_1] שלא עומד ביעדים?"}),
        ),
        say("כדאי לקבוע עם [TRAINEE_1] שוב שעון מעורר, ולהתחיל מפעמיים בשבוע."),
        answers=[NO_RISK, NO_RISK, coach_answer()],  # the agent's screen, the coach's screen, the coach
    )
    result = agent.chat(staff(repos, 2), "איך להתמודד עם דני שלא עומד ביעדים?")

    assert result.outcome is AgentOutcome.ANSWER
    assert result.answer == "כדאי לקבוע עם דני אברהם שוב שעון מעורר, ולהתחיל מפעמיים בשבוע."
    assert [s.tool for s in result.steps] == ["find_trainees", "attendance", "ask_coach"]
    assert all(s.ok for s in result.steps)
    assert result.steps[2].summary == "2 steps, 1 grounded"
    assert [s.citation for s in result.sources] == ['משנה תורה, הלכות תלמוד תורה (רמב"ם)']
    assert result.note == PROCEDURE_NOTE and result.rounds == 3 and result.prompt_ref == "agent@v1"


def test_the_model_never_reads_a_name_the_personal_file_or_anything_it_did_not_ask_for(uow, repos):
    agent, cloud = build(
        uow,
        use(("trainee_record", {"trainee_id": 1})),
        say("סיימתי"),
    )
    agent.chat(staff(repos, 2), "מה המצב של דני?")
    read = "\n".join(chat.everything() for chat in cloud.chats)
    for name in ("דני", "אברהם", "נועה", "לוי", "רונית"):
        assert name not in read
    assert "גיטרה" not in read  # the personal file is not a tool
    assert "[TRAINEE_1]" in read and "[STAFF_2]" in read  # the asker, as a token
    assert "שעון מעורר" in cloud.chats[1].tool_results()[0]  # what worked with him: the record arrived


def test_the_first_message_says_what_day_it_is_and_who_asks(uow, repos):
    agent, cloud = build(uow, say("בסדר"))
    agent.chat(staff(repos, 1), "מה נשמע?")
    first = cloud.chats[0].messages[-1]["content"]
    assert first.startswith("(Today is Tuesday 2026-10-27. Asked by [STAFF_1], the manager.)")


def test_tool_results_go_back_with_the_id_of_the_call_they_answer(uow, repos):
    agent, cloud = build(uow, use("week_schedule", ("find_trainees", {})), say('זה הלו"ז'))
    agent.chat(staff(repos, 2), "מה יש השבוע?")
    messages = cloud.chats[1].messages
    asked = [c["id"] for c in messages[-3]["tool_calls"]]
    answered = [m["tool_call_id"] for m in messages[-2:]]
    assert asked == answered == ["call_1_0", "call_1_1"]


def test_every_tool_is_offered_with_a_closed_schema(uow, repos):
    agent, cloud = build(uow, say("בסדר"))
    agent.chat(staff(repos, 2), "שלום")
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in cloud.chats[0].tools}
    assert set(tools) == {
        "find_trainees",
        "trainee_record",
        "attendance",
        "week_schedule",
        "search_library",
        "ask_coach",
        "report_exceptional_incident",
    }
    assert all(schema["additionalProperties"] is False for schema in tools.values())
    assert "actor" not in json.dumps(tools) and "staff_id" not in json.dumps(tools)  # never "who is asking"


# --- permissions: the actor is bound in code ----------------------------------------------------


def test_a_trainee_of_another_group_is_not_found_in_the_same_words_as_one_that_does_not_exist(uow, repos):
    agent, cloud = build(
        uow, use(("trainee_record", {"trainee_id": 1}), ("trainee_record", {"trainee_id": 99})), say("לא מצאתי")
    )
    result = agent.chat(staff(repos, 3), "מה עם דני?")  # Dana: Dani is not hers
    hidden, missing = cloud.chats[1].tool_results()
    assert hidden == missing == "Error: not found (it does not exist, or is not yours to see)"
    assert [s.ok for s in result.steps] == [False, False]
    assert "כשקבענו" not in cloud.chats[1].everything()


def test_find_trainees_lists_only_whom_the_asker_may_see(uow, repos):
    agent, cloud = build(uow, use("find_trainees"), say("אלה החניכים"))
    agent.chat(staff(repos, 2), "מי החניכים שלי?")
    listed = cloud.chats[1].tool_results()[0]
    assert "[TRAINEE_1]" in listed and "[TRAINEE_3]" in listed
    assert "[TRAINEE_4]" not in listed  # Yael Cohen is in Dana's group


def test_a_trainee_the_asker_may_not_see_comes_back_only_as_they_wrote_them(uow, repos):
    """Dana writes "דני": the answer must not complete it to "דני אברהם" (review rule of stage 5)."""
    agent, cloud = build(uow, say("אין לי גישה ל-[TRAINEE_1]."))
    result = agent.chat(staff(repos, 3), "מה עם דני?")
    assert result.answer == "אין לי גישה ל-דני."


# --- ambiguous names: one token space for the whole conversation ---------------------------------


def test_an_ambiguous_name_keeps_one_token_across_the_message_and_the_tools(uow, repos):
    agent, cloud = build(uow, use(("find_trainees", {"name": "[NAME_1]"})), say("איזו [NAME_1]?"))
    result = agent.chat(staff(repos, 1), "איך יעל מסתדרת?")  # the manager sees both Yaels
    assert "[NAME_1] מסתדרת" in cloud.chats[0].messages[-1]["content"]
    found = cloud.chats[1].tool_results()[0]
    assert "[TRAINEE_3]" in found and "[TRAINEE_4]" in found  # both, by token
    assert result.answer == "איזו יעל?"
    assert result.steps[0].arguments == {"name": "יעל"}  # the trace shows the word, not the token


def test_text_that_looks_like_a_token_is_defused_and_never_becomes_a_name(uow, repos):
    agent, cloud = build(uow, say("[TRAINEE_1]"))
    agent.chat(staff(repos, 3), "מה זה [TRAINEE_1]?")
    assert "[TRAINEE_1]" not in cloud.chats[0].messages[-1]["content"]
    assert "TRAINEE_1" in cloud.chats[0].messages[-1]["content"]


def test_a_token_the_model_invented_is_reported_and_left_as_written(uow, repos):
    agent, cloud = build(uow, say("כדאי לדבר עם [TRAINEE_77]."))
    result = agent.chat(staff(repos, 2), "עם מי לדבר?")
    assert result.answer == "כדאי לדבר עם [TRAINEE_77]."
    assert [i.code for i in result.issues] == [IssueCode.UNKNOWN_TOKEN]


# --- safety: the order of the checks ------------------------------------------------------------


def test_a_risk_in_the_message_gets_the_procedure_and_the_agent_never_starts(uow, repos):
    agent, cloud = build(uow, say("לא אמור להגיע לכאן"))
    result = agent.chat(staff(repos, 2), "דני אמר לי שהוא רוצה למות")
    assert result.outcome is AgentOutcome.REFERRAL and result.referral is REFERRAL
    assert cloud.chats == [] and cloud.calls == []  # the phrase list: no model at all


def test_a_risk_in_an_earlier_turn_of_the_history_counts_too(uow, repos):
    agent, cloud = build(uow, say("לא אמור להגיע לכאן"))
    history = [Turn("user", "הוא אמר שהוא רוצה למות"), Turn("assistant", "...")]
    result = agent.chat(staff(repos, 2), "ומה עכשיו?", history)
    assert result.outcome is AgentOutcome.REFERRAL and cloud.chats == []


def test_the_screen_finds_what_the_phrases_missed_and_the_loop_never_starts(uow, repos):
    """The screen's own judgement is tested in test_risk_screen.py; here only what the agent does with it."""
    risk = json.dumps({"risk": True, "kinds": ["self_harm"], "reason": "a signal"})
    agent, cloud = build(uow, say("לא אמור להגיע לכאן"), answers=[risk])
    result = agent.chat(staff(repos, 2), "דני מתנהג מוזר השבוע")
    assert result.outcome is AgentOutcome.REFERRAL and cloud.chats == []
    assert [c.prompt.ref for c in cloud.calls] == ["risk_screen@v1"]


def test_the_agent_itself_can_report_an_incident_and_the_procedure_replaces_any_answer(uow, repos):
    agent, cloud = build(
        uow, use(("trainee_record", {"trainee_id": 1})), use("report_exceptional_incident"), say("עצה")
    )
    result = agent.chat(staff(repos, 2), "מה עם דני?")
    assert result.outcome is AgentOutcome.REFERRAL and result.referral is REFERRAL and result.answer is None
    assert [s.tool for s in result.steps] == ["trainee_record", "report_exceptional_incident"]
    assert len(cloud.chats) == 2  # it stopped at once


def test_a_referral_from_the_coach_ends_the_conversation_with_the_procedure(uow, repos):
    risk = json.dumps({"risk": True, "kinds": ["violence"], "reason": "a threat"})
    agent, cloud = build(
        uow,
        use(("ask_coach", {"trainee_id": 1, "question": "[TRAINEE_1] איים שיביא סכין מחר, מה לעשות?"})),
        say("עצה"),
        answers=[NO_RISK, risk],
    )
    result = agent.chat(staff(repos, 2), "מה לעשות עם דני?")
    assert result.outcome is AgentOutcome.REFERRAL and len(cloud.chats) == 1


# --- health: never sent, never shown -------------------------------------------------------------


def test_a_message_only_about_health_is_never_sent(uow, repos):
    agent, cloud = build(uow, say("..."))
    result = agent.chat(staff(repos, 2), "דני לוקח ריטלין.")
    assert result.outcome is AgentOutcome.NOT_SENT and result.note == NOT_SENT_NOTE
    assert cloud.chats == [] and result.removed_sentences == 1


def test_a_health_note_in_a_tool_result_is_removed_before_the_model_reads_it(uow, repos):
    repos.notes.add(
        TraineeNote(trainee_id=1, kind=NoteKind.OBSERVATION, text="נפגש עם פסיכולוג פעם בשבוע", author_id=2)
    )
    agent, cloud = build(uow, use(("trainee_record", {"trainee_id": 1})), say("סיימתי"))
    result = agent.chat(staff(repos, 2), "מה ידוע על דני?")
    assert "פסיכולוג" not in cloud.chats[1].everything()
    assert "שעון מעורר" in cloud.chats[1].everything()  # the rest of the record stays
    assert result.removed_sentences == 1


def test_a_health_sentence_in_the_answer_is_not_shown(uow, repos):
    agent, cloud = build(uow, say("כדאי לשבת איתו. אולי יש לו ADHD."))
    result = agent.chat(staff(repos, 2), "מה עם דני?")
    assert result.answer == "כדאי לשבת איתו."
    assert [i.code for i in result.issues] == [IssueCode.SENSITIVE_TEXT]


# --- the history ---------------------------------------------------------------------------------


def test_the_history_is_anonymized_again_and_only_its_last_turns_are_read(uow, repos):
    history = [Turn("user", f"שאלה {i}") for i in range(12)] + [Turn("assistant", "דני אברהם מאחר לספורט")]
    agent, cloud = build(uow, say("כן"))
    agent.chat(staff(repos, 2), "ומה עם נדב?", history)
    messages = cloud.chats[0].messages
    assert len(messages) == 11  # the last 10 turns, and the new message
    assert messages[-2] == {"role": "assistant", "content": "[TRAINEE_1] מאחר לספורט"}
    assert "שאלה 0" not in json.dumps(messages, ensure_ascii=False)


def test_a_history_turn_only_about_health_is_sent_as_a_placeholder(uow, repos):
    agent, cloud = build(uow, say("כן"))
    agent.chat(staff(repos, 2), "ומה עכשיו?", [Turn("user", "הוא לוקח ריטלין."), Turn("assistant", "הבנתי")])
    assert cloud.chats[0].messages[0] == {"role": "user", "content": EMPTY_LEFT}


# --- errors are results, limits stop the loop ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("delete_trainee", "{}", "There is no tool named 'delete_trainee'"),
        ("trainee_record", "{not json", "not valid JSON"),
        ("trainee_record", "[1]", "must be a JSON object"),
        ("trainee_record", '{"trainee_id": "שבע"}', "trainee_id"),
        ("trainee_record", '{"trainee_id": 1, "actor_id": 1}', "actor_id"),
        ("attendance", '{"trainee_id": 1, "group_id": 1}', "not both"),
    ],
)
def test_a_wrong_call_is_an_error_the_model_reads_and_the_loop_goes_on(uow, repos, name, arguments, expected):
    agent, cloud = build(uow, use((name, arguments)), say("טעיתי"))
    result = agent.chat(staff(repos, 2), "מה עם דני?")
    (error,) = cloud.chats[1].tool_results()
    assert error.startswith("Error:") and expected in error
    assert result.outcome is AgentOutcome.ANSWER and result.steps[0].ok is False


def test_a_validation_error_does_not_echo_the_value(uow, repos):
    agent, cloud = build(uow, use(("trainee_record", '{"trainee_id": "דני אברהם"}')), say("טעיתי"))
    agent.chat(staff(repos, 2), "מה עם דני?")
    assert "דני" not in cloud.chats[1].tool_results()[0]


def test_the_rounds_limit_the_last_round_must_answer_in_text(uow, repos):
    agent, cloud = build(uow, use("week_schedule"), max_rounds=3)
    result = agent.chat(staff(repos, 2), "מה יש השבוע?")
    assert [c.allow_tools for c in cloud.chats] == [True, True, False]
    assert result.outcome is AgentOutcome.STOPPED and result.stopped_by == "rounds"
    assert result.answer == STOPPED_ANSWER and len(result.steps) == 2


def test_the_tool_calls_limit_every_call_still_gets_a_result(uow, repos):
    agent, cloud = build(uow, use("week_schedule", "week_schedule", "week_schedule"), say("די"), max_tool_calls=2)
    result = agent.chat(staff(repos, 2), "מה יש השבוע?")
    results = cloud.chats[1].tool_results()
    assert len(results) == 3 and "limit of tool calls" in results[2]
    assert len(result.steps) == 2 and cloud.chats[1].allow_tools is False
    assert result.outcome is AgentOutcome.ANSWER


def test_the_cost_limit_counts_the_models_inside_tools_too(uow, repos):
    agent, cloud = build(uow, use("week_schedule", cost_usd=0.2), say("די"), max_cost_usd=0.15)
    result = agent.chat(staff(repos, 2), "מה יש השבוע?")
    assert result.outcome is AgentOutcome.STOPPED and result.stopped_by == "cost" and len(cloud.chats) == 1


def test_an_unknown_cost_makes_the_total_unknown_never_zero(uow, repos):
    agent, cloud = build(uow, say("כן", cost_usd=None))
    assert agent.chat(staff(repos, 2), "שלום").cost_usd is None


def test_an_empty_answer_is_a_stop_not_an_empty_answer(uow, repos):
    agent, cloud = build(uow, say("   "))
    result = agent.chat(staff(repos, 2), "שלום")
    assert result.outcome is AgentOutcome.STOPPED and result.stopped_by == "empty"


def test_an_empty_message_is_refused_before_anything(uow, repos):
    agent, cloud = build(uow)
    with pytest.raises(ValueError):
        agent.chat(staff(repos, 2), "   ")


# --- the tools' content --------------------------------------------------------------------------


def test_attendance_of_a_group_lowest_first_with_template_sessions_only(uow, repos):
    session = repos.sessions.add(
        ScheduledSession(cohort_id=1, on=date(2026, 10, 20), start=time(7), end=time(8), title="ספורט בוקר", slot_id=1)
    )
    one_off = repos.sessions.add(
        ScheduledSession(cohort_id=1, on=date(2026, 10, 21), start=time(9), end=time(10), title="פגישה עם דני על משהו")
    )
    for s in (session, one_off):
        repos.attendance.save(
            AttendanceRecord(session_id=s.key, trainee_id=1, status=AttendanceStatus.ABSENT, recorded_by_id=2)
        )
        repos.attendance.save(
            AttendanceRecord(session_id=s.key, trainee_id=2, status=AttendanceStatus.PRESENT, recorded_by_id=2)
        )
    agent, cloud = build(uow, use(("attendance", {"group_id": 1})), say("זה המצב"))
    agent.chat(staff(repos, 2), "מי מגיע הכי פחות?")
    lines = cloud.chats[1].tool_results()[0].splitlines()
    assert lines[1].startswith("- [TRAINEE_1] (trainee_id 1): 0%")
    assert lines[2] == "  - missed: ספורט בוקר (1)"
    assert "פגישה" not in cloud.chats[1].tool_results()[0]  # a one-off's free-text title stays off the model


def test_week_schedule_and_search_library(uow, repos):
    repos.sessions.add(
        ScheduledSession(cohort_id=1, on=date(2026, 10, 26), start=time(7), end=time(8), title="ספורט בוקר")
    )
    agent, cloud = build(uow, use("week_schedule", ("search_library", {"query": "חלוקת הלימוד"})), say("הנה"))
    result = agent.chat(staff(repos, 2), "מה השבוע ומה אומרים המקורות?")
    week, library = cloud.chats[1].tool_results()
    assert "The week of Sunday 2026-10-25" in week and "2026-10-26 07:00-08:00 · ספורט בוקר" in week
    assert "לפי כוחם" in library and "מקור ערכי-מסורתי" in library
    assert [s.summary for s in result.steps] == ["week of 2026-10-25: 1 sessions", "1 passages"]


def test_the_trace_shows_counts_never_content(uow, repos):
    agent, cloud = build(uow, use(("trainee_record", {"trainee_id": 1})), say("סיימתי"))
    (step,) = agent.chat(staff(repos, 2), "מה עם דני?").steps
    assert step.summary == "trainee 1: 1 goals, 1 notes" and step.arguments == {"trainee_id": 1}


def test_the_trace_restores_names_with_the_same_rule_as_the_answer(uow, repos):
    """Dana wrote "דני": the trace under the answer must not complete it either."""
    agent, cloud = build(uow, use(("search_library", {"query": "איך לעזור ל[TRAINEE_1] להגיע בזמן"})), say("הנה"))
    (step,) = agent.chat(staff(repos, 3), "איך לעזור לדני להגיע בזמן?").steps
    assert step.arguments == {"query": "איך לעזור לדני להגיע בזמן"}


def test_two_different_ambiguous_words_never_share_a_token(uow, repos):
    """ "יעל" (two trainees) in the message, and "בר" (a name that is also a word) in a tool's result:
    anonymized one by one, each would be [NAME_1]. In one conversation they must not be."""
    repos.sessions.add(
        ScheduledSession(cohort_id=1, on=date(2026, 10, 26), start=time(19), end=time(21), title="ערב בר מצווה")
    )
    agent, cloud = build(uow, use("week_schedule"), say("[NAME_1]: [NAME_2]"))
    result = agent.chat(staff(repos, 1), "האם יעל צריכה להגיע לאירוע השבוע?")
    assert "[NAME_1]" in cloud.chats[0].messages[-1]["content"]
    week = cloud.chats[1].tool_results()[0]
    assert "ערב [NAME_2] מצווה" in week and "[NAME_1]" not in week
    assert result.answer == "יעל: בר"
