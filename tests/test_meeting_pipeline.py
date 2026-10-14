"""The meeting pipeline, on the demo academy in memory, with fake models. No database, no network.

What is checked here, besides the happy path:
* privacy: the cloud model never receives a known name, the title, or raw text;
* transactions: no model is ever called while a (fake) transaction is open;
* content validation: owners, dates, duplicates and invented tokens;
* failures leave the meeting where it was, and running again continues from there.
"""

import json
from datetime import date

import pytest
from fake_llm import FakeLLMClient, FakeTranscriber, fake_router
from fakes import FakeUnitOfWork, empty_repos

from pacer.ai import LLMUnavailableError
from pacer.ai.anonymization import AnonymizationError, Anonymizer, load_roster
from pacer.ai.meeting_pipeline import (
    MeetingExtraction,
    MeetingPipeline,
    extraction_message,
)
from pacer.ai.routing import PrivacyViolation, RoutingError
from pacer.ai.structured import StructuredOutputError
from pacer.ai.transcription import RecordingStore, TranscriptionError
from pacer.demo import build_academy
from pacer.demo.meetings import SHORT, WEEKLY, SampleMeeting
from pacer.domain import MeetingStatus, StaffMember
from pacer.services import MeetingService, PermissionDeniedError, RuleViolationError

# What a good model answer to the WEEKLY sample looks like: tokens only, as the model saw them.
WEEKLY_ANSWER = {
    "summary": "[STAFF_1] פתחה את הישיבה. [STAFF_5] יתאם את ההסעה לטיול השנתי. "
    "[STAFF_2] תקבע שעות התנדבות בבית האבות. [STAFF_4] תכין את תוכן הסמינר. "
    "[STAFF_3] ישוחח עם [TRAINEE_16] על ריצות הבוקר.",
    "tasks": [
        {"description": "לתאם את ההסעה לטיול השנתי עם חברת האוטובוסים", "owner": "[STAFF_5]", "due_date": "2026-10-15"},
        {"description": "לקבוע עם רכזת בית האבות את שעות ההתנדבות", "owner": "[STAFF_2]", "due_date": None},
        {"description": "להכין את תוכן הסמינר על מנהיגות ולשלוח לצוות", "owner": "[STAFF_4]", "due_date": "2026-10-18"},
        {"description": "לשוחח עם [TRAINEE_16] על ריצות הבוקר", "owner": "[STAFF_3]", "due_date": None},
    ],
}


def answer(summary: str = "סיכום קצר של הישיבה, בלי שמות בכלל.", tasks: list | None = None) -> str:
    return json.dumps({"summary": summary, "tasks": tasks or []}, ensure_ascii=False)


@pytest.fixture
def repos():
    repos = empty_repos()
    build_academy(repos)  # 1 רונית (manager), 2 נועה, 3 עומר, 4 שירה, 5 איתי; 35 trainees
    return repos


@pytest.fixture
def uow(repos) -> FakeUnitOfWork:
    return FakeUnitOfWork(repos)


@pytest.fixture
def manager(repos) -> StaffMember:
    return repos.staff.get(1)


@pytest.fixture
def cloud() -> FakeLLMClient:
    return FakeLLMClient(answers=[json.dumps(WEEKLY_ANSWER, ensure_ascii=False)], model="fake/cloud", runs_locally=False)


@pytest.fixture
def recordings(tmp_path) -> RecordingStore:
    return RecordingStore(tmp_path / "recordings")


def make_pipeline(uow, cloud, *, transcriber=None, recordings=None) -> MeetingPipeline:
    return MeetingPipeline(uow, fake_router(cloud), transcriber=transcriber, recordings=recordings)


def record(uow, manager, sample: SampleMeeting = WEEKLY, *, with_transcript: bool = True) -> int:
    with uow() as repos:
        service = MeetingService(repos.meetings, repos.proposals, repos.tasks, repos.staff)
        meeting = service.create_meeting(
            manager,
            title=sample.title,
            held_on=sample.held_on,
            participant_ids=sample.participant_ids,
            transcript=sample.transcript if with_transcript else None,
        )
    return meeting.id


# --- the happy path ---------------------------------------------------------------------------


def test_a_transcript_becomes_a_summary_and_proposals_with_real_names(uow, repos, manager, cloud):
    meeting_id = record(uow, manager)
    result = make_pipeline(uow, cloud).process(manager, meeting_id)

    assert result.meeting.status is MeetingStatus.PENDING_APPROVAL
    assert [(p.owner_id, p.due_date) for p in result.proposals] == [
        (5, date(2026, 10, 15)),
        (2, None),
        (4, date(2026, 10, 18)),
        (3, None),
    ]
    assert result.proposals[3].description == "לשוחח עם אדל אזולאי על ריצות הבוקר"  # token -> name, here
    assert "איתי ביטון" in result.meeting.summary and "[STAFF_" not in result.meeting.summary
    assert result.issues == []
    assert result.prompt_ref == "meeting_extract@v1" and result.attempts == 1
    assert repos.proposals.list_by_meeting(meeting_id) == result.proposals  # saved, not only returned


def test_the_stored_anonymized_transcript_is_exactly_what_the_model_saw(uow, repos, manager, cloud):
    meeting_id = record(uow, manager)
    make_pipeline(uow, cloud).process(manager, meeting_id)

    stored = repos.meetings.get(meeting_id).anonymized_transcript
    assert stored in cloud.calls[0].user_message
    assert not Anonymizer(load_roster(repos.staff, repos.trainees)).contains_known_name(stored)


def test_the_model_gets_the_pinned_prompt_the_schema_and_the_route_budget(uow, manager, cloud):
    make_pipeline(uow, cloud).process(manager, record(uow, manager))
    [call] = cloud.calls
    assert call.prompt.ref == "meeting_extract@v1"
    assert call.response_schema == MeetingExtraction.model_json_schema()
    assert call.max_tokens == 4000  # ROUTES[EXTRACT_MEETING]


# --- privacy -----------------------------------------------------------------------------------


@pytest.mark.parametrize("sample", [WEEKLY, SHORT], ids=["weekly", "short"])
def test_the_cloud_model_never_sees_a_known_name_or_the_title(uow, repos, manager, cloud, sample):
    make_pipeline(uow, cloud).process(manager, record(uow, manager, sample))

    anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))
    for call in cloud.calls:
        assert not anonymizer.contains_known_name(call.user_message)
        assert sample.title not in call.user_message  # free text from the manager: may hold names
        for person in [*repos.staff.list_all(), *repos.trainees.list_all()]:
            assert person.full_name not in call.user_message


def test_the_message_lists_participants_by_token_and_role_and_the_date(uow, manager, cloud):
    make_pipeline(uow, cloud).process(manager, record(uow, manager, SHORT))
    message = cloud.calls[0].user_message
    assert "Meeting date: 2026-10-13 (Tuesday)" in message
    assert "- [STAFF_1]: manager\n- [STAFF_3]: counselor\n- [STAFF_5]: counselor" in message
    assert "[STAFF_4]" in message  # שירה is mentioned in the text, but is not listed as a participant
    assert "- [STAFF_4]" not in message


def test_the_message_builder_refuses_raw_text():
    with pytest.raises(TypeError, match="AnonymizedText"):
        extraction_message("נועה תתאם הסעה", date(2026, 10, 12), [])  # type: ignore[arg-type]


def test_a_name_in_the_final_message_stops_the_call(uow, manager, cloud, monkeypatch):
    """Defense in depth: even if building the message went wrong, nothing with a name is sent."""
    import pacer.ai.meeting_pipeline as module

    monkeypatch.setattr(module, "extraction_message", lambda safe, held_on, participants: WEEKLY.transcript)
    meeting_id = record(uow, manager)
    with pytest.raises(AnonymizationError):
        make_pipeline(uow, cloud).process(manager, meeting_id)
    assert cloud.calls == []


def test_a_cloud_model_that_claims_to_be_local_is_refused(uow, manager):
    """The router compares the table with the real client: a mismatch never gets a call."""
    liar = FakeLLMClient(runs_locally=True)  # in the table as cloud, says it is local
    with pytest.raises(RoutingError, match="declared cloud"):
        MeetingPipeline(uow, fake_router(liar)).process(manager, record(uow, manager))
    assert liar.calls == []


# --- transactions --------------------------------------------------------------------------------


def test_no_transaction_is_open_while_the_model_works(uow, manager):
    seen_open: list[int] = []

    def model(message: str) -> str:
        seen_open.append(uow.open)
        return json.dumps(WEEKLY_ANSWER, ensure_ascii=False)

    cloud = FakeLLMClient(answers=model, runs_locally=False)
    make_pipeline(uow, cloud).process(manager, record(uow, manager))
    assert seen_open == [0]
    assert uow.open == 0


def test_summary_and_proposals_are_saved_together_or_not_at_all(uow, repos, manager, cloud, monkeypatch):
    def broken(self, meeting_id, drafts):
        raise RuntimeError("e.g. the database went away mid-write")

    monkeypatch.setattr(MeetingService, "submit_proposals", broken)
    meeting_id = record(uow, manager)
    with pytest.raises(RuntimeError):
        make_pipeline(uow, cloud).process(manager, meeting_id)

    meeting = repos.meetings.get(meeting_id)
    assert meeting.status is MeetingStatus.ANONYMIZED and meeting.summary is None  # rolled back
    assert repos.proposals.list_by_meeting(meeting_id) == []


def test_an_answer_that_arrives_after_another_request_finished_is_discarded(uow, repos, manager):
    """Two requests processed the same meeting; the other one saved first, while ours waited."""
    meeting_id = record(uow, manager)

    def model(message: str) -> str:
        with uow() as other:  # the other request finishes the stage meanwhile
            meeting = other.meetings.get(meeting_id)
            other.meetings.update(meeting.advance(MeetingStatus.EXTRACTED, summary="סיכום של הבקשה האחרת"))
        return json.dumps(WEEKLY_ANSWER, ensure_ascii=False)

    pipeline = make_pipeline(uow, FakeLLMClient(answers=model, runs_locally=False))
    with pytest.raises(RuleViolationError, match="another request"):
        pipeline.process(manager, meeting_id)
    assert repos.meetings.get(meeting_id).summary == "סיכום של הבקשה האחרת"
    assert repos.proposals.list_by_meeting(meeting_id) == []  # nothing saved twice


# --- failures, and running again ------------------------------------------------------------------


def test_an_invalid_answer_twice_leaves_the_meeting_anonymized_and_a_rerun_continues(uow, repos, manager):
    meeting_id = record(uow, manager)
    bad = FakeLLMClient(answers=['{"summary": "קצר"}'], runs_locally=False)
    with pytest.raises(StructuredOutputError):
        make_pipeline(uow, bad).process(manager, meeting_id)
    assert repos.meetings.get(meeting_id).status is MeetingStatus.ANONYMIZED
    assert len(bad.calls) == 2  # the first answer and one correction

    good = FakeLLMClient(answers=[json.dumps(WEEKLY_ANSWER, ensure_ascii=False)], runs_locally=False)
    result = make_pipeline(uow, good).process(manager, meeting_id)
    assert result.meeting.status is MeetingStatus.PENDING_APPROVAL
    assert len(result.proposals) == 4


def test_a_model_that_is_down_changes_nothing(uow, repos, manager):
    def down(message: str) -> str:
        raise LLMUnavailableError("overloaded")

    meeting_id = record(uow, manager)
    with pytest.raises(LLMUnavailableError):
        make_pipeline(uow, FakeLLMClient(answers=down, runs_locally=False)).process(manager, meeting_id)
    assert repos.meetings.get(meeting_id).status is MeetingStatus.ANONYMIZED
    assert repos.meetings.get(meeting_id).summary is None


def test_each_stage_can_run_on_its_own(uow, repos, manager, cloud):
    meeting_id = record(uow, manager)
    pipeline = make_pipeline(uow, cloud)

    assert pipeline.anonymize(manager, meeting_id).status is MeetingStatus.ANONYMIZED
    assert cloud.calls == []  # anonymizing is local: no model at all
    assert pipeline.extract(manager, meeting_id).meeting.status is MeetingStatus.PENDING_APPROVAL


def test_a_stage_out_of_order_is_refused(uow, manager, cloud):
    meeting_id = record(uow, manager)
    with pytest.raises(RuleViolationError, match="needs anonymized"):
        make_pipeline(uow, cloud).extract(manager, meeting_id)  # not anonymized yet
    assert cloud.calls == []


def test_a_processed_meeting_has_nothing_left_to_process(uow, manager, cloud):
    meeting_id = record(uow, manager)
    pipeline = make_pipeline(uow, cloud)
    pipeline.process(manager, meeting_id)
    with pytest.raises(RuleViolationError, match="nothing left"):
        pipeline.process(manager, meeting_id)
    assert len(cloud.calls) == 1  # never paid for twice


def test_a_meeting_without_tasks_is_closed_at_once(uow, manager):
    cloud = FakeLLMClient(answers=[answer()], runs_locally=False)
    result = make_pipeline(uow, cloud).process(manager, record(uow, manager))
    assert result.proposals == []
    assert result.meeting.status is MeetingStatus.APPROVED  # nothing for the manager to decide


def test_only_the_manager_runs_the_pipeline(uow, repos, manager, cloud):
    meeting_id = record(uow, manager)
    counselor = repos.staff.get(2)
    with pytest.raises(PermissionDeniedError):
        make_pipeline(uow, cloud).process(counselor, meeting_id)
    with pytest.raises(PermissionDeniedError):
        make_pipeline(uow, cloud).anonymize(counselor, meeting_id)
    assert cloud.calls == []


# --- audio: transcription on this computer, then the recording is deleted ---------------------------


TRANSCRIBED = "רונית: נתחיל. איתי, תתאם בבקשה את ההסעה לטיול עד יום חמישי."


def upload(pipeline, manager, meeting_id, data: bytes = b"fake-audio", suffix: str = ".m4a") -> int:
    return pipeline.attach_recording(manager, meeting_id, iter([data]), suffix)


def test_a_recording_is_transcribed_locally_then_deleted(uow, repos, manager, cloud, recordings):
    transcriber = FakeTranscriber(text=TRANSCRIBED)
    pipeline = make_pipeline(uow, cloud, transcriber=transcriber, recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)

    assert upload(pipeline, manager, meeting_id) == len(b"fake-audio")
    result = pipeline.process(manager, meeting_id)

    assert transcriber.calls == [recordings.directory / f"meeting-{meeting_id}.m4a"]
    assert recordings.find(meeting_id) is None  # deleted after the transcript was saved
    assert repos.meetings.get(meeting_id).transcript == TRANSCRIBED
    assert result.meeting.status is MeetingStatus.PENDING_APPROVAL


def test_a_failed_transcription_keeps_the_recording_and_the_status(uow, repos, manager, cloud, recordings):
    transcriber = FakeTranscriber(error=TranscriptionError("corrupt file"))
    pipeline = make_pipeline(uow, cloud, transcriber=transcriber, recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    upload(pipeline, manager, meeting_id)

    with pytest.raises(TranscriptionError):
        pipeline.process(manager, meeting_id)
    assert recordings.find(meeting_id) is not None  # kept, so it can be tried again
    assert repos.meetings.get(meeting_id).status is MeetingStatus.UPLOADED
    assert cloud.calls == []


def test_a_silent_recording_is_not_saved_as_a_transcript(uow, repos, manager, cloud, recordings):
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text="  אה  "), recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    upload(pipeline, manager, meeting_id)

    with pytest.raises(RuleViolationError, match="almost no text"):
        pipeline.process(manager, meeting_id)
    assert recordings.find(meeting_id) is not None
    assert repos.meetings.get(meeting_id).status is MeetingStatus.UPLOADED


def test_a_transcriber_that_is_not_local_is_refused(uow, manager, cloud, recordings):
    remote = FakeTranscriber(text=TRANSCRIBED, runs_locally=False)
    pipeline = make_pipeline(uow, cloud, transcriber=remote, recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    with pytest.raises(PrivacyViolation):
        upload(pipeline, manager, meeting_id)
    assert remote.calls == []


def test_without_a_transcriber_recordings_are_refused_up_front(uow, manager, cloud, recordings):
    pipeline = make_pipeline(uow, cloud, recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    with pytest.raises(RuleViolationError, match="not set up"):
        upload(pipeline, manager, meeting_id)
    assert recordings.find(meeting_id) is None


def test_a_meeting_that_has_a_transcript_takes_no_recording(uow, manager, cloud, recordings):
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text=TRANSCRIBED), recordings=recordings)
    meeting_id = record(uow, manager)  # transcribed already
    with pytest.raises(RuleViolationError, match="needs uploaded"):
        upload(pipeline, manager, meeting_id)


def test_a_counselor_cannot_upload_a_recording(uow, repos, manager, cloud, recordings):
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text=TRANSCRIBED), recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    with pytest.raises(PermissionDeniedError):
        upload(pipeline, repos.staff.get(3), meeting_id)
    assert recordings.find(meeting_id) is None


def test_a_leftover_recording_is_deleted_by_the_next_stage(uow, manager, cloud, recordings):
    """If deleting failed after transcription (a crash), anonymizing removes it."""
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text=TRANSCRIBED), recordings=recordings)
    meeting_id = record(uow, manager)
    recordings.save(meeting_id, iter([b"audio"]), ".wav")  # as if left behind
    pipeline.anonymize(manager, meeting_id)
    assert recordings.find(meeting_id) is None


def test_if_saving_the_transcript_fails_the_recording_is_kept(uow, repos, manager, cloud, recordings, monkeypatch):
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text=TRANSCRIBED), recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    upload(pipeline, manager, meeting_id)

    def broken(meeting):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(repos.meetings, "update", broken)
    with pytest.raises(RuntimeError):
        pipeline.transcribe(manager, meeting_id)
    assert recordings.find(meeting_id) is not None  # deleted only after the transcript is safe


def test_the_transcribe_stage_itself_deletes_the_recording(uow, manager, cloud, recordings):
    """Not only the sweep in the next stage: transcription deletes what it transcribed."""
    pipeline = make_pipeline(uow, cloud, transcriber=FakeTranscriber(text=TRANSCRIBED), recordings=recordings)
    meeting_id = record(uow, manager, with_transcript=False)
    upload(pipeline, manager, meeting_id)
    pipeline.transcribe(manager, meeting_id)
    assert recordings.find(meeting_id) is None


def test_a_participant_who_never_spoke_can_be_named_by_token(uow, manager):
    """נועה (2) attended but is never named in the text. The model may still give her a task by her
    participant token, and both the task and the summary get her name back."""
    silent = SampleMeeting(
        title="ישיבה",
        held_on=date(2026, 10, 12),
        participant_ids=[1, 2],
        transcript="רונית: מי שאחראית על ההתנדבות תקבע את השעות עם בית האבות עד חמישי.",
    )
    reply = answer(
        summary="[STAFF_1] ביקשה מ[STAFF_2] לקבוע את שעות ההתנדבות.",
        tasks=[{"description": "לקבוע את שעות ההתנדבות", "owner": "[STAFF_2]", "due_date": "2026-10-15"}],
    )
    result = make_pipeline(uow, FakeLLMClient(answers=[reply], runs_locally=False)).process(
        manager, record(uow, manager, silent)
    )
    assert result.meeting.summary == "רונית שמעוני ביקשה מנועה לוי לקבוע את שעות ההתנדבות."
    assert result.proposals[0].owner_id == 2
    assert result.issues == []  # her token was in the message, so it is not "invented"
