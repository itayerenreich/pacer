"""Meetings, proposals and tasks against PostgreSQL: the third migration, the participants link
table, and the whole manager flow over HTTP."""

from datetime import date

import pytest
from fastapi.testclient import TestClient
from test_repositories import seed_group

from pacer.api.app import create_app
from pacer.api.dependencies import get_db_engine
from pacer.domain import MeetingStatus, ProposalStatus, ProposedTask, StaffMeeting, StaffMember, TaskStatus
from pacer.repositories import (
    MeetingRepository,
    ProposedTaskRepository,
    StaffRepository,
    TaskRepository,
    unit_of_work,
)
from pacer.services import MeetingService, TaskDraft

pytestmark = pytest.mark.integration
TRANSCRIPT = "נועה: נתחיל. עומר: צריך לתאם הסעה לטיול ולהזמין ציוד."


@pytest.fixture
def staff(engine) -> dict[str, int]:
    seed = seed_group(engine)
    with unit_of_work(engine) as session:
        manager = StaffRepository(session).add(StaffMember(full_name="Ronit Manager", role="manager"))
    return {"manager": manager.id, "noa": seed.counselor_ids[0], "omer": seed.counselor_ids[1]}


def test_a_meeting_round_trip_with_its_participants(engine, staff):
    with unit_of_work(engine) as session:
        saved = MeetingRepository(session).add(
            StaffMeeting(
                title="ישיבת צוות",
                held_on=date(2026, 10, 12),
                created_by_id=staff["manager"],
                participant_ids=[staff["omer"], staff["noa"]],
                status=MeetingStatus.TRANSCRIBED,
                transcript=TRANSCRIPT,
            )
        )
    with unit_of_work(engine) as session:
        repo = MeetingRepository(session)
        loaded = repo.get(saved.id)
        repo.update(loaded.model_copy(update={"participant_ids": [staff["manager"], staff["noa"]]}))
    with unit_of_work(engine) as session:
        reloaded = MeetingRepository(session).get(saved.id)

    assert loaded.transcript == TRANSCRIPT  # Hebrew text survives
    assert sorted(loaded.participant_ids) == sorted([staff["omer"], staff["noa"]])
    assert reloaded.participant_ids == sorted([staff["manager"], staff["noa"]])  # replaced, not appended


def test_open_proposals_are_counted_in_the_database(engine, staff):
    with unit_of_work(engine) as session:
        meeting = MeetingRepository(session).add(
            StaffMeeting(
                title="ישיבה",
                held_on=date(2026, 10, 12),
                created_by_id=staff["manager"],
                participant_ids=[staff["manager"]],
            )
        )
        proposals = ProposedTaskRepository(session)
        first = proposals.add(
            ProposedTask(meeting_id=meeting.id, description="משימה ראשונה", owner_id=staff["noa"])
        )
        proposals.add(ProposedTask(meeting_id=meeting.id, description="משימה שנייה"))
        proposals.update(first.reject(staff["manager"]))

    with unit_of_work(engine) as session:
        assert ProposedTaskRepository(session).count_open(meeting.id) == 1


def test_the_manager_flow_end_to_end_over_http(engine, staff):
    manager, omer = {"X-Staff-Id": str(staff["manager"])}, {"X-Staff-Id": str(staff["omer"])}
    app = create_app()
    app.dependency_overrides[get_db_engine] = lambda: engine

    with TestClient(app) as client:
        created = client.post(
            "/meetings",
            json={
                "title": "ישיבת צוות שבועית",
                "held_on": "2026-10-12",
                "participant_ids": [staff["manager"], staff["omer"]],
                "transcript": TRANSCRIPT,
            },
            headers=manager,
        )
        meeting_id = created.json()["id"]

        # Stand-in for the AI pipeline of task 6: anonymize, extract, submit.
        with unit_of_work(engine) as session:
            meetings = MeetingRepository(session)
            meeting = meetings.get(meeting_id).advance(MeetingStatus.ANONYMIZED, anonymized_transcript="טקסט")
            meetings.update(meeting.advance(MeetingStatus.EXTRACTED, summary="סוכם לתאם הסעה"))
            MeetingService(
                meetings, ProposedTaskRepository(session), TaskRepository(session), StaffRepository(session)
            ).submit_proposals(meeting_id, [TaskDraft(description="לתאם הסעה לטיול", owner_id=staff["omer"])])

        [proposal] = client.get(f"/meetings/{meeting_id}/proposals", headers=manager).json()
        task = client.post(
            f"/proposals/{proposal['id']}/approve", json={"due_date": "2026-10-20"}, headers=manager
        ).json()
        mine = client.get("/tasks/mine", headers=omer).json()
        done = client.post(f"/tasks/{task['id']}/complete", headers=omer).json()
        meeting_now = client.get(f"/meetings/{meeting_id}", headers=manager).json()

    assert created.status_code == 201 and created.json()["status"] == "transcribed"
    assert [t["id"] for t in mine] == [task["id"]]
    assert task["due_date"] == "2026-10-20"  # the manager's edit was applied
    assert done["status"] == TaskStatus.DONE
    assert meeting_now["status"] == MeetingStatus.APPROVED  # its only proposal was decided
    with unit_of_work(engine) as session:
        assert ProposedTaskRepository(session).get(proposal["id"]).status is ProposalStatus.APPROVED
