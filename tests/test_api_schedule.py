"""The schedule and attendance endpoints, over in-memory fakes."""

from collections.abc import Iterator
from datetime import date

import pytest
from fakes import World
from fastapi.testclient import TestClient

from pacer.api.app import create_app
from pacer.api.dependencies import (
    get_attendance_repository,
    get_cohort_repository,
    get_db_engine,
    get_group_repository,
    get_scheduled_session_repository,
    get_slot_repository,
    get_staff_repository,
    get_today,
    get_trainee_repository,
)

SUNDAY = "2026-10-18"


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(world: World) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides = {
        get_db_engine: lambda: (_ for _ in ()).throw(AssertionError("no database in these tests")),
        get_staff_repository: lambda: world.staff,
        get_group_repository: lambda: world.groups,
        get_trainee_repository: lambda: world.trainees,
        get_cohort_repository: lambda: world.cohorts,
        get_slot_repository: lambda: world.slots,
        get_scheduled_session_repository: lambda: world.sessions,
        get_attendance_repository: lambda: world.attendance,
        get_today: lambda: date(2026, 10, 25),
    }
    with TestClient(app) as test_client:
        yield test_client


def as_(staff) -> dict[str, str]:
    return {"X-Staff-Id": str(staff.id)}


@pytest.fixture
def week(client, world):
    """A template of two slots (Sunday sport, Monday lesson), published for one week."""
    for weekday, title in [(6, "ספורט בוקר"), (0, "שיעור עיון")]:
        body = {"cohort_id": world.cohort.id, "weekday": weekday, "start": "07:00", "end": "08:00", "title": title}
        assert client.post("/schedule/slots", json=body, headers=as_(world.manager)).status_code == 201
    response = client.post(f"/schedule/weeks/{SUNDAY}/publish", params={"cohort_id": world.cohort.id}, headers=as_(world.manager))
    assert response.status_code == 200, response.text
    return response.json()


def test_the_template_and_the_week(client, world, week):
    slots = client.get("/schedule/slots", params={"cohort_id": world.cohort.id}, headers=as_(world.noa)).json()
    assert [(s["day"], s["title"]) for s in slots] == [("ראשון", "ספורט בוקר"), ("שני", "שיעור עיון")]
    sessions = client.get("/schedule/weeks/2026-10-21", params={"cohort_id": world.cohort.id}, headers=as_(world.noa)).json()
    assert [(s["on"], s["weekday"]) for s in sessions] == [(SUNDAY, "ראשון"), ("2026-10-19", "שני")]
    assert [s["id"] for s in week] == [s["id"] for s in sessions]


def test_a_counselor_cannot_change_the_schedule(client, world, week):
    body = {"cohort_id": world.cohort.id, "weekday": 6, "start": "09:00", "end": "10:00", "title": "סדנה"}
    assert client.post("/schedule/slots", json=body, headers=as_(world.noa)).status_code == 403
    assert client.post(f"/schedule/sessions/{week[0]['id']}/cancel", headers=as_(world.noa)).status_code == 403


def test_bad_input_is_422_or_409(client, world):
    bad_times = {"cohort_id": world.cohort.id, "weekday": 6, "start": "09:00", "end": "08:00", "title": "סדנה"}
    assert client.post("/schedule/slots", json=bad_times, headers=as_(world.manager)).status_code == 422
    monday = client.post("/schedule/weeks/2026-10-19/publish", params={"cohort_id": world.cohort.id}, headers=as_(world.manager))
    assert monday.status_code == 409 and "Sunday" in monday.json()["detail"]


def test_marking_and_the_sheet(client, world, week):
    session_id = week[0]["id"]
    sheet = client.get(f"/schedule/sessions/{session_id}/attendance", headers=as_(world.noa)).json()
    assert sheet["trainees"] == [{"trainee_id": world.yael.id, "name": "Yael Mizrahi", "status": None}]
    marks = {"marks": [{"trainee_id": world.yael.id, "status": "absent"}]}
    response = client.put(f"/schedule/sessions/{session_id}/attendance", json=marks, headers=as_(world.noa))
    assert response.status_code == 200, response.text
    assert response.json()["trainees"][0]["status"] == "absent"


def test_marking_needs_a_known_status_and_no_reason_field(client, world, week):
    url = f"/schedule/sessions/{week[0]['id']}/attendance"
    assert client.put(url, json={"marks": [{"trainee_id": world.yael.id, "status": "late"}]}, headers=as_(world.noa)).status_code == 422
    with_reason = {"marks": [{"trainee_id": world.yael.id, "status": "excused", "reason": "אצל רופא"}]}
    assert client.put(url, json=with_reason, headers=as_(world.noa)).status_code == 422  # never stored


def test_a_counselor_of_another_group_gets_404_for_the_trainee(client, world, week):
    marks = {"marks": [{"trainee_id": world.yael.id, "status": "present"}]}
    assert client.put(f"/schedule/sessions/{week[0]['id']}/attendance", json=marks, headers=as_(world.dana)).status_code == 404
    assert client.get(f"/trainees/{world.yael.id}/attendance", headers=as_(world.dana)).status_code == 404


def test_the_rates(client, world, week):
    for session, status in zip(week, ("present", "excused"), strict=True):
        marks = {"marks": [{"trainee_id": world.yael.id, "status": status}]}
        client.put(f"/schedule/sessions/{session['id']}/attendance", json=marks, headers=as_(world.noa))
    one = client.get(f"/trainees/{world.yael.id}/attendance", params={"from": SUNDAY}, headers=as_(world.noa)).json()
    assert (one["percent"], one["present"], one["excused"], one["unmarked"]) == (100, 1, 1, 0)
    rows = client.get("/attendance", params={"cohort_id": world.cohort.id}, headers=as_(world.manager)).json()
    assert [r["name"] for r in rows] == ["Yael Mizrahi"]
