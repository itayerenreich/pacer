"""Integration tests: repositories against a real PostgreSQL (see conftest.py).

Each test writes in one unit of work and reads back in a NEW one, so the data
really went through the database and is not just cached in a session.
"""

from dataclasses import dataclass
from datetime import date

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from pacer.domain import (
    Cohort,
    Goal,
    GoalCategory,
    GoalStatus,
    GoalUpdate,
    Group,
    PersonalFile,
    StaffMember,
    Trainee,
    TraineeStatus,
)
from pacer.repositories import (
    CohortRepository,
    GoalRepository,
    GoalUpdateRepository,
    GroupRepository,
    PersonalFileRepository,
    StaffRepository,
    TraineeRepository,
    unit_of_work,
)

pytestmark = pytest.mark.integration  # every test in this file needs Docker


@dataclass
class Seed:
    """The ids of the rows every test needs before it can save a trainee."""

    cohort_id: int
    counselor_ids: list[int]
    group_id: int


def seed_group(engine: Engine) -> Seed:
    """A cohort, two counselors, and a group that both counselors lead."""
    with unit_of_work(engine) as session:
        cohort = CohortRepository(session).add(
            Cohort(name="Mahzor 12", start_date=date(2026, 9, 1), end_date=date(2027, 6, 30))
        )
        staff = StaffRepository(session)
        counselors = [
            staff.add(StaffMember(full_name="Noa Levi", role="counselor")),
            staff.add(StaffMember(full_name="Omer Cohen", role="counselor")),
        ]
        counselor_ids = [c.id for c in counselors]
        group = GroupRepository(session).add(
            Group(name="Group A", cohort_id=cohort.id, counselor_ids=counselor_ids)
        )
    return Seed(cohort_id=cohort.id, counselor_ids=counselor_ids, group_id=group.id)


def make_trainee(seed: Seed, **overrides) -> Trainee:
    data = {
        "first_name": "Yael",
        "last_name": "Mizrahi",
        "cohort_id": seed.cohort_id,
        "group_id": seed.group_id,
        "primary_counselor_id": seed.counselor_ids[0],
    }
    data.update(overrides)
    return Trainee(**data)


def test_trainee_with_two_goals_round_trip(engine):
    """The week 1 milestone: save a trainee with 2 goals, load them back intact."""
    seed = seed_group(engine)

    with unit_of_work(engine) as session:
        trainee = TraineeRepository(session).add(make_trainee(seed))
        goals = GoalRepository(session)
        goals.add(Goal(trainee_id=trainee.id, title="Run 5 km", category="physical"))
        goals.add(
            Goal(
                trainee_id=trainee.id,
                title="Lead a group discussion",
                category="leadership",
                status="in_progress",
                progress=40,
                target_date=date(2027, 1, 15),
            )
        )

    with unit_of_work(engine) as session:
        loaded = TraineeRepository(session).get(trainee.id)
        loaded_goals = GoalRepository(session).list_by_trainee(trainee.id)

    assert loaded == trainee
    assert loaded.status is TraineeStatus.ACTIVE  # text in the DB, an Enum again in Python
    assert [g.title for g in loaded_goals] == ["Run 5 km", "Lead a group discussion"]
    assert loaded_goals[1].category is GoalCategory.LEADERSHIP
    assert loaded_goals[1].progress == 40
    assert loaded_goals[1].target_date == date(2027, 1, 15)
    assert loaded_goals[0].created_at.utcoffset() is not None  # still timezone-aware


def test_group_keeps_its_counselors(engine):
    seed = seed_group(engine)

    with unit_of_work(engine) as session:
        group = GroupRepository(session).get(seed.group_id)

    assert group.counselor_ids == seed.counselor_ids


def test_goal_update_is_saved(engine):
    seed = seed_group(engine)
    with unit_of_work(engine) as session:
        trainee = TraineeRepository(session).add(make_trainee(seed))
        goal = GoalRepository(session).add(
            Goal(trainee_id=trainee.id, title="Run 5 km", category="physical")
        )

    with unit_of_work(engine) as session:
        finished = goal.model_copy(update={"status": GoalStatus.ACHIEVED, "progress": 100})
        GoalRepository(session).update(finished)
        GoalUpdateRepository(session).add(
            GoalUpdate(goal_id=goal.id, author_id=seed.counselor_ids[0], progress=100, note="Done!")
        )

    with unit_of_work(engine) as session:
        loaded = GoalRepository(session).get(goal.id)
        history = GoalUpdateRepository(session).list_by_goal(goal.id)

    assert loaded.status is GoalStatus.ACHIEVED
    assert [u.note for u in history] == ["Done!"]


def test_personal_file_answers_round_trip(engine):
    seed = seed_group(engine)
    answers = {"Why did you join?": "To grow as a leader", "Hobbies": "Guitar"}

    with unit_of_work(engine) as session:
        trainee = TraineeRepository(session).add(make_trainee(seed))
        PersonalFileRepository(session).add(PersonalFile(trainee_id=trainee.id, intake_answers=answers))

    with unit_of_work(engine) as session:
        loaded = PersonalFileRepository(session).get_by_trainee(trainee.id)

    assert loaded.intake_answers == answers  # dict -> JSON column -> dict


def test_failed_unit_of_work_saves_nothing(engine):
    """If anything inside the block raises, the whole transaction is rolled back."""
    seed = seed_group(engine)

    with pytest.raises(RuntimeError), unit_of_work(engine) as session:
        TraineeRepository(session).add(make_trainee(seed))
        raise RuntimeError("something failed halfway")

    with unit_of_work(engine) as session:
        assert TraineeRepository(session).list_by_group(seed.group_id) == []


def test_trainee_in_missing_group_is_rejected(engine):
    """The database itself enforces the foreign key, even if the code forgets to check."""
    seed = seed_group(engine)

    with pytest.raises(IntegrityError), unit_of_work(engine) as session:
        TraineeRepository(session).add(make_trainee(seed, group_id=9999))
