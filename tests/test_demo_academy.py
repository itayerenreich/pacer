"""The demo academy: right size, and every object obeys the system's rules.

Built on the in-memory fakes (no database). Demo data that breaks a business rule
would make every demo and every manual test misleading, so the rules that the
services enforce are checked here on the generated data too.
"""

from datetime import UTC, datetime

import pytest
from fakes import empty_repos

from pacer.demo import Summary, build_academy
from pacer.domain import (
    ApprovalKind,
    ApprovalStatus,
    GoalStatus,
    StaffRole,
    TraineeStatus,
)
from pacer.repositories import Repos
from pacer.services import ApprovalService, GoalService, TraineeService

AS_OF = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def fake_repos() -> Repos:
    return empty_repos()


@pytest.fixture(scope="module")
def built() -> tuple[Repos, Summary]:
    repos = fake_repos()
    return repos, build_academy(repos, as_of=AS_OF)


@pytest.fixture(scope="module")
def repos(built) -> Repos:
    return built[0]


@pytest.fixture(scope="module")
def summary(built) -> Summary:
    return built[1]


def is_hebrew(text: str) -> bool:
    return any("֐" <= char <= "׿" for char in text)


# --- size and shape ---------------------------------------------------------------


def test_staff_one_manager_four_counselors(repos):
    roles = [s.role for s in repos.staff.list_all()]
    assert roles.count(StaffRole.MANAGER) == 1
    assert roles.count(StaffRole.COUNSELOR) == 4


def test_two_groups_of_two_counselors_and_35_trainees(repos, summary):
    assert len(summary.trainees) == 35
    assert [len(g.counselor_ids) for g in summary.groups] == [2, 2]
    assert [len(repos.trainees.list_by_group(g.id)) for g in summary.groups] == [18, 17]
    assert set(summary.groups[0].counselor_ids).isdisjoint(summary.groups[1].counselor_ids)


def test_each_counselor_has_a_fair_share_of_trainees(repos, summary):
    sizes = sorted(len(repos.trainees.list_by_counselor(c.id)) for c in summary.counselors)
    assert sizes == [8, 9, 9, 9]


def test_names_are_hebrew_and_unique(summary):
    names = [t.full_name for t in summary.trainees]
    assert len(set(names)) == 35
    assert all(is_hebrew(name) for name in names)


def test_statuses_33_active_1_on_leave_1_left(summary):
    assert summary.trainee_statuses == {
        TraineeStatus.ACTIVE: 33,
        TraineeStatus.ON_LEAVE: 1,
        TraineeStatus.LEFT: 1,
    }


# --- the business rules hold on the generated data --------------------------------


def test_primary_counselor_leads_the_trainees_group(repos, summary):
    for trainee in summary.trainees:
        assert trainee.primary_counselor_id in repos.groups.get(trainee.group_id).counselor_ids


def test_every_trainee_has_an_intake_only_personal_file(repos, summary):
    for trainee in summary.trainees:
        personal_file = repos.personal_files.get_by_trainee(trainee.id)
        assert len(personal_file.intake_answers) == 4
        assert all(is_hebrew(question) for question in personal_file.intake_answers)
        assert personal_file.background_notes is None  # no background beyond the questionnaire


def test_each_trainee_has_two_to_four_goals_in_different_categories(repos, summary):
    for trainee in summary.trainees:
        goals = repos.goals.list_by_trainee(trainee.id)
        assert 2 <= len(goals) <= 4
        assert len({g.category for g in goals}) == len(goals)


def test_goal_state_matches_its_history(repos, summary):
    """The goal's progress is its last update; times rise; the author is the primary counselor."""
    for trainee in summary.trainees:
        for goal in repos.goals.list_by_trainee(trainee.id):
            history = repos.goal_updates.list_by_goal(goal.id)
            if not history:
                assert (goal.progress, goal.status) in {(0, GoalStatus.NOT_STARTED), (0, GoalStatus.DROPPED)}
                continue
            assert goal.progress == history[-1].progress
            assert [u.progress for u in history] == sorted(u.progress for u in history)
            assert goal.created_at < history[0].created_at
            assert history[-1].created_at <= AS_OF
            assert {u.author_id for u in history} == {trainee.primary_counselor_id}


def test_goal_statuses_are_varied(repos, summary):
    statuses = {g.status for t in summary.trainees for g in repos.goals.list_by_trainee(t.id)}
    assert statuses == set(GoalStatus)


def test_trainee_who_left_has_no_open_goals_and_no_late_updates(repos, summary):
    [left] = [t for t in summary.trainees if t.status is TraineeStatus.LEFT]
    [request] = [r for r in summary.approvals if r.requested_status is TraineeStatus.LEFT]
    for goal in repos.goals.list_by_trainee(left.id):
        assert goal.status in (GoalStatus.ACHIEVED, GoalStatus.DROPPED)
        assert all(u.created_at <= request.created_at for u in repos.goal_updates.list_by_goal(goal.id))


def test_approvals_match_the_trainees(repos, summary):
    by_status = {s: [r for r in summary.approvals if r.status is s] for s in ApprovalStatus}
    assert [len(by_status[s]) for s in ApprovalStatus] == [2, 3, 1]  # pending, approved, rejected

    for request in summary.approvals:
        trainee = repos.trainees.get(request.trainee_id)
        assert request.requested_by_id in repos.groups.get(trainee.group_id).counselor_ids
        assert request.created_at <= AS_OF
        if request.kind is ApprovalKind.STATUS_CHANGE and request.status is ApprovalStatus.APPROVED:
            assert trainee.status is request.requested_status


# --- the services work on it ------------------------------------------------------


def test_services_run_on_the_demo_data(repos, summary):
    manager = summary.manager
    goals = GoalService(repos.goals, repos.goal_updates, repos.trainees, repos.groups)
    approvals = ApprovalService(repos.approvals, repos.trainees, repos.groups)
    trainees = TraineeService(repos.trainees, repos.groups, repos.personal_files, repos.approvals)

    assert len(approvals.list_pending(manager)) == 2

    [viewer] = [
        r
        for r in summary.approvals
        if r.kind is ApprovalKind.VIEW_PERSONAL_FILE and r.status is ApprovalStatus.APPROVED
    ]
    counselor = repos.staff.get(viewer.requested_by_id)
    assert trainees.get_personal_file(counselor, viewer.trainee_id).intake_answers

    active = next(t for t in summary.trainees if t.is_active)
    goal = next(g for g in repos.goals.list_by_trainee(active.id) if g.status is not GoalStatus.DROPPED)
    author = repos.staff.get(active.primary_counselor_id)
    assert goals.update_progress(author, goal.id, 100, "Demo check").status is GoalStatus.ACHIEVED


# --- determinism ------------------------------------------------------------------


def test_same_seed_same_academy_other_seed_different():
    def names(seed: int) -> list[str]:
        return [t.full_name for t in build_academy(fake_repos(), seed=seed, as_of=AS_OF).trainees]

    assert names(2026) == names(2026)
    assert names(2026) != names(7)


def test_too_early_as_of_is_rejected():
    with pytest.raises(ValueError, match="3 weeks"):
        build_academy(fake_repos(), as_of=datetime(2026, 9, 10, tzinfo=UTC))


def test_a_few_trainees_have_approved_notes_for_the_coach():
    """The coach builds on what worked with a trainee: a few demo trainees have approved notes, written
    by their primary counselor, about education only (no health word, nothing the safety gate would stop)."""
    from pacer.ai.safety import check_safety
    from pacer.ai.sensitive import is_sensitive
    from pacer.demo import content

    repos = empty_repos()
    summary = build_academy(repos, as_of=datetime(2026, 10, 25, tzinfo=UTC))
    by_name = {t.full_name: t for t in summary.trainees}
    assert set(content.TRAINEE_NOTES) <= set(by_name)  # every name is a real demo trainee
    for name, expected in content.TRAINEE_NOTES.items():
        notes = repos.notes.list_by_trainee(by_name[name].key)
        assert len(notes) == len(expected)
        for note in notes:
            assert note.author_id == by_name[name].primary_counselor_id
            assert not is_sensitive(note.text) and not check_safety(note.text).is_risk, note.text
    assert summary.note_count == sum(len(v) for v in content.TRAINEE_NOTES.values())
