"""Builds a complete demo academy: one cohort, 2 groups, a manager, 4 counselors,
35 trainees with personal files, goals with progress history, and approval requests.

Design:
* Deterministic: random.Random(seed) makes the same academy every time, so tests can
  rely on it and a demo looks the same on every machine.
* Written through the repository Protocols, not SQLAlchemy directly. The same code
  fills the in-memory fakes in fast tests and PostgreSQL for real.
* Dated history: goals and updates are spread over the weeks since the cohort began,
  up to `as_of`, so the history looks like real use. Every object still passes the
  domain validation (achieved = 100%, aware UTC times, field lengths...).
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    ApprovalStatus,
    Cohort,
    Goal,
    GoalCategory,
    GoalStatus,
    GoalUpdate,
    Group,
    PersonalFile,
    StaffMember,
    StaffRole,
    Trainee,
    TraineeStatus,
)
from pacer.repositories import Repos

from . import content

TRAINEE_COUNT = 35
COHORT_START = date(2026, 9, 1)
COHORT_END = date(2027, 6, 30)
FIRST_DAY = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


@dataclass
class Summary:
    """What was created, and the ids someone needs to try the API (X-Staff-Id)."""

    manager: StaffMember
    counselors: list[StaffMember]
    groups: list[Group]
    trainees: list[Trainee]
    goal_count: int = 0
    update_count: int = 0
    approvals: list[ApprovalRequest] = field(default_factory=list)

    @property
    def trainee_statuses(self) -> Counter:
        return Counter(t.status for t in self.trainees)


def build_academy(repos: Repos, *, seed: int = 2026, as_of: datetime | None = None) -> Summary:
    """Create the whole academy through `repos`. The caller owns the unit of work."""
    return _Builder(repos, random.Random(seed), as_of or datetime.now(UTC)).build()


class _Builder:
    def __init__(self, repos: Repos, rng: random.Random, as_of: datetime) -> None:
        if as_of < FIRST_DAY + timedelta(days=21):
            raise ValueError(f"as_of must be at least 3 weeks after {FIRST_DAY:%Y-%m-%d}")
        self.r = repos
        self.rng = rng
        self.now = min(as_of, datetime.combine(COHORT_END, datetime.min.time(), UTC))
        self.history_ends: dict[int, datetime] = {}  # trainees whose story stops early (left)

    def build(self) -> Summary:
        cohort = self.r.cohorts.add(
            Cohort(name=content.COHORT_NAME, start_date=COHORT_START, end_date=COHORT_END)
        )
        manager = self.r.staff.add(StaffMember(full_name=content.MANAGER_NAME, role=StaffRole.MANAGER))
        counselors = [
            self.r.staff.add(StaffMember(full_name=name, role=StaffRole.COUNSELOR, responsibility_area=area))
            for name, area in content.COUNSELORS
        ]
        groups = [
            self.r.groups.add(
                Group(
                    name=name,
                    cohort_id=cohort.id,
                    counselor_ids=[c.id for c in counselors[2 * i : 2 * i + 2]],
                )
            )
            for i, name in enumerate(content.GROUP_NAMES)
        ]
        summary = Summary(manager=manager, counselors=counselors, groups=groups, trainees=[])

        for index, (first, last) in enumerate(self._unique_names(TRAINEE_COUNT)):
            group = groups[0] if index < (TRAINEE_COUNT + 1) // 2 else groups[1]
            counselor_id = group.counselor_ids[index % 2]  # alternate between the group's 2 counselors
            trainee = self.r.trainees.add(
                Trainee(
                    first_name=first,
                    last_name=last,
                    cohort_id=cohort.id,
                    group_id=group.id,
                    primary_counselor_id=counselor_id,
                )
            )
            self.r.personal_files.add(PersonalFile(trainee_id=trainee.id, intake_answers=self._intake()))
            summary.trainees.append(trainee)

        requests = self._plan_approvals(summary, groups)
        for trainee in summary.trainees:
            self._add_goals(trainee, summary)
        self._add_approvals(summary, requests)
        return summary

    # --- trainees ---------------------------------------------------------------------

    def _unique_names(self, count: int) -> list[tuple[str, str]]:
        names: list[tuple[str, str]] = []
        while len(names) < count:
            name = (self.rng.choice(content.FIRST_NAMES), self.rng.choice(content.LAST_NAMES))
            if name not in names:
                names.append(name)
        return names

    def _intake(self) -> dict[str, str]:
        answers = {
            question: self.rng.choice(options)
            for question, options in content.INTAKE_QUESTIONS.items()
            if question != "תחביבים"
        }
        answers["תחביבים"] = ", ".join(self.rng.sample(content.INTAKE_QUESTIONS["תחביבים"], 2))
        return answers

    # --- goals and their history ------------------------------------------------------

    def _add_goals(self, trainee: Trainee, summary: Summary) -> None:
        categories = self.rng.sample(list(GoalCategory), self.rng.randint(2, 4))
        for category in categories:
            created = FIRST_DAY + timedelta(days=self.rng.randint(2, 14), hours=self.rng.randint(0, 9))
            outcome = self.rng.choices(
                ["achieved", "in_progress", "not_started", "dropped"], weights=[20, 60, 12, 8]
            )[0]
            steps = self._progress_steps(outcome)
            end = self.history_ends.get(trainee.id, self.now)
            times = self._update_times(created, end, len(steps))

            new_goal = Goal(
                trainee_id=trainee.id,
                title=self.rng.choice(content.GOALS[category]),
                category=category,
                target_date=self._target_date(),
                created_at=created,
            )
            # The final state, built the same way a service builds it: with_progress picks the status.
            final = new_goal.with_progress(steps[-1]) if steps else new_goal
            if outcome == "dropped":
                final = Goal.model_validate({**final.model_dump(), "status": GoalStatus.DROPPED})
            goal = self.r.goals.add(final)
            summary.goal_count += 1

            for i, (value, when) in enumerate(zip(steps, times, strict=True)):
                last = i == len(steps) - 1
                note = self._note(value, dropped=(last and outcome == "dropped"))
                self.r.goal_updates.add(
                    GoalUpdate(
                        goal_id=goal.id,
                        author_id=trainee.primary_counselor_id,
                        progress=value,
                        note=note,
                        created_at=when,
                    )
                )
                summary.update_count += 1

    def _progress_steps(self, outcome: str) -> list[int]:
        """The progress value of each update, rising toward the outcome's final value."""
        if outcome == "not_started":
            return []
        final = {
            "achieved": 100,
            "in_progress": self.rng.randrange(20, 95, 5),
            "dropped": self.rng.randrange(10, 50, 5),
        }[outcome]
        count = self.rng.randint(2, 4) if final >= 40 else self.rng.randint(1, 2)
        middle = sorted(self.rng.sample(range(5, final, 5), min(count - 1, len(range(5, final, 5)))))
        return [*middle, final]

    def _update_times(self, created: datetime, end: datetime, count: int) -> list[datetime]:
        """`count` increasing moments between the goal's creation and `end`."""
        span = (end - created).total_seconds()
        offsets = sorted(self.rng.uniform(0.1, 0.98) * span for _ in range(count))
        return [created + timedelta(seconds=offset) for offset in offsets]

    def _target_date(self) -> date | None:
        if self.rng.random() < 0.3:
            return None
        return date(2026, 12, 1) + timedelta(days=self.rng.randint(0, 150))

    def _note(self, progress: int, *, dropped: bool) -> str:
        if dropped:
            return content.NOTE_DROPPED
        if progress == 100:
            return self.rng.choice(content.NOTES_DONE)
        return self.rng.choice(content.NOTES_EARLY if progress < 40 else content.NOTES_MIDDLE)

    # --- approvals and the statuses they changed ----------------------------------------

    def _plan_approvals(self, summary: Summary, groups: list[Group]) -> list[tuple]:
        """Who asks for what. Planned before the goals, because a trainee who left
        must not get progress updates dated after the day they left."""
        noa, omer, shira, itay = summary.counselors
        group_a = [t for t in summary.trainees if t.group_id == groups[0].id]
        group_b = [t for t in summary.trainees if t.group_id == groups[1].id]

        def of(counselor: StaffMember, trainees: list[Trainee], skip: int = 0) -> Trainee:
            return [t for t in trainees if t.primary_counselor_id == counselor.id][skip]

        def days_ago(days: float) -> datetime:
            return self.now - timedelta(days=days)

        leaving, sent_home = of(omer, group_a), of(itay, group_b)
        requests = [
            # Approved status changes: the trainees' statuses really change below.
            (
                omer,
                leaving,
                ApprovalKind.STATUS_CHANGE,
                TraineeStatus.LEFT,
                "החלטה לעזוב את המכינה, אחרי שיחה עם ההורים ועם הצוות",
                ApprovalStatus.APPROVED,
                "אחרי שיחה משותפת ההחלטה התקבלה. נלווה את הפרידה מהקבוצה",
                9,
            ),
            (
                itay,
                sent_home,
                ApprovalKind.STATUS_CHANGE,
                TraineeStatus.ON_LEAVE,
                "שליחה הביתה לשלושה ימים, בעקבות הפרת נהלים חוזרת",
                ApprovalStatus.APPROVED,
                "מאושר. שיחת חזרה עם המדריך ביום ראשון",
                2,
            ),
            # Personal file requests: approved, rejected, pending.
            (
                shira,
                of(shira, group_b),
                ApprovalKind.VIEW_PERSONAL_FILE,
                None,
                "שיחה אישית ראשונה, רוצה להכיר את התשובות מהשאלון",
                ApprovalStatus.APPROVED,
                "מאושר לצורך השיחה",
                6,
            ),
            (
                noa,
                of(noa, group_a, 1),
                ApprovalKind.VIEW_PERSONAL_FILE,
                None,
                "רוצה לקרוא את השאלון לפני שיחת אמצע שנה",
                ApprovalStatus.REJECTED,
                "אין צורך כרגע, אפשר לשאול ישירות בשיחה",
                4,
            ),
            (
                noa,
                of(noa, group_a, 2),
                ApprovalKind.VIEW_PERSONAL_FILE,
                None,
                "קושי חברתי בקבוצה, רוצה להבין את הרקע לפני שיחה",
                ApprovalStatus.PENDING,
                None,
                1,
            ),
            (
                itay,
                of(itay, group_b, 1),
                ApprovalKind.STATUS_CHANGE,
                TraineeStatus.ON_LEAVE,
                "בקשה ליציאה לשבוע בגלל אירוע משפחתי",
                ApprovalStatus.PENDING,
                None,
                0.5,
            ),
        ]
        self.history_ends[leaving.id] = days_ago(9)  # the day the request to leave was made
        return requests

    def _add_approvals(self, summary: Summary, requests: list[tuple]) -> None:
        manager_id = summary.manager.id
        for counselor, trainee, kind, requested, reason, status, note, age in requests:
            created = self.now - timedelta(days=age)
            decided = status != ApprovalStatus.PENDING
            request = self.r.approvals.add(
                ApprovalRequest(
                    kind=kind,
                    trainee_id=trainee.id,
                    requested_by_id=counselor.id,
                    reason=reason,
                    requested_status=requested,
                    status=status,
                    decided_by_id=manager_id if decided else None,
                    decision_note=note,
                    created_at=created,
                    decided_at=created + timedelta(hours=20) if decided else None,
                )
            )
            summary.approvals.append(request)
            if decided and kind == ApprovalKind.STATUS_CHANGE and status == ApprovalStatus.APPROVED:
                self._apply_status(summary, trainee, requested)

    def _apply_status(self, summary: Summary, trainee: Trainee, status: TraineeStatus) -> None:
        updated = self.r.trainees.update(trainee.with_status(status))
        summary.trainees[[t.id for t in summary.trainees].index(trainee.id)] = updated
        if status == TraineeStatus.LEFT:  # unfinished goals of a trainee who left are dropped
            for goal in self.r.goals.list_by_trainee(trainee.id):
                if goal.status in (GoalStatus.NOT_STARTED, GoalStatus.IN_PROGRESS):
                    self.r.goals.update(goal.model_copy(update={"status": GoalStatus.DROPPED}))
