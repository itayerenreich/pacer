"""In-memory repositories for fast service tests (no database, no Docker).

Each fake has the same methods as its Protocol in repositories/interfaces.py, so a
service cannot tell it apart from the real SQLAlchemy repository. Objects are
stored as copies, like a database would: changing an object you got back does not
change what is stored.
"""

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel

from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    ApprovalStatus,
    Cohort,
    Goal,
    GoalUpdate,
    Group,
    PersonalFile,
    StaffMember,
    StaffRole,
    Trainee,
)


class _Store[M: BaseModel]:
    """add / get / update by id, with ids handed out like a database sequence."""

    def __init__(self) -> None:
        self.rows: dict[int, M] = {}

    def add(self, item: M) -> M:
        new_id = len(self.rows) + 1
        saved = item.model_copy(update={"id": new_id})
        self.rows[new_id] = saved.model_copy()
        return saved

    def get(self, item_id: int) -> M | None:
        item = self.rows.get(item_id)
        return None if item is None else item.model_copy()

    def update(self, item: M) -> M:
        if item.id not in self.rows:
            raise LookupError(f"{type(item).__name__} {item.id} does not exist")
        self.rows[item.id] = item.model_copy()
        return item

    def where(self, **fields) -> list[M]:
        return [
            item.model_copy()
            for item in self.rows.values()
            if all(getattr(item, name) == value for name, value in fields.items())
        ]


class FakeStaffRepo(_Store[StaffMember]):
    def list_all(self) -> list[StaffMember]:
        return self.where()


class FakeCohortRepo(_Store[Cohort]):
    def list_all(self) -> list[Cohort]:
        return self.where()


class FakeGroupRepo(_Store[Group]):
    def list_by_cohort(self, cohort_id: int) -> list[Group]:
        return self.where(cohort_id=cohort_id)


class FakeTraineeRepo(_Store[Trainee]):
    def list_all(self) -> list[Trainee]:
        return self.where()

    def list_by_group(self, group_id: int) -> list[Trainee]:
        return self.where(group_id=group_id)

    def list_by_counselor(self, counselor_id: int) -> list[Trainee]:
        return self.where(primary_counselor_id=counselor_id)


class FakePersonalFileRepo(_Store[PersonalFile]):
    def get_by_trainee(self, trainee_id: int) -> PersonalFile | None:
        matches = self.where(trainee_id=trainee_id)
        return matches[0] if matches else None


class FakeGoalRepo(_Store[Goal]):
    def list_by_trainee(self, trainee_id: int) -> list[Goal]:
        return self.where(trainee_id=trainee_id)


class FakeGoalUpdateRepo(_Store[GoalUpdate]):
    def list_by_goal(self, goal_id: int) -> list[GoalUpdate]:
        return sorted(self.where(goal_id=goal_id), key=lambda u: u.created_at)


class FakeApprovalRequestRepo(_Store[ApprovalRequest]):
    def list_pending(self, requested_by_id: int | None = None) -> list[ApprovalRequest]:
        pending = self.where(status=ApprovalStatus.PENDING)
        if requested_by_id is not None:
            pending = [r for r in pending if r.requested_by_id == requested_by_id]
        return sorted(pending, key=lambda r: r.created_at)

    def exists_with_status(
        self, *, kind: ApprovalKind, trainee_id: int, requested_by_id: int, status: ApprovalStatus
    ) -> bool:
        return bool(
            self.where(kind=kind, trainee_id=trainee_id, requested_by_id=requested_by_id, status=status)
        )


@dataclass
class World:
    """A small academy for service tests.

    Group A is led by Noa and Omer; Yael is in group A and Noa is her primary counselor.
    Group B is led by Dana, who has nothing to do with Yael.
    """

    staff: FakeStaffRepo = field(default_factory=FakeStaffRepo)
    cohorts: FakeCohortRepo = field(default_factory=FakeCohortRepo)
    groups: FakeGroupRepo = field(default_factory=FakeGroupRepo)
    trainees: FakeTraineeRepo = field(default_factory=FakeTraineeRepo)
    personal_files: FakePersonalFileRepo = field(default_factory=FakePersonalFileRepo)
    goals: FakeGoalRepo = field(default_factory=FakeGoalRepo)
    goal_updates: FakeGoalUpdateRepo = field(default_factory=FakeGoalUpdateRepo)
    approvals: FakeApprovalRequestRepo = field(default_factory=FakeApprovalRequestRepo)

    def __post_init__(self) -> None:
        self.manager = self.staff.add(StaffMember(full_name="Ronit Manager", role=StaffRole.MANAGER))
        self.noa = self.staff.add(StaffMember(full_name="Noa Levi", role=StaffRole.COUNSELOR))
        self.omer = self.staff.add(StaffMember(full_name="Omer Cohen", role=StaffRole.COUNSELOR))
        self.dana = self.staff.add(StaffMember(full_name="Dana Katz", role=StaffRole.COUNSELOR))

        self.cohort = self.cohorts.add(
            Cohort(name="Mahzor 12", start_date=date(2026, 9, 1), end_date=date(2027, 6, 30))
        )
        self.group_a = self.groups.add(
            Group(name="Group A", cohort_id=self.cohort.id, counselor_ids=[self.noa.id, self.omer.id])
        )
        self.group_b = self.groups.add(
            Group(name="Group B", cohort_id=self.cohort.id, counselor_ids=[self.dana.id])
        )

        self.yael = self.trainees.add(
            Trainee(
                first_name="Yael",
                last_name="Mizrahi",
                cohort_id=self.cohort.id,
                group_id=self.group_a.id,
                primary_counselor_id=self.noa.id,
            )
        )
        self.personal_files.add(PersonalFile(trainee_id=self.yael.id, intake_answers={"Hobbies": "Guitar"}))
        self.goal = self.goals.add(Goal(trainee_id=self.yael.id, title="Run 5 km", category="physical"))
