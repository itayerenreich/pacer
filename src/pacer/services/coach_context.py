"""What the counselor coach knows about a trainee: built in code, from approved data only.

    context = CoachContextService(...).build(actor, trainee_id)

Decisions (2026-10-25):
* Built in CODE, not by a model summarizing (a 4B local model failed that in stage 4: 50% of its claims
  supported, a brother became a sister). What goes in is already curated by people, so nothing needs
  "summarizing": it needs choosing.
* Who may ask: whoever may SEE the trainee (the group's counselors and the manager). Anyone else gets
  404, exactly as if the trainee did not exist (visibility.py).
* What goes in: the goals, the last UPDATES_PER_GOAL progress updates of each, and the last NOTES_LIMIT
  approved notes (TraineeNote: what worked, what did not, observations, follow-ups). "What worked with
  THIS trainee" is what makes the advice personal.
* What NEVER goes in: the personal file, the raw notes of personal talks, anything not approved by a
  person. And nothing about health: the AI layer removes such sentences before anything is sent.

This is a service, so it knows nothing about models or text formats: it returns domain objects, and
the AI layer (ai/coach.py) renders and anonymizes them.
"""

from dataclasses import dataclass

from pacer.domain import Goal, GoalUpdate, StaffMember, Trainee, TraineeNote
from pacer.repositories.interfaces import GoalRepo, GoalUpdateRepo, GroupRepo, TraineeNoteRepo, TraineeRepo

from .visibility import visible_trainee

NOTES_LIMIT = 10  # the most recent approved notes
UPDATES_PER_GOAL = 3  # the most recent progress updates of each goal


@dataclass(frozen=True)
class GoalContext:
    goal: Goal
    recent_updates: list[GoalUpdate]  # oldest first, at most UPDATES_PER_GOAL


@dataclass(frozen=True)
class TraineeContext:
    trainee: Trainee
    goals: list[GoalContext]
    notes: list[TraineeNote]  # oldest first, at most NOTES_LIMIT


class CoachContextService:
    def __init__(
        self,
        trainees: TraineeRepo,
        groups: GroupRepo,
        goals: GoalRepo,
        goal_updates: GoalUpdateRepo,
        notes: TraineeNoteRepo,
    ) -> None:
        self._trainees = trainees
        self._groups = groups
        self._goals = goals
        self._goal_updates = goal_updates
        self._notes = notes

    def build(self, actor: StaffMember, trainee_id: int) -> TraineeContext:
        trainee = visible_trainee(actor, self._trainees, self._groups, trainee_id)
        goals = [
            GoalContext(goal, self._goal_updates.list_by_goal(goal.key)[-UPDATES_PER_GOAL:])
            for goal in self._goals.list_by_trainee(trainee.key)
        ]
        notes = self._notes.list_by_trainee(trainee.key)[-NOTES_LIMIT:]
        return TraineeContext(trainee=trainee, goals=goals, notes=notes)
