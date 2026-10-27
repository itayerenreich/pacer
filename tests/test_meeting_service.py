"""MeetingService and TaskService with in-memory fakes: the AI proposes, the manager decides."""

from datetime import date

import pytest
from fakes import World

from pacer.domain import InvalidTransitionError, MeetingStatus, ProposalStatus, TaskStatus
from pacer.services import (
    MeetingService,
    NotFoundError,
    PermissionDeniedError,
    ProposalEdits,
    RuleViolationError,
    TaskDraft,
    TaskService,
)

TRANSCRIPT = "נועה: נתחיל. עומר: צריך לתאם הסעה לטיול ולהזמין ציוד."
HELD_ON = date(2026, 10, 12)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> MeetingService:
    return MeetingService(world.meetings, world.proposals, world.tasks, world.staff)


@pytest.fixture
def tasks(world) -> TaskService:
    return TaskService(world.tasks, world.staff, world.trainees, world.groups)


def new_meeting(world, service):
    return service.create_meeting(
        world.manager,
        title="ישיבת צוות",
        held_on=HELD_ON,
        participant_ids=[world.manager.id, world.noa.id, world.omer.id],
        transcript=TRANSCRIPT,
    )


def extracted(world, service, *drafts: TaskDraft):
    """What the pipeline (task 6) will do: anonymize, extract, then submit validated drafts."""
    meeting = new_meeting(world, service)
    meeting = meeting.advance(MeetingStatus.ANONYMIZED, anonymized_transcript="[STAFF_2]: נתחיל")
    meeting = world.meetings.update(meeting.advance(MeetingStatus.EXTRACTED, summary="סוכם לתאם הסעה"))
    return meeting, service.submit_proposals(meeting.id, list(drafts))


TRANSPORT = TaskDraft(description="לתאם הסעה לטיול", owner_id=3, due_date=date(2026, 10, 20))
EQUIPMENT = TaskDraft(description="להזמין ציוד לטיול", owner_id=None)  # the model named no one


# --- creating meetings ------------------------------------------------------------------------


def test_a_text_meeting_starts_transcribed(world, service):
    assert new_meeting(world, service).status is MeetingStatus.TRANSCRIBED


def test_a_meeting_starts_transcribed(world, service):
    assert new_meeting(world, service).status is MeetingStatus.TRANSCRIBED


def test_only_the_manager_records_meetings_and_reads_transcripts(world, service):
    with pytest.raises(PermissionDeniedError):
        service.create_meeting(
            world.noa, title="ישיבה", held_on=HELD_ON, participant_ids=[world.noa.id], transcript=TRANSCRIPT
        )
    meeting = new_meeting(world, service)
    with pytest.raises(PermissionDeniedError):
        service.get_meeting(world.noa, meeting.id)


def test_participants_must_exist(world, service):
    with pytest.raises(NotFoundError, match="Staff member 999"):
        service.create_meeting(
            world.manager, title="ישיבה", held_on=HELD_ON, participant_ids=[999], transcript=TRANSCRIPT
        )


# --- proposals in -------------------------------------------------------------------------------


def test_submitted_proposals_wait_for_the_manager(world, service):
    meeting, stored = extracted(world, service, TRANSPORT, EQUIPMENT)

    assert [p.status for p in stored] == [ProposalStatus.PROPOSED] * 2
    assert world.meetings.get(meeting.id).status is MeetingStatus.PENDING_APPROVAL
    assert world.tasks.list_all() == []  # nothing is a real task yet


def test_proposals_only_after_extraction(world, service):
    meeting = new_meeting(world, service)  # still TRANSCRIBED
    with pytest.raises(RuleViolationError, match="after extraction"):
        service.submit_proposals(meeting.id, [TRANSPORT])


def test_a_meeting_with_no_proposals_closes_itself(world, service):
    meeting, _ = extracted(world, service)
    assert world.meetings.get(meeting.id).status is MeetingStatus.APPROVED


# --- the manager decides --------------------------------------------------------------------------


def test_approving_creates_the_real_task(world, service):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    task = service.approve_proposal(world.manager, transport.id)

    assert (task.owner_id, task.created_by_id, task.status) == (3, world.manager.id, TaskStatus.OPEN)
    assert world.proposals.get(transport.id).task_id == task.id


def test_the_manager_can_edit_before_approving(world, service):
    _, [_, equipment] = extracted(world, service, TRANSPORT, EQUIPMENT)
    task = service.approve_proposal(
        world.manager,
        equipment.id,
        ProposalEdits(owner_id=world.omer.id, description="להזמין ציוד שטח לטיול"),
    )
    assert (task.owner_id, task.description) == (world.omer.id, "להזמין ציוד שטח לטיול")


def test_no_owner_means_no_approval_until_one_is_chosen(world, service):
    _, [_, equipment] = extracted(world, service, TRANSPORT, EQUIPMENT)
    with pytest.raises(RuleViolationError, match="Choose an owner"):
        service.approve_proposal(world.manager, equipment.id)


def test_a_due_date_before_the_meeting_is_rejected(world, service):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    with pytest.raises(RuleViolationError, match="before the meeting"):
        service.approve_proposal(world.manager, transport.id, ProposalEdits(due_date=date(2026, 10, 1)))


def test_an_owner_who_does_not_exist_is_rejected(world, service):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    with pytest.raises(NotFoundError):
        service.approve_proposal(world.manager, transport.id, ProposalEdits(owner_id=999))


def test_approving_twice_creates_one_task_only(world, service):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    service.approve_proposal(world.manager, transport.id)
    with pytest.raises(InvalidTransitionError):
        service.approve_proposal(world.manager, transport.id)
    assert len(world.tasks.list_all()) == 1


def test_only_the_manager_decides(world, service):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    with pytest.raises(PermissionDeniedError):
        service.approve_proposal(world.noa, transport.id)
    with pytest.raises(PermissionDeniedError):
        service.reject_proposal(world.omer, transport.id)
    with pytest.raises(PermissionDeniedError):
        service.list_proposals(world.noa, transport.meeting_id)


def test_the_meeting_closes_when_every_proposal_is_decided(world, service):
    meeting, [transport, equipment] = extracted(world, service, TRANSPORT, EQUIPMENT)
    service.approve_proposal(world.manager, transport.id)
    assert world.meetings.get(meeting.id).status is MeetingStatus.PENDING_APPROVAL  # one still open

    service.reject_proposal(world.manager, equipment.id)
    assert world.meetings.get(meeting.id).status is MeetingStatus.APPROVED


# --- tasks ---------------------------------------------------------------------------------------


def test_each_staff_member_sees_their_own_tasks_and_completes_them(world, service, tasks):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    task = service.approve_proposal(world.manager, transport.id)  # owner 3 = Omer

    assert [t.id for t in tasks.my_tasks(world.omer)] == [task.id]
    assert tasks.my_tasks(world.noa) == []
    assert tasks.complete(world.omer, task.id).status is TaskStatus.DONE


def test_only_the_owner_or_the_manager_completes_a_task(world, service, tasks):
    _, [transport, _] = extracted(world, service, TRANSPORT, EQUIPMENT)
    task = service.approve_proposal(world.manager, transport.id)
    with pytest.raises(NotFoundError):  # someone else's task does not exist for a counselor
        tasks.complete(world.noa, task.id)
    assert tasks.complete(world.manager, task.id).status is TaskStatus.DONE


def test_only_the_manager_sees_everyones_tasks(world, tasks):
    with pytest.raises(PermissionDeniedError):
        tasks.all_tasks(world.noa)
    assert tasks.all_tasks(world.manager) == []


def test_the_meeting_also_closes_when_the_last_decision_is_an_approval(world, service):
    meeting, [transport] = extracted(world, service, TRANSPORT)
    service.approve_proposal(world.manager, transport.id)
    assert world.meetings.get(meeting.id).status is MeetingStatus.APPROVED
