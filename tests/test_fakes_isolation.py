"""The fakes behave like a database where it matters: a caller cannot change a stored row by
editing what it got back, a rollback restores everything, and ids are never reused."""

from fakes import World


def test_editing_a_returned_list_does_not_change_the_store():
    world = World()
    group = world.groups.get(world.group_a.key)
    group.counselor_ids.append(999)
    assert 999 not in world.groups.get(world.group_a.key).counselor_ids


def test_a_rollback_restores_a_row_whose_list_was_edited_in_place():
    world = World()
    try:
        with world.uow() as repos:
            group = repos.groups.get(world.group_a.key)
            repos.groups.update(group.model_copy(update={"counselor_ids": [*group.counselor_ids, world.dana.key]}))
            raise RuntimeError("roll back")
    except RuntimeError:
        pass
    assert world.dana.key not in world.groups.get(world.group_a.key).counselor_ids


def test_an_id_is_never_reused_after_a_delete():
    from pacer.domain import NoteKind, TraineeNote

    world = World()
    first = world.notes.add(TraineeNote(trainee_id=world.yael.key, kind=NoteKind.WORKED, text="ראשונה", author_id=world.noa.key))
    second = world.notes.add(TraineeNote(trainee_id=world.yael.key, kind=NoteKind.WORKED, text="שנייה", author_id=world.noa.key))
    world.notes.delete(first.key)
    third = world.notes.add(TraineeNote(trainee_id=world.yael.key, kind=NoteKind.WORKED, text="שלישית", author_id=world.noa.key))
    assert third.key not in (first.key, second.key)
    assert world.notes.get(second.key).text == "שנייה"
