"""Persistence for personal talks, their proposals, and trainee notes."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from pacer.domain import PersonalTalk, ProposalStatus, TalkProposal, TraineeNote

from ._mapping import row_values
from .tables import PersonalTalkRow, TalkProposalRow, TraineeNoteRow


class PersonalTalkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, talk: PersonalTalk) -> PersonalTalk:
        row = PersonalTalkRow(**row_values(talk))
        self._session.add(row)
        self._session.flush()
        return talk.model_copy(update={"id": row.id})

    def get(self, talk_id: int) -> PersonalTalk | None:
        row = self._session.get(PersonalTalkRow, talk_id)
        return None if row is None else PersonalTalk.model_validate(row, from_attributes=True)

    def lock(self, talk_id: int) -> PersonalTalk | None:
        """SELECT ... FOR UPDATE (see MeetingRepository.lock)."""
        row = self._session.scalars(
            select(PersonalTalkRow)
            .where(PersonalTalkRow.id == talk_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()
        return None if row is None else PersonalTalk.model_validate(row, from_attributes=True)

    def update(self, talk: PersonalTalk) -> PersonalTalk:
        row = self._session.get(PersonalTalkRow, talk.id)
        if row is None:
            raise LookupError(f"Talk {talk.id} does not exist")
        for column, value in row_values(talk).items():
            setattr(row, column, value)
        self._session.flush()
        return talk

    def list_by_trainee(self, trainee_id: int) -> list[PersonalTalk]:
        rows = self._session.scalars(
            select(PersonalTalkRow)
            .where(PersonalTalkRow.trainee_id == trainee_id)
            .order_by(PersonalTalkRow.held_on.desc(), PersonalTalkRow.id.desc())
        )
        return [PersonalTalk.model_validate(row, from_attributes=True) for row in rows]


    def delete_with_proposals(self, talk_id: int) -> None:
        """The talk and every proposal made from it. Notes and progress approved from it stay
        (they point to it with ON DELETE SET NULL)."""
        self._session.execute(delete(TalkProposalRow).where(TalkProposalRow.talk_id == talk_id))
        self._session.execute(delete(PersonalTalkRow).where(PersonalTalkRow.id == talk_id))
        self._session.flush()


class TalkProposalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, proposal: TalkProposal) -> TalkProposal:
        row = TalkProposalRow(**row_values(proposal))
        self._session.add(row)
        self._session.flush()
        return proposal.model_copy(update={"id": row.id})

    def get(self, proposal_id: int) -> TalkProposal | None:
        row = self._session.get(TalkProposalRow, proposal_id)
        return None if row is None else TalkProposal.model_validate(row, from_attributes=True)

    def lock(self, proposal_id: int) -> TalkProposal | None:
        """SELECT ... FOR UPDATE, re-read even if this session saw the row before."""
        row = self._session.scalars(
            select(TalkProposalRow)
            .where(TalkProposalRow.id == proposal_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()
        return None if row is None else TalkProposal.model_validate(row, from_attributes=True)

    def update(self, proposal: TalkProposal) -> TalkProposal:
        row = self._session.get(TalkProposalRow, proposal.id)
        if row is None:
            raise LookupError(f"Talk proposal {proposal.id} does not exist")
        for column, value in row_values(proposal).items():
            setattr(row, column, value)
        self._session.flush()
        return proposal

    def list_by_talk(self, talk_id: int) -> list[TalkProposal]:
        rows = self._session.scalars(
            select(TalkProposalRow).where(TalkProposalRow.talk_id == talk_id).order_by(TalkProposalRow.id)
        )
        return [TalkProposal.model_validate(row, from_attributes=True) for row in rows]

    def count_open(self, talk_id: int) -> int:
        return len(
            self._session.scalars(
                select(TalkProposalRow.id).where(
                    TalkProposalRow.talk_id == talk_id, TalkProposalRow.status == ProposalStatus.PROPOSED.value
                )
            ).all()
        )


class TraineeNoteRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, note: TraineeNote) -> TraineeNote:
        row = TraineeNoteRow(**row_values(note))
        self._session.add(row)
        self._session.flush()
        return note.model_copy(update={"id": row.id})

    def get(self, note_id: int) -> TraineeNote | None:
        row = self._session.get(TraineeNoteRow, note_id)
        return None if row is None else TraineeNote.model_validate(row, from_attributes=True)

    def delete(self, note_id: int) -> None:
        """Really deleted, not hidden: the proposal it came from keeps no copy (note_id becomes NULL)."""
        self._session.execute(delete(TraineeNoteRow).where(TraineeNoteRow.id == note_id))
        self._session.flush()

    def list_by_trainee(self, trainee_id: int) -> list[TraineeNote]:
        rows = self._session.scalars(
            select(TraineeNoteRow)
            .where(TraineeNoteRow.trainee_id == trainee_id)
            .order_by(TraineeNoteRow.created_at, TraineeNoteRow.id)
        )
        return [TraineeNote.model_validate(row, from_attributes=True) for row in rows]
