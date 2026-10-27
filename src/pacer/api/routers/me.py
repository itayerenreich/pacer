"""GET /me: who the API thinks you are (handy for checking the X-Staff-Id header); GET /groups and
GET /staff: the academy's groups and staff, so a client can show names instead of ids (and the manager
can choose whom a task is for). Every staff member may see all three: colleagues, not trainees."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pacer.domain import Group, StaffMember
from pacer.repositories.interfaces import CohortRepo, GroupRepo, StaffRepo

from ..dependencies import Actor, get_cohort_repository, get_group_repository, get_staff_repository

router = APIRouter(tags=["me"])


@router.get("/me", summary="The staff member making this request")
def who_am_i(actor: Actor) -> StaffMember:
    return actor


@router.get("/groups", summary="Every group of every cohort, with its counselors' ids")
def list_groups(
    actor: Actor,
    cohorts: Annotated[CohortRepo, Depends(get_cohort_repository)],
    groups: Annotated[GroupRepo, Depends(get_group_repository)],
) -> list[Group]:
    return [g for c in cohorts.list_all() for g in groups.list_by_cohort(c.key)]


@router.get("/staff", summary="Every staff member")
def list_staff(actor: Actor, staff: Annotated[StaffRepo, Depends(get_staff_repository)]) -> list[StaffMember]:
    return staff.list_all()
