"""GET /me: who the API thinks you are (handy for checking the X-Staff-Id header), and GET /groups:
the academy's groups, so a client can show names instead of ids. Every staff member may see both."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pacer.domain import Group, StaffMember
from pacer.repositories.interfaces import CohortRepo, GroupRepo

from ..dependencies import Actor, get_cohort_repository, get_group_repository

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
