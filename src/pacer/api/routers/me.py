"""GET /me: who the API thinks you are. Handy for checking the X-Staff-Id header."""

from fastapi import APIRouter

from pacer.domain import StaffMember

from ..dependencies import Actor

router = APIRouter(tags=["me"])


@router.get("/me", summary="The staff member making this request")
def who_am_i(actor: Actor) -> StaffMember:
    return actor
