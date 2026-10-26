"""Endpoints under /coach: the counselor coach (stage 5).

The answer waits for models (the safety screen, the embedding model, the coach), so the endpoint uses
DetachedActor and the coach's own short transactions, never SessionDep: no transaction is held open
while a model works.
"""

from fastapi import APIRouter

from ..dependencies import CoachDep, DetachedActor
from ..schemas import CoachOut, CoachQuestion

router = APIRouter(prefix="/coach", tags=["coach"])


@router.post(
    "/ask",
    summary="AI: educational advice about a trainee, or the procedure for an exceptional incident",
    description=(
        "Whoever may see the trainee may ask (otherwise 404). The question first passes the safety gate: an "
        "exceptional incident gets the fixed procedure (outcome=referral) and no advice. Otherwise the coach "
        "reads the trainee's goals and approved notes, anonymized and without health sentences, with passages "
        "from the library, and answers in a fixed structure. Each step says whether it rests on a source "
        "(grounded) or is general advice. Omit trainee_id for a general question."
    ),
)
def ask(body: CoachQuestion, actor: DetachedActor, coach: CoachDep) -> CoachOut:
    return CoachOut.of(coach.ask(actor, body.question, body.trainee_id))
