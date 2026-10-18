"""Endpoints under /library: questions answered from the academy's knowledge library.

The answer waits for two models (the embedding model on this computer, then Claude), so the endpoint
uses DetachedActor and the answerer's own short transactions, never SessionDep: no transaction is held
open while a model works.
"""

from fastapi import APIRouter

from ..dependencies import DetachedActor, LibraryAnswererDep
from ..schemas import AskOut, LibraryQuestion

router = APIRouter(prefix="/library", tags=["library"])


@router.post(
    "/ask",
    summary="AI: answer a question only from the library, with a citation for every claim (any staff member)",
    description=(
        "Retrieves the passages closest in meaning to the question (on this computer), then Claude answers "
        "from them alone. Names in the question are replaced by tokens before anything is sent. With no "
        "relevant passage, no model is asked and the answer says the library has no source."
    ),
)
def ask(body: LibraryQuestion, actor: DetachedActor, answerer: LibraryAnswererDep) -> AskOut:
    return AskOut.of(answerer.ask(body.question))
