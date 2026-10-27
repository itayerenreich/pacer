"""Endpoints under /agent: the assistant that answers by calling tools (stage 6).

Like /coach/ask, it waits for models, so it uses DetachedActor and the agent's own short transactions:
no transaction is held open while a model works. Nothing about the conversation is stored: the client
sends the history with every message (decision 2026-10-27).
"""

from fastapi import APIRouter

from pacer.ai.agent import Turn

from ..dependencies import AgentDep, DetachedActor
from ..schemas import AgentChat, AgentOut

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/chat",
    summary="AI: ask Pacer anything about your trainees, attendance, the schedule or the library",
    description=(
        "The agent decides what to look up: your trainees, a trainee's record, attendance, the week's schedule, "
        "the library, or the coach. Every tool runs with your own permissions (read only), and everything the "
        "model reads is anonymized, without health sentences. An exceptional incident gets the fixed procedure "
        "(outcome=referral). `steps` shows what was looked up; `sources` the library sources it was shown."
    ),
)
def chat(body: AgentChat, actor: DetachedActor, agent: AgentDep) -> AgentOut:
    history = [Turn(turn.role, turn.content) for turn in body.history]
    return AgentOut.of(agent.chat(actor, body.message, history))
