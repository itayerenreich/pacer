"""The FastAPI application.

Run it:
    docker compose up -d                       # PostgreSQL
    uv run alembic upgrade head                # tables
    uv run uvicorn pacer.api.app:app --reload  # http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI

from pacer.logging_config import configure_logging

from .errors import register_error_handlers
from .routers import approvals, goals, me, meetings, talks, tasks, trainees

DESCRIPTION = """
Staff API for a pre-military academy: trainees, goals and progress, approvals, staff meetings and tasks,
personal talks and trainee notes.

**Who are you?** Send the `X-Staff-Id` header with a staff member's id
(development only, see `api/dependencies.py`). In these docs: fill it in per request.
"""


def create_app() -> FastAPI:
    """Build the app. A function, so tests can build a fresh app with their own overrides."""
    configure_logging()
    app = FastAPI(title="Pacer API", version="0.1.0", description=DESCRIPTION)
    register_error_handlers(app)
    for module in (me, trainees, goals, approvals, meetings, tasks, talks):
        app.include_router(module.router)
    return app


app = create_app()
