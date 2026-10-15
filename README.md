# Pacer

[![tests](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml/badge.svg)](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml)

A backend for the staff of a pre-military academy (מכינה קדם-צבאית): counselors and a
manager track trainees, the personal goals they set together, and progress over time.
Sensitive actions, like opening a trainee's personal file or sending a trainee home,
go through manager approval.

On top of this data layer sit AI features: staff meetings (a recording or a transcript)
become a summary and proposed tasks, and a counselor's notes from a personal talk become a
summary, goal progress updates and notes worth remembering. The AI sits **above** the data,
never instead of it: it proposes, and a person approves every item before anything changes.

## Architecture

```
  HTTP clients (/docs, UI, agent)
          │
     api/            FastAPI: routing, request schemas, who is asking, errors -> HTTP codes
          │
     ai/             model routing, anonymization, structured output, the meeting and talk pipelines
          │
     services/       business rules and permissions (role-based), one unit of work per request
          │
     repositories/   SQLAlchemy persistence behind Protocol interfaces
          │
     domain/         Pydantic models and single-object rules; imports nothing from other layers
          │
     PostgreSQL 17 (Docker), schema managed by Alembic migrations
```

Some of the decisions behind it:

- **Domain and tables are separate classes.** Pydantic models hold the rules, SQLAlchemy
  rows describe storage, and repositories translate. The domain never depends on the database.
- **Unit of Work.** Repositories only flush; one transaction per request commits or rolls back
  everything together. The API commits *before* the response is sent, so a failed save is
  never reported as success (there is a test for exactly that).
- **Updates build new objects** (`goal.with_progress(60)`) that are fully validated, instead of
  mutating fields in place.
- **Approval requests are a state machine** (`PENDING -> APPROVED / REJECTED`); approving a
  status change applies it in the same transaction.
- **Privacy by design:** no ID numbers, addresses or medical data; sensitive answers live in a
  separate personal file with stricter access; logs carry ids, never names.
- **Errors stay in their layer:** services raise business errors, and only the API maps them to
  401 / 403 / 404 / 409 / 422 (and 502 / 503 for AI steps, with fixed messages).
- **Privacy is enforced in code, not in prompts.** Every AI task declares how sensitive its data
  is, and the router refuses to send personal data to a model that does not run locally. Names
  are replaced by tokens (`[TRAINEE_7]`) before any cloud model; the pipeline only accepts an
  `AnonymizedText`, a type that only the anonymizer can create. Recordings are transcribed by a
  local Whisper model and deleted once the transcript is saved.
- **No model call inside a database transaction.** A pipeline reads in a short transaction,
  calls the model with none open, then writes in a short one that locks the row and re-checks
  its status, so two requests on the same meeting never save twice (tested with real threads
  against PostgreSQL).
- **The model's answer is validated twice:** for shape (Pydantic, one correction attempt) and
  for content against the database (is the owner a participant? is it this trainee's goal?).
  Whatever cannot be used is cleared and reported, never guessed.

## Stack

Python 3.12 · uv · FastAPI · Pydantic · SQLAlchemy 2 · PostgreSQL 17 · Alembic · Docker ·
pytest · testcontainers · GitHub Actions · LiteLLM · Claude · Ollama · faster-whisper

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env                          # then set a password
docker compose up -d                          # PostgreSQL
uv run alembic upgrade head                   # create the tables
uv run python -m pacer.demo                   # demo academy: 35 trainees, 4 counselors, a manager
uv run uvicorn pacer.api.app:app --reload     # API
```

Open http://127.0.0.1:8000/docs. Every request needs an `X-Staff-Id` header; the demo
loader prints the ids (1 is the manager). This header is a development stand-in for real
authentication, isolated in one function so it can be replaced without touching endpoints.

AI features need `ANTHROPIC_API_KEY` in `.env` (never in Git). Try them on sample data:

```bash
uv run python -m pacer.demo.meeting weekly          # dry run: prints exactly what would be sent
uv run python -m pacer.demo.meeting weekly --send   # summary and proposed tasks (about $0.001)
uv run python -m pacer.demo.talk maya --send        # a personal talk: progress and note proposals
uv sync --extra audio                               # optional: local Whisper, for recordings
```

## Tests

```bash
uv run pytest                        # everything (Docker must be running)
uv run pytest -m "not integration"   # fast tests only, no Docker
uv run pytest --run-llm -m llm       # against real models: costs a fraction of a cent, never in CI
```

- **Fast tests** run services and the HTTP API against in-memory fake repositories,
  swapped in through dependency injection.
- **Integration tests** run the real chain against a throwaway PostgreSQL container,
  including the Alembic migrations themselves.
- **AI tests** use a fake model that records what it was sent, so tests can check that no
  known name ever reached a cloud model, and that no transaction was open during the call.

CI runs the full suite on every pull request.

## Status

| Stage | Contents | State |
|---|---|---|
| 1. Data layer | domain models, PostgreSQL, migrations, repositories | done |
| 2. Logic and API | services, roles, approvals, REST API, demo data, CI | done |
| 3. AI core | LLM client, model routing, anonymization, meeting pipeline, talk summaries | core done |
| 4. Knowledge and quality | RAG over education texts, evaluation set, model comparison | planned |
| 5. Assistant and UI | tool-using agent, staff interface | planned |
