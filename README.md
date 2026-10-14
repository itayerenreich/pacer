# Pacer

[![tests](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml/badge.svg)](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml)

A backend for the staff of a pre-military academy (מכינה קדם-צבאית): counselors and a
manager track trainees, the personal goals they set together, and progress over time.
Sensitive actions, like opening a trainee's personal file or sending a trainee home,
go through manager approval.

AI features (meeting summaries, a knowledge base, an assistant) come next, on top of
this data layer: the AI sits **above** the data, never instead of it.

## Architecture

```
  HTTP clients (/docs, UI, agent)
          │
     api/            FastAPI: routing, request schemas, who is asking, errors -> HTTP codes
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
  401 / 403 / 404 / 409 / 422.

## Stack

Python 3.12 · uv · FastAPI · Pydantic · SQLAlchemy 2 · PostgreSQL 17 · Alembic · Docker ·
pytest · testcontainers · GitHub Actions

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

## Tests

```bash
uv run pytest                        # everything (Docker must be running)
uv run pytest -m "not integration"   # fast tests only, no Docker
```

- **Fast tests** run services and the HTTP API against in-memory fake repositories,
  swapped in through dependency injection.
- **Integration tests** run the real chain against a throwaway PostgreSQL container,
  including the Alembic migrations themselves.

CI runs the full suite on every pull request.

## Status

| Stage | Contents | State |
|---|---|---|
| 1. Data layer | domain models, PostgreSQL, migrations, repositories | done |
| 2. Logic and API | services, roles, approvals, REST API, demo data, CI | done |
| 3. AI core | LLM client (LiteLLM), model routing, meeting summary pipeline | next |
| 4. Knowledge and quality | RAG over education texts, evaluation set, model comparison | planned |
| 5. Assistant and UI | tool-using agent, staff interface | planned |
