# Pacer

[![tests](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml/badge.svg)](https://github.com/itayerenreich/pacer/actions/workflows/tests.yml)

> **Work in progress.** The data layer, the API, the AI core, the knowledge library (RAG), the counselor
> coach, the assistant and the staff interface are built and tested. Before real users: real sign-in,
> deployment and backups (see [Status](#status)).

**In short:** Pacer helps pre-military academies (מכינות קדם-צבאיות) run their year, educate by the
texts and values they believe in, and give their counselors practical tools for the work with each trainee.

- **Layered architecture, privacy enforced in code:** personal data never leaves the computer; names are
  anonymized and health details removed before any cloud model, and the types make that step impossible to skip.
- **RAG over the academy's own library:** answers come only from its texts, every claim cites its passage,
  and when no passage fits, the system says so instead of guessing.
- **AI proposes, people decide:** every summary, task or goal update waits for a staff member's approval.
- **Measured, not assumed:** a versioned evaluation set, an LLM judge, a commercial vs open model comparison,
  and tests with CI on every pull request.

**About me:** I'm Itay Erenreich, a computer science and psychology student. In the two years before my
degree I worked as a counselor and branch manager at a pre-military academy, and saw firsthand the need
Pacer is built to answer.

## What it does

A backend for the staff of a pre-military academy (מכינה קדם-צבאית): counselors and a
manager track trainees, the personal goals they set together, and progress over time.
Sensitive actions, like opening a trainee's personal file or sending a trainee home,
go through manager approval.

A weekly schedule with attendance per session shows each trainee's attendance rate.

On top of this data layer sit AI features: the text of a staff meeting becomes a
summary and proposed tasks, and a counselor's notes from a personal talk become a
summary, goal progress updates and notes worth remembering. A **counselor coach** gives practical
advice about a specific trainee, built on what already worked with them and on the academy's library.
The AI sits **above** the data, never instead of it: it proposes, and a person approves every item
before anything changes.

An **assistant** answers a staff member's question by deciding for itself what to look up (their trainees,
a trainee's record, attendance, the week, the library, the coach), with the staff member's own permissions,
read only. A web interface in Hebrew puts it next to the trainees, the week's schedule and attendance.

## Architecture

```
     web/            React + TypeScript interface, in Hebrew, right to left; talks only to the API
          │
  HTTP clients (/docs, web/)
          │
     api/            FastAPI: routing, request schemas, who is asking, errors -> HTTP codes
          │
     evaluation/     versioned answer keys (evals/), a runner, an LLM judge, reports
          │
     ai/             model routing, anonymization, structured output, the meeting and talk pipelines,
          │          retrieval and answers with citations, the coach, the agent and its tools
     library/        importing texts (Word, PDF): cleaning, chunking, a manifest per file
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
  separate personal file with stricter access; logs carry ids, never names. Who may not see a
  record gets 404, exactly as if it did not exist, so ids cannot be probed. The notes and personal
  talks of a trainee whose file was closed are deleted after a retention period
  (`python -m pacer.maintenance purge-notes`).
- **Errors stay in their layer:** services raise business errors, and only the API maps them to
  401 / 403 / 404 / 409 / 422 (and 502 / 503 for AI steps, with fixed messages).
- **Privacy is enforced in code, not in prompts.** Every AI task declares how sensitive its data
  is, and the router refuses to send personal data to a model that does not run locally. Names
  are replaced by tokens (`[TRAINEE_7]`) before any cloud model; the pipeline only accepts an
  `AnonymizedText`, a type that only the anonymizer can create. Hebrew names that are also
  everyday words ("כבר" is כ + בר) become neutral tokens that turn back into the word. A second,
  independent word-by-word check runs before anything is sent. Sentences about health are removed
  from talk notes and meeting transcripts before anything is sent. There are no audio recordings
  at all: a meeting comes in as text (decision 2026-10-19, after weighing Hebrew speech-to-text).
- **No model call inside a database transaction.** A pipeline reads in a short transaction,
  calls the model with none open, then writes in a short one that locks the row and re-checks
  its status, so two requests on the same meeting never save twice (tested with real threads
  against PostgreSQL).
- **The model's answer is validated twice:** for shape (Pydantic, one correction attempt) and
  for content against the database (is the owner a staff member named in the meeting? is it this
  trainee's goal?). Whatever cannot be used is cleared and reported, never guessed. Prompts are
  versioned files (`meeting_extract@v2`); a used version is never edited.

## The knowledge library and its evaluation

Staff can ask a question and get an answer **only from the academy's library**, with every claim
citing the passage it rests on (RAG). Texts are cleaned, split into chunks, embedded by an open model
on this computer (bge-m3) and searched in PostgreSQL with pgvector. Citations are built from our own
metadata, never by the model; a claim with no passage behind it is dropped and reported; and when no
passage is relevant enough, no model is asked at all. Choices like the embedding model, the chunking
and the relevance floor were made by measurement (`python -m pacer.library compare`).

Quality is measured, not assumed. `python -m pacer.evaluation run` takes a versioned answer key
(`evals/`) and checks every answer twice: by **code**, where code can (was a right source retrieved
and cited? is goal progress in range? did a health detail slip through?), and by a **second, stronger
model acting as a judge**, where it cannot (does each claim really appear in its passage? does the
answer meet the case's checklist? how useful is it?). The judge gives a reason for every verdict, so
a person can check it. Each question is also answered by the same model **without** the library, to
see what retrieval adds.

First run, Claude Haiku answering, Claude Sonnet judging (`library@v1`, 23 questions; `talks@v1`, 8 talks):

| | with the library | without |
|---|---|---|
| claims supported by the passage they cite | 98% | - |
| a right source retrieved and cited | 94% | - |
| checklist items met | 92% | 74% |
| usefulness (judge, 1-5) | 3.0 | 3.5 |
| talk summaries: goal progress in the expected range | 100% | |

> **Read these numbers for what they are.** The demo library is deliberately tiny: five classical
> Jewish texts on education (public domain), chosen to build and test the mechanism. An academy's
> real library would hold many more sources, and more practical ones: its own guidelines,
> professional books, studies. So the answers here are faithful but stay at the level of principle,
> and a model answering from general knowledge reads as more practical. What this shows is the
> **method**: a pipeline that answers only from its sources and says when it has none, and a
> second model that checks the first one's RAG against an answer key, so every change (a prompt,
> a model, a source) can be measured instead of guessed.

### A commercial model or an open one on this laptop?

The same evaluation, run on the first 4 library questions and 4 talks, with Claude Haiku (cloud) and
gemma3:4b (open, through Ollama, on a 16 GB laptop with no GPU), Claude Sonnet judging both:

| | Claude Haiku | gemma3:4b, local |
|---|---|---|
| claims supported by the passage they cite | 95% | 50% |
| a right source cited | 100% | 75% |
| usefulness (judge, 1-5) | 3.25 | 1.75 |
| talk summaries: goal progress in the expected range | 100% | 0% |
| talk summaries: expected notes found | 100% | 89% |
| seconds per answer / per talk | 17 / 4 | 109 / 118 |

The local model kept every name on the computer, and failed the task: it overstated progress (two
remarks in a group talk became 67% of a goal), skipped goals the talk was about, and once turned
"his younger brother" into "his younger sister". On this hardware, the evidence for anything written
into a trainee's file points to the stronger model, with privacy kept in code (names anonymized, health
details removed, the router refusing personal data to the cloud). Two technical findings on
the way: Ollama silently cuts a message longer than its default 2,048-token context, and a small
model broke its JSON on Hebrew abbreviations (רמב"ם) until Ollama was asked to enforce the schema.
Four cases are few, and a larger open model on a GPU could do far better: the runner can measure it
(`--model`, `python -m pacer.evaluation compare`).

## The counselor coach and its safety gate

A counselor asks "how do I help Nadav take part in the group talks?" and gets steps they can start this
week, built on Nadav's approved record: his goals, and notes on what **worked** with him and what did not
(`POST /coach/ask`, `python -m pacer.demo.coach`).

```
question ─► permissions (404 if this counselor may not see him)
         ─► safety gate: a phrase list in code, then Claude as a second net ──► exceptional incident?
                                                                            the fixed procedure, no advice
         ─► his record, built in code ─► health sentences out ─► anonymized as one text
         ─► library passages (bge-m3, local) ─► Claude: understanding, steps, who to involve
         ─► validation in code ─► advice + the procedure line on every answer
```

- **Grounded where it can be, marked where it cannot.** Each step is grounded only by a passage that was
  really sent; anything else is labeled general advice. The code decides, not the model.
- **One procedure for every exceptional incident** (self-harm, violence, abuse, sexual harm): immediate
  danger, the manager, removing means, the reporting duty for minors, help lines. The same text every
  time, never written by a model, so a wrong classification cannot drop a step.
- **The safety gate, measured.** The first phrase list found 15 of 59 risk sentences an independent
  reviewer wrote to break it. After fixing what that taught, **30 of 60** on a held-out set nobody tuned it on: it finds words, and subtle risk is described as a
  scene ("stood on the roof at night"). With Claude as a second net that can only *add* a referral:
  **60 of 60**, and no false alarm added (`python -m pacer.evaluation safety --send`). The screen sees
  exactly the anonymized text the coach would send anyway, so it costs no privacy; if it fails, no
  unscreened advice goes out.
- **Measured personalization:** `python -m pacer.evaluation coach` answers each case with and without
  the trainee's record, and a judge (Claude Sonnet) scores both on the same checklist (`coach@v1`, 9 cases):

  | Claude Haiku as the coach | |
  |---|---|
  | the right outcome (advice, or the procedure for an exceptional incident) | 9 / 9 |
  | items about THIS trainee met, **with** their record | **95%** |
  | the same items, **without** it | 48% |
  | no diagnosis, no medical labels, no treatment advice | 100% |
  | usefulness (judge, 1-5) | 3.9 |
  | steps grounded in the library | 0% |
  | cost of the 9 answers | $0.02 |

  Without the record, the advice for a quiet trainee is reasonable and generic; with it, it asks him to
  prepare one point in advance (what worked with him) and warns against calling on him in front of the
  group (what did not). Claude Sonnet scored the same and cost 16 times more, so Haiku stays.
  **0% grounded is honest, not a bug:** no library passage was close enough to these practical questions
  (the demo library is five classical texts), so every step is marked general advice. The first run also
  found a real failure: Haiku sometimes added a stray key to a valid answer, which made it a 502 for 1
  question in 9; the coach now drops keys it does not know, and a test holds it.

## The weekly schedule and attendance

The manager defines a typical week once (a **template**: "Sunday 06:45 morning sport"), and publishes each
week from it. One week can change (a trip, a cancelled evening) without touching the template, and a
change to the template never rewrites a week that already happened: template and instance are two tables.
Counselors mark each session for the trainees of their groups: **present, absent, or excused**.

- **The rate is present / (present + absent).** Excused does not count against a trainee; a session nobody
  marked is shown as unmarked and not counted (a counselor who forgot must not lower a trainee's rate); a
  cancelled session is left out; "nothing counted yet" is no rate, not 0%.
- **Who needs attention comes first:** `GET /attendance` lists the trainees you may see, the lowest rate
  first, each with the sessions they miss most (in the demo: Ben, whose note says he is late after social
  evenings, missed morning sport 8 of his 9 absences).
- **No reason text, on purpose:** "he was at the doctor" is health data. There is no field for it, and the
  API rejects one.
- **The coach reads it:** the last 30 days' rate and the sessions missed most go into the trainee's record,
  so advice about a trainee who is late is advice about mornings.
- **The database enforces it too:** one session per template slot and date, one mark per trainee and
  session (marking again replaces it), and the academy's "today" is Israel's, not the server's UTC.

## The assistant (an agent) and the interface

Everything before this is a **pipeline**: the code decides the steps. The assistant is an **agent**: the
model decides which tools to call, and our code runs them (`POST /agent/chat`, `ai/agent.py`). Asked "how
do I deal with Nadav Nahum, who isn't meeting his goals?", it looks up his attendance and his record, asks
the coach (which searches the library itself), and answers.

```
message ─► phrase list (every user turn) ─► the screen ─► exceptional incident? the fixed procedure
        ─► health sentences out, anonymized in ONE token space for the whole conversation
        ─► loop: the model answers, or asks for tools ─► our code runs each tool as THIS staff member
                 ─► health sentences out of the result ─► anonymized ─► back to the model
           limits: 6 rounds, 10 tool calls, $0.25; the last round must answer in text
        ─► the answer: health out, unknown tokens reported, names restored as the asker may see them
```

- **The model only asks; the code decides.** Seven read-only tools, each a Pydantic model (its JSON schema
  is what the model sees) over a service that already exists. None has a "who is asking" parameter: the
  actor is bound in code, so a trainee of another group is "not found", in the same words as one that does
  not exist. A wrong argument is an error the model reads and corrects, never a crash.
- **One token space per conversation.** `[TRAINEE_7]` is stable by itself, but ambiguous names were
  numbered per text; across many tool results two different words would both be `[NAME_1]`. A session
  renumbers them, and a token typed into a message (even with an invisible character inside) is defused.
- **A hidden trainee is never completed.** Written back only in the form the asker wrote for that token:
  "כהן" stays "כהן", even if the same message mentions another "יעל".
- **Nothing about the conversation is stored.** The interface sends the history with each message; the
  server checks it with the phrase list and anonymizes it again every time, as untrusted text.
- **What was looked up is shown.** Under each answer, the route the agent took: each tool call as a
  waypoint, with counts (never content), and the library sources it was shown.
- **Measured like the rest:** `python -m pacer.evaluation agent` runs 19 cases (`evals/agent-v1.json`) and
  checks in code the outcome, the tools it had to call and must not call, that every call was about the
  right trainee, and the answer's text; a judge scores the checklists. Only the agent's model changes
  between runs (`--model`), so Haiku and Sonnet are compared on choosing tools.

The interface (`web/`, React + TypeScript + Vite) has four screens: the assistant, the trainees (goals,
approved notes, attendance), the week with attendance marking, and an overview (attendance by group and by
trainee, the lowest first; approvals waiting for the manager; tasks). It talks only to the API, so every
permission is the API's.

## Stack

Python 3.12 · uv · FastAPI · Pydantic · SQLAlchemy 2 · PostgreSQL 17 · Alembic · Docker ·
pytest · testcontainers · GitHub Actions · LiteLLM · Claude · Ollama · pgvector ·
React · TypeScript · Vite

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

The interface needs [Node](https://nodejs.org) 22:

```bash
cd web && npm install && npm run dev          # http://localhost:5173 (forwards /api to the API on 8000)
```

Sign in with a staff id (the same development identity). Or everything in Docker, in one container:

```bash
docker compose --profile app up -d --build    # http://localhost:8000: the interface, and the API under /api
docker compose exec app uv run --no-sync python -m pacer.demo   # the demo academy, once
```

AI features need `ANTHROPIC_API_KEY` in `.env` (never in Git). Try them on sample data:

```bash
uv run python -m pacer.demo.meeting weekly          # dry run: prints exactly what would be sent
uv run python -m pacer.demo.meeting weekly --send   # summary and proposed tasks (about $0.001)
uv run python -m pacer.demo.talk maya --send        # a personal talk: progress and note proposals
uv run python -m pacer.library ask "איך מתאימים את ההדרכה לחניך?" --send --out data/answer.md
uv run python -m pacer.evaluation run --limit 1 --out data/eval-try.md   # the evaluation, on one case
uv run python -m pacer.demo.coach "נדב הררי" "איך לעזור לו להשתתף יותר בשיחות הקבוצה?"         # dry run
uv run python -m pacer.demo.coach "נדב הררי" "איך לעזור לו להשתתף יותר בשיחות הקבוצה?" --send  # the coach
uv run python -m pacer.evaluation agent --out data/agent-haiku.md   # the agent on its 19 cases (about $0.30)
```

The library needs [Ollama](https://ollama.com) with `ollama pull bge-m3`, and the texts imported
(`python -m pacer.library import`; the files themselves stay out of Git). Reports are Markdown,
written right-to-left: open them in VS Code with Ctrl+Shift+V.

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

CI runs ruff, mypy and the full suite on every pull request, and builds the interface (tsc and Vite).

## Status

| Stage | Contents | State |
|---|---|---|
| 1. Data layer | domain models, PostgreSQL, migrations, repositories | done |
| 2. Logic and API | services, roles, approvals, REST API, demo data, CI | done |
| 3. AI core | LLM client, model routing, anonymization, meeting pipeline, talk summaries | done |
| 4. Knowledge and quality | a curated library, RAG with citations, evaluation set, open vs commercial models | done |
| 5. Counselor coach | a two-layer safety gate, the trainee's approved record, advice grounded in the library where it can be, measured | done |
| 6. Assistant and UI | a tool-using agent over the services, the staff interface, Docker | built; measurement and demo next |
| Before real users | real sign-in, where it runs, backups, onboarding an academy, privacy review | planned |
