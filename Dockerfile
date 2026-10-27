# Pacer in one image: the interface (built with Node) and the API (Python), served together by
# pacer.api.web. Migrations run on start, so an update to the image brings the database up to date.
#
#   docker compose --profile app up -d --build      # the database and the app: http://localhost:8000
#
# Secrets come from .env at run time (compose's env_file); none is baked into the image.

FROM node:22.11-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer: a change to the code does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --locked
COPY --from=web /web/dist ./web/dist

RUN useradd --create-home --uid 10001 pacer
USER pacer
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/openapi.json')"
CMD ["sh", "-c", "uv run --no-sync alembic upgrade head && exec uv run --no-sync uvicorn pacer.api.web:app --host 0.0.0.0 --port 8000"]
