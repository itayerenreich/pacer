"""The whole of Pacer at one address: the API under /api, and the web interface (web/, built) at /.

    cd web && npm run build                          # once: web/dist
    uv run uvicorn pacer.api.web:app                 # http://127.0.0.1:8000

The API itself (pacer.api.app) stays as it is, so /docs and the tests do not change. While developing
the interface, `npm run dev` serves it with hot reload and forwards /api to the API on port 8000.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .app import create_app

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_web_app(dist: Path = WEB_DIST) -> FastAPI:
    site = FastAPI(title="Pacer", docs_url=None, redoc_url=None, openapi_url=None)
    site.mount("/api", create_app())
    if dist.is_dir():
        site.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:

        @site.get("/", include_in_schema=False)
        def no_interface() -> RedirectResponse:
            return RedirectResponse("/api/docs")  # not built yet: the API's docs at least

    return site


app = create_web_app()
