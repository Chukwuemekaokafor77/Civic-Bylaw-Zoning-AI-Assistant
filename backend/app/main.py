"""FastAPI entry point.

Phase 1 scope: application shell, structured logging, CORS, and health
checks. The /stream SSE route and the retrieval services land in Phase 3 —
this module deliberately exposes no query surface yet.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.models.schemas import DependencyStatus, HealthResponse

VERSION = "0.1.0"

# ---------------------------------------------------------------------
#  Structured logging (Section 1: structlog + request tracing)
# ---------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
        if get_settings().is_local
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.http = httpx.AsyncClient(timeout=10.0)
    log.info(
        "startup",
        environment=settings.environment,
        version=VERSION,
        supabase_url=settings.supabase_url,
        languages=settings.supported_languages,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        log.info("shutdown")


app = FastAPI(
    title="Atlantic Canada Civic Bylaw & Zoning AI Assistant",
    description=(
        "Retrieval-augmented assistant over municipal zoning bylaws in "
        "New Brunswick, Nova Scotia, Prince Edward Island, and "
        "Newfoundland and Labrador."
    ),
    version=VERSION,
    lifespan=lifespan,
)

_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """Attach a request id and log latency for every call (Section 7 audit trail)."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", method=request.method)
        raise
    finally:
        structlog.contextvars.unbind_contextvars("request_id", "path")
    response.headers["x-request-id"] = request_id
    log.info(
        "request",
        method=request.method,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


# ---------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness: does the process answer? No outbound calls."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        version=VERSION,
        checked_at=datetime.now(timezone.utc),
    )


async def _check_supabase(settings: Settings, client: httpx.AsyncClient) -> DependencyStatus:
    """Confirm Supabase PostgREST is reachable and the service-role key works.

    Hits the `provinces` table, which the Phase 1 DDL seeds with four rows.
    A 200 here proves URL, key, and schema are all in place at once.
    """
    started = time.perf_counter()
    url = f"{settings.supabase_url}/rest/v1/provinces"
    key = settings.supabase_service_role_key.get_secret_value()
    try:
        resp = await client.get(
            url,
            params={"select": "code", "limit": "1"},
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
        )
    except httpx.HTTPError as exc:
        return DependencyStatus(
            name="supabase",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    latency = round((time.perf_counter() - started) * 1000, 2)
    if resp.status_code != 200:
        return DependencyStatus(
            name="supabase",
            ok=False,
            detail=f"HTTP {resp.status_code}: {resp.text[:200]}",
            latency_ms=latency,
        )
    return DependencyStatus(name="supabase", ok=True, latency_ms=latency)


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(request: Request) -> JSONResponse:
    """Readiness: can the app actually reach its dependencies?

    This is the Phase 1 exit check ("Supabase reachable from both").
    Returns 503 when a dependency is down so a platform health probe
    treats it as not-ready rather than healthy-but-broken.
    """
    settings = get_settings()
    deps = [await _check_supabase(settings, request.app.state.http)]

    # Key presence only. Groq and OpenAI are metered, so readiness must not
    # spend money on every probe; real calls are exercised in Phases 2 and 3.
    deps.append(
        DependencyStatus(
            name="openai_key",
            ok=settings.openai_api_key is not None,
            detail=None if settings.openai_api_key else "OPENAI_API_KEY not set (needed from Phase 2)",
        )
    )
    deps.append(
        DependencyStatus(
            name="groq_key",
            ok=settings.groq_api_key is not None,
            detail=None if settings.groq_api_key else "GROQ_API_KEY not set (needed from Phase 3)",
        )
    )

    all_ok = all(d.ok for d in deps)
    body = HealthResponse(
        status="ok" if all_ok else "degraded",
        environment=settings.environment,
        version=VERSION,
        checked_at=datetime.now(timezone.utc),
        dependencies=deps,
    )
    supabase_ok = deps[0].ok
    return JSONResponse(
        status_code=200 if supabase_ok else 503,
        content=body.model_dump(mode="json"),
    )
