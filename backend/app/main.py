"""FastAPI entry point.

Application shell, structured logging, CORS, health checks, and the
/stream SSE route that answers one question about one municipality.

/stream wires the Phase 3 services together in a fixed order: retrieve
(hybrid), emit citations, stream generated tokens, then write the audit
row. Citations go out before the tokens so the reader sees which bylaw
sections an answer rests on even if they stop reading halfway.
"""

# NOTE: no `from __future__ import annotations` here, deliberately.
# Postponed annotations turn parameter types into strings, and FastAPI
# resolves them against the function's __globals__. slowapi's @limit
# wrapper has its own module globals, so "ChatRequest" becomes
# unresolvable and FastAPI silently demotes the body parameter to a query
# parameter - every POST /stream then fails validation with a 422.

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.models.schemas import (
    ChatRequest,
    Chunk,
    Citation,
    DependencyStatus,
    HealthResponse,
    StreamEventType,
)
from app.services.audit_logger import AuditLogger, QueryRecord
from app.services.rag_engine import GenerationContext, RagEngine
from app.services.rate_limit import (
    build_limiter,
    rate_limit_handler,
    stream_limits,
)
from app.services.retrieval import HybridRetriever, MunicipalityInfo

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
    # Built once per process: each owns a connection pool, and the
    # embedder's SDK client is comparatively expensive to construct.
    app.state.retriever = await HybridRetriever.create(settings)
    app.state.auditor = await AuditLogger.create(settings)
    app.state.engine = RagEngine(settings)
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
    title="Canadian Civic Bylaw & Zoning AI Assistant",
    description=(
        "Retrieval-augmented assistant over Canadian municipal zoning "
        "bylaws. Answers cite the bylaw section they come from, and are "
        "scoped to one municipality per request."
    ),
    version=VERSION,
    lifespan=lifespan,
)

_settings = get_settings()

# Section 1: per-IP and per-session throttling. Both providers behind an
# answer are metered, so an unthrottled loop spends the day's free-tier
# quota and leaves everyone else with the fallback message.
limiter = build_limiter(_settings)
STREAM_LIMITS = stream_limits(_settings)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

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

    Hits the `provinces` table, which the Phase 1 migrations seed with all
    13 Canadian jurisdictions. A 200 here proves URL, key, and schema are
    all in place at once.
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

    # Key presence only. Groq is metered and Gemini is quota-limited, so
    # readiness must not spend either on every probe; real calls are
    # exercised in Phases 2 and 3.
    #
    # Truthiness, not `is not None`: an env file containing `GEMINI_API_KEY=`
    # parses to SecretStr("") rather than None. SecretStr defines __len__, so
    # an empty one is falsy but not None — checking identity here would report
    # a blank key as healthy, which is exactly the deploy mistake this probe
    # exists to catch.
    deps.append(
        DependencyStatus(
            name="gemini_key",
            ok=bool(settings.gemini_api_key),
            detail=None if settings.gemini_api_key else "GEMINI_API_KEY not set (needed from Phase 2)",
        )
    )
    deps.append(
        DependencyStatus(
            name="groq_key",
            ok=bool(settings.groq_api_key),
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


# ---------------------------------------------------------------------
#  Chat streaming (Phase 3, Step 5)
# ---------------------------------------------------------------------

def _citations_for(chunks: list[Chunk], info: MunicipalityInfo) -> list[Citation]:
    """Build CitationCard payloads from the retrieved chunks.

    `source_url` comes from the chunk's own metadata rather than the
    municipality row: a bilingual municipality has a different document
    per language, and linking a French citation to the English PDF would
    send a reader to a document that does not contain the clause quoted.
    """
    citations: list[Citation] = []
    for chunk in chunks:
        citations.append(
            Citation(
                chunk_id=chunk.id,
                municipality_name=info.name,
                bylaw_name=chunk.bylaw_name,
                section_number=chunk.section_number,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                source_url=chunk.metadata.get("url") or info.source_url,
                language=chunk.language,
            )
        )
    return citations


def _event(payload: StreamEventType) -> dict:
    """Wrap one envelope as an SSE frame."""
    return {
        "event": payload.type,
        "data": json.dumps(payload.model_dump(mode="json")["data"]),
    }


@app.post("/stream", tags=["chat"])
@limiter.limit(STREAM_LIMITS)
async def stream(request: Request, body: ChatRequest):
    """Answer one question about one municipality's bylaws, as SSE.

    Citations are emitted BEFORE the tokens. The frontend can then render
    the source cards while the answer streams in, and - more importantly -
    a reader who stops reading halfway has still been shown which bylaw
    sections the answer rests on.
    """
    retriever: HybridRetriever = request.app.state.retriever
    auditor: AuditLogger = request.app.state.auditor
    engine: RagEngine = request.app.state.engine

    info = await retriever.municipality(body.municipality_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown municipality: {body.municipality_id}")
    if not info.is_active:
        # The registry marks Halifax and Charlottetown inactive for stated
        # reasons. Answering anyway would cite a source the project has
        # already judged unusable.
        raise HTTPException(
            status_code=409,
            detail=(
                f"{info.name} is not yet available. Its bylaw source is "
                "still being verified."
            ),
        )

    async def publish():
        answer_parts: list[str] = []
        chunks: list[Chunk] = []
        try:
            result = await retriever.retrieve(
                body.query, body.municipality_id, language=body.language
            )
            chunks = result.chunks

            yield _event(
                StreamEventType(type="citations", data=_citations_for(chunks, info))
            )

            context = GenerationContext(
                municipality_id=info.id,
                municipality_name=info.name,
                province_code=info.province_code,
                language=body.language,
                bylaw_last_verified_at=info.bylaw_last_verified_at,
            )

            async for token in engine.stream(body.query, chunks, context):
                answer_parts.append(token)
                yield _event(StreamEventType(type="token", data=token))

            yield _event(StreamEventType(type="done", data=None))

        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            log.exception("stream_failed", municipality=body.municipality_id)
            yield _event(
                StreamEventType(
                    type="error",
                    data="The assistant could not complete this answer. Please try again.",
                )
            )
            _ = exc
        finally:
            # Audit last, and never let it break a delivered answer.
            await auditor.record(
                QueryRecord.from_answer(
                    municipality_id=body.municipality_id,
                    user_query=body.query,
                    chunks=chunks,
                    response_text="".join(answer_parts),
                    language=body.language,
                )
            )

    return EventSourceResponse(publish())
