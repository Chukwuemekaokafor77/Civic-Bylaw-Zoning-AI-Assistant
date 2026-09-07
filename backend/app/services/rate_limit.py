"""Request throttling (Phase 5, Step 2).

Section 1 asks for per-IP and per-session throttling, because both
providers behind an answer are metered: a Gemini embedding call and a Groq
generation call. Without a limit, one script can spend a day's free-tier
quota in a minute and leave the assistant returning the "not found"
fallback to everyone else - a denial of service that costs the attacker
nothing and looks, to a resident, like the tool simply not working.

The key combines IP and session id, so:

  * a single browser tab cannot loop faster than the per-session limit, and
  * rotating the client-generated session id does not buy more quota,
    because the IP half of the key does not change.

The session half is read from the `X-Session-Id` HEADER, not from the
request body. The limit is evaluated before the handler runs, and reading
the body that early consumes the receive stream - the handler then sees an
empty body and every request fails validation with a 422. A header is
readable without touching the stream at all.

The session id is used here and nowhere else. It is not stored in the
audit log (see audit_logger) - it exists to make a limit stick to a caller
for a few minutes, not to identify anyone.
"""

from __future__ import annotations

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import Settings

log = structlog.get_logger(__name__)

# Message shown to a throttled caller. Deliberately not phrased as an
# error: on a free-tier deployment this is the system working as intended,
# and a civic tool should not tell a resident something went wrong when
# nothing did.
RATE_LIMIT_MESSAGE = (
    "This assistant limits how many questions it answers per minute so it "
    "can stay free to run. Please wait a moment and ask again."
)


SESSION_HEADER = "X-Session-Id"

# A client-supplied string is untrusted input and lands in a limiter key,
# so it is length-capped rather than used verbatim.
MAX_SESSION_ID_LENGTH = 64


def client_key(request: Request) -> str:
    """Rate-limit key: IP, plus session id when the caller supplied one.

    IP alone would throttle everyone behind one municipal office NAT
    together. Session alone would be trivially bypassed by generating a new
    id per request. Combined, a shared network still gets per-tab limits
    while a rotating id gains nothing.
    """
    address = get_remote_address(request) or "unknown"
    session = (request.headers.get(SESSION_HEADER) or "").strip()
    if session:
        return f"{address}:{session[:MAX_SESSION_ID_LENGTH]}"
    return address


def build_limiter(settings: Settings) -> Limiter:
    return Limiter(
        key_func=client_key,
        default_limits=[],
        headers_enabled=True,  # surfaces Retry-After to the client
        enabled=settings.environment != "test",
    )


def stream_limits(settings: Settings) -> str:
    """Limits for the answering route, as slowapi's ";"-separated form."""
    return (
        f"{settings.rate_limit_per_minute}/minute;"
        f"{settings.rate_limit_per_day}/day"
    )


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    log.info(
        "rate_limited",
        client=client_key(request),
        limit=str(exc.detail),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=429,
        content={"detail": RATE_LIMIT_MESSAGE, "limit": str(exc.detail)},
        headers={"Retry-After": "60"},
    )
