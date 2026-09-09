"""Unit tests for app.services.source_tracker (Phase 2, Step 4).

Hermetic: HTTP goes through httpx.MockTransport (a real client over fake
responses, so the redirect and streaming paths are genuinely exercised),
and Supabase through a recording stub.

The fetch cases mirror failures actually observed while verifying the
registry - a login gateway served as HTTP 200, a retired consolidation
URL, and two municipalities whose documents arrive only after a redirect.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import date, datetime, timezone

import httpx
import pytest

from app.config import Settings
from app.services.source_tracker import (
    MUNICIPALITIES_TABLE,
    SOURCES_TABLE,
    ChangeReason,
    FetchResult,
    NotADocument,
    SourceGone,
    SourceTracker,
    fetch_source,
)

PDF_BODY = b"%PDF-1.7\nzoning bylaw content"
LOGIN_PAGE = b"<html><head><title>Sign In</title></head><body>Please login</body></html>"


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        supabase_url="https://test-placeholder.supabase.co",
        supabase_service_role_key="test-placeholder",
    )


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def fetch(url: str, handler) -> FetchResult:
    async def run():
        async with client_for(handler) as client:
            return await fetch_source(url, client=client)

    return asyncio.run(run())


# ---------------------------------------------------------------------
#  Fetching
# ---------------------------------------------------------------------


def test_pdf_is_fetched_and_hashed():
    result = fetch("https://example.ca/z5.pdf", lambda r: httpx.Response(200, content=PDF_BODY))
    assert result.content == PDF_BODY
    assert result.content_hash == hashlib.sha256(PDF_BODY).hexdigest()
    assert result.redirected is False


def test_hash_is_content_addressed_not_url_addressed():
    a = fetch("https://a.ca/x.pdf", lambda r: httpx.Response(200, content=PDF_BODY))
    b = fetch("https://b.ca/y.pdf", lambda r: httpx.Response(200, content=PDF_BODY))
    assert a.content_hash == b.content_hash


def test_changed_content_changes_the_hash():
    a = fetch("https://a.ca/x.pdf", lambda r: httpx.Response(200, content=PDF_BODY))
    b = fetch("https://a.ca/x.pdf", lambda r: httpx.Response(200, content=PDF_BODY + b" amended"))
    assert a.content_hash != b.content_hash


def test_redirect_is_followed_and_resolved_url_recorded():
    """St. John's cites a stable alias that 302s to a dated CDN filename."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.stjohns.ca":
            return httpx.Response(
                302,
                headers={"location": "https://cdn.example.ca/regs-august-26-2026.pdf"},
            )
        return httpx.Response(200, content=PDF_BODY)

    result = fetch("https://www.stjohns.ca/Development-Regulations.pdf", handler)

    assert result.redirected is True
    assert result.citation_url == "https://www.stjohns.ca/Development-Regulations.pdf"
    assert result.resolved_url == "https://cdn.example.ca/regs-august-26-2026.pdf"


@pytest.mark.parametrize("status", [404, 410])
def test_retired_url_raises_source_gone(status):
    """CBRM and Saint John retire the URL when a consolidation is replaced."""
    with pytest.raises(SourceGone, match="re-discovered"):
        fetch("https://cbrm.ns.ca/LUB-to-June-23-2026.pdf", lambda r: httpx.Response(status))


def test_login_page_served_as_200_is_refused():
    """Charlottetown's GetFile.ashx returns 200 and a sign-in gateway."""
    with pytest.raises(NotADocument, match="sign-in page"):
        fetch(
            "https://www.charlottetown.ca/common/pages/GetFile.ashx?key=x",
            lambda r: httpx.Response(200, content=LOGIN_PAGE, headers={"content-type": "text/html"}),
        )


def test_non_pdf_body_is_refused_even_without_login_markers():
    with pytest.raises(NotADocument, match="not a PDF"):
        fetch("https://example.ca/x.pdf", lambda r: httpx.Response(200, content=b"just text"))


def test_server_error_is_not_swallowed():
    with pytest.raises(httpx.HTTPStatusError):
        fetch("https://example.ca/x.pdf", lambda r: httpx.Response(503))


def test_expect_can_be_relaxed_for_html_sources():
    """Section 4 anticipates HTML-published bylaws (Ontario, CBRM index)."""

    async def run():
        async with client_for(lambda r: httpx.Response(200, content=b"<html>bylaw</html>")) as c:
            return await fetch_source("https://example.ca/bylaw", client=c, expect="html")

    assert asyncio.run(run()).content == b"<html>bylaw</html>"


# ---------------------------------------------------------------------
#  Supabase stub
# ---------------------------------------------------------------------


class StubQuery:
    def __init__(self, table: "StubTable") -> None:
        self._table = table
        self._filters: dict = {}

    def select(self, columns):
        return self

    def update(self, values):
        self._table.updates.append({"values": values, "filters": self._filters})
        return self

    def upsert(self, row, on_conflict=None):
        self._table.upserts.append({"row": row, "on_conflict": on_conflict})
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def limit(self, n):
        return self

    async def execute(self):
        # Updates record their filters at call time, after eq() ran.
        for entry in self._table.updates:
            entry.setdefault("filters", {}).update(self._filters)
        return StubResponse(self._table.rows)


class StubResponse:
    def __init__(self, data):
        self.data = data


class StubTable:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.upserts: list[dict] = []
        self.updates: list[dict] = []


class StubClient:
    def __init__(self) -> None:
        self.tables: dict[str, StubTable] = {}

    def table(self, name):
        self.tables.setdefault(name, StubTable())
        return StubQuery(self.tables[name])


def make_tracker() -> tuple[SourceTracker, StubClient]:
    client = StubClient()
    return SourceTracker(client, settings()), client  # type: ignore[arg-type]


def decide(tracker, current_hash, **kwargs):
    return asyncio.run(
        tracker.decide("nb_fredericton", "en", "Zoning By-law Z-5", current_hash, **kwargs)
    )


# ---------------------------------------------------------------------
#  Change decisions
# ---------------------------------------------------------------------


def test_never_seen_source_is_new():
    tracker, _ = make_tracker()
    decision = decide(tracker, "abc123")
    assert decision.reason is ChangeReason.NEW
    assert decision.should_ingest is True


def test_identical_hash_skips_the_embedding_bill():
    tracker, client = make_tracker()
    client.tables.setdefault(SOURCES_TABLE, StubTable()).rows = [{"content_hash": "abc123"}]

    decision = decide(tracker, "abc123")
    assert decision.reason is ChangeReason.UNCHANGED
    assert decision.should_ingest is False
    assert "skipping" in decision.summary


def test_different_hash_triggers_reingestion():
    tracker, client = make_tracker()
    client.tables.setdefault(SOURCES_TABLE, StubTable()).rows = [{"content_hash": "old"}]

    decision = decide(tracker, "new")
    assert decision.reason is ChangeReason.CONTENT_CHANGED
    assert decision.should_ingest is True
    assert decision.previous_hash == "old"


def test_force_overrides_an_unchanged_hash():
    tracker, client = make_tracker()
    client.tables.setdefault(SOURCES_TABLE, StubTable()).rows = [{"content_hash": "abc123"}]

    decision = decide(tracker, "abc123", force=True)
    assert decision.reason is ChangeReason.FORCED
    assert decision.should_ingest is True


def test_null_stored_hash_is_treated_as_new():
    """A registry row can exist before the first successful ingestion."""
    tracker, client = make_tracker()
    client.tables.setdefault(SOURCES_TABLE, StubTable()).rows = [{"content_hash": None}]

    assert decide(tracker, "abc123").reason is ChangeReason.NEW


# ---------------------------------------------------------------------
#  Recording
# ---------------------------------------------------------------------


def result_for(url: str = "https://example.ca/z5.pdf") -> FetchResult:
    return FetchResult(
        citation_url=url,
        resolved_url="https://cdn.example.ca/z5-2026.pdf",
        content=PDF_BODY,
        content_hash="hash-value",
        content_type="application/pdf",
        fetched_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
    )


def test_record_fetch_upserts_on_the_source_natural_key():
    tracker, client = make_tracker()
    asyncio.run(
        tracker.record_fetch("nb_fredericton", "en", "Zoning By-law Z-5", result_for())
    )

    call = client.tables[SOURCES_TABLE].upserts[0]
    assert call["on_conflict"] == "municipality_id,language,bylaw_name"
    assert call["row"]["content_hash"] == "hash-value"
    assert call["row"]["last_fetched_at"].startswith("2026-09-07")


def test_record_fetch_stores_the_citable_url_not_the_resolved_one():
    """The stable alias is what a citation must point at."""
    tracker, client = make_tracker()
    asyncio.run(
        tracker.record_fetch("nl_stjohns", "en", "Development Regulations", result_for())
    )
    assert client.tables[SOURCES_TABLE].upserts[0]["row"]["source_url"] == (
        "https://example.ca/z5.pdf"
    )


def test_record_fetch_never_sets_the_human_verification_date():
    """A successful download is not a human confirming the text is in force."""
    tracker, client = make_tracker()
    asyncio.run(
        tracker.record_fetch("nb_fredericton", "en", "Zoning By-law Z-5", result_for())
    )

    row = client.tables[SOURCES_TABLE].upserts[0]["row"]
    assert "last_verified_at" not in row
    assert MUNICIPALITIES_TABLE not in client.tables


def test_mark_human_verified_updates_both_tables():
    tracker, client = make_tracker()
    asyncio.run(tracker.mark_human_verified("nb_fredericton", date(2026, 9, 7)))

    municipality = client.tables[MUNICIPALITIES_TABLE].updates[0]
    assert municipality["values"] == {"bylaw_last_verified_at": "2026-09-07"}
    assert municipality["filters"]["id"] == "nb_fredericton"
    assert client.tables[SOURCES_TABLE].updates[0]["values"] == {
        "last_verified_at": "2026-09-07"
    }


# ---------------------------------------------------------------------
#  TLS chain completion (Moncton)
#
#  Moncton's server sends only its leaf certificate and omits the
#  intermediate, so OpenSSL cannot build a path to a trusted root. The
#  chain is completed from the issuer named in the certificate's AIA
#  extension - the same thing a browser does - and never bypassed.
# ---------------------------------------------------------------------


def test_untrusted_source_is_a_source_error():
    """Callers that catch SourceError broadly still handle it."""
    from app.services.source_tracker import SourceError, UntrustedSource

    assert issubclass(UntrustedSource, SourceError)


def test_certificate_errors_are_recognised():
    from app.services.source_tracker import _is_certificate_error

    assert _is_certificate_error(Exception("[SSL: CERTIFICATE_VERIFY_FAILED] ...")) is True
    assert _is_certificate_error(Exception("unable to get local issuer certificate")) is True
    assert _is_certificate_error(Exception("Connection refused")) is False


def test_issuer_url_is_read_from_the_aia_extension():
    """Without an AIA URL there is nothing to fetch and the chain stands."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from datetime import datetime, timedelta, timezone

    from app.services.source_tracker import _issuer_url

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "test")])
    now = datetime.now(timezone.utc)

    without = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    assert _issuer_url(without) is None

    with_aia = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(2)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.AuthorityInformationAccess(
                [
                    x509.AccessDescription(
                        x509.oid.AuthorityInformationAccessOID.CA_ISSUERS,
                        x509.UniformResourceIdentifier("http://ca.example/int.crt"),
                    )
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    assert _issuer_url(with_aia) == "http://ca.example/int.crt"
    _ = serialization  # imported for parity with the module under test
