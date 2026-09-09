"""Unit tests for the readiness probe.

This endpoint reported "ok" for a full day while the application could
not answer a single question. A server had been left running across the
move from Gemini embeddings at 1536 dimensions to Voyage at 1024: it
held a valid key for the provider it no longer used, embedded every
question at the old width, and had each one rejected by a VECTOR(1024)
column. Nothing in the probe could see that, because it only asked
whether a key was present.

So these tests are about what the probe SAYS, not only whether it
passes: it has to name the provider it actually embeds with and report
the width it embeds at, and its deep form has to exercise retrieval end
to end rather than infer health from configuration.
"""

from __future__ import annotations

import asyncio
import types

from app.config import Settings
from app.main import _check_retrieval, readiness
from app.models.schemas import DependencyStatus


def settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://test-placeholder.supabase.co",
        "supabase_service_role_key": "test-placeholder",
        "groq_api_key": "gsk-test",
        "voyage_api_key": "pa-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class StubResult:
    def __init__(self, data):
        self.data = data


class StubQuery:
    """Enough of the Supabase query builder to answer one select."""

    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    async def execute(self):
        return StubResult(self._rows)


class StubClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return StubQuery(self._rows)


def request_with(retriever) -> types.SimpleNamespace:
    # `http` is read before the Supabase check runs, so the stub needs it
    # even when that check is patched out.
    return types.SimpleNamespace(
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(retriever=retriever, http=None)
        )
    )


def retriever_that(rows, retrieve):
    store = types.SimpleNamespace(_client=StubClient(rows))
    return types.SimpleNamespace(_store=store, retrieve=retrieve)


# ---------------------------------------------------------------------
#  What the probe reports
# ---------------------------------------------------------------------


def dependency(body: dict, name: str) -> dict:
    return next(d for d in body["dependencies"] if d["name"] == name)


def read(response) -> dict:
    import json

    return json.loads(response.body)


def test_probe_names_the_embedding_provider_and_its_width(monkeypatch):
    """A probe that only answers "ok" cannot show a provider mismatch."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings())
    monkeypatch.setattr(
        "app.main._check_supabase",
        lambda *_: _ok("supabase"),
    )

    body = read(asyncio.run(readiness(request_with(None))))
    embedding = dependency(body, "embedding_key")

    assert embedding["ok"] is True
    assert "voyage-4-large" in embedding["detail"]
    assert "1024" in embedding["detail"]


def test_a_blank_embedding_key_is_not_ready(monkeypatch):
    """`VOYAGE_API_KEY=` parses to an empty secret, not to None."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings(voyage_api_key=""))
    monkeypatch.setattr("app.main._check_supabase", lambda *_: _ok("supabase"))

    body = read(asyncio.run(readiness(request_with(None))))
    embedding = dependency(body, "embedding_key")

    assert embedding["ok"] is False
    assert "no question can be answered" in embedding["detail"]


def test_shallow_probe_does_not_spend_an_embedding(monkeypatch):
    """Routine probes must not draw on a metered per-minute allowance."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings())
    monkeypatch.setattr("app.main._check_supabase", lambda *_: _ok("supabase"))

    called = False

    async def retrieve(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("shallow readiness must not retrieve")

    request = request_with(retriever_that([{"id": "nb_fredericton"}], retrieve))
    body = read(asyncio.run(readiness(request)))

    assert called is False
    assert [d["name"] for d in body["dependencies"]] == [
        "supabase",
        "embedding_key",
        "groq_key",
    ]


# ---------------------------------------------------------------------
#  The deep probe
# ---------------------------------------------------------------------


def test_deep_probe_reports_a_dimension_mismatch_rather_than_ok():
    """The failure this endpoint could not see.

    A stale process embeds at the old width and the column rejects it.
    Configuration looks perfect; only a real search shows the fault.
    """

    async def retrieve(*_args, **_kwargs):
        raise RuntimeError(
            "expected 1024 dimensions, not 1536"
        )

    request = request_with(retriever_that([{"id": "nb_fredericton"}], retrieve))
    status = asyncio.run(_check_retrieval(request))

    assert status.ok is False
    assert "1536" in (status.detail or "")


def test_deep_probe_passes_when_retrieval_returns_chunks():
    async def retrieve(*_args, **_kwargs):
        return types.SimpleNamespace(chunks=[object(), object()])

    request = request_with(retriever_that([{"id": "nb_moncton"}], retrieve))
    status = asyncio.run(_check_retrieval(request))

    assert status.ok is True
    assert "2 chunk(s)" in (status.detail or "")


def test_deep_probe_says_so_when_there_is_nothing_to_probe():
    async def retrieve(*_args, **_kwargs):
        raise AssertionError("must not retrieve without a municipality")

    request = request_with(retriever_that([], retrieve))
    status = asyncio.run(_check_retrieval(request))

    assert status.ok is False
    assert "no active municipality" in (status.detail or "")


def test_deep_failure_makes_the_service_not_ready(monkeypatch):
    """Not-ready is exactly what "cannot answer anything" should report."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings())
    monkeypatch.setattr("app.main._check_supabase", lambda *_: _ok("supabase"))

    async def retrieve(*_args, **_kwargs):
        raise RuntimeError("expected 1024 dimensions, not 1536")

    request = request_with(retriever_that([{"id": "nb_fredericton"}], retrieve))
    response = asyncio.run(readiness(request, deep=True))

    assert response.status_code == 503
    assert read(response)["status"] == "degraded"


async def _ok(name: str) -> DependencyStatus:
    return DependencyStatus(name=name, ok=True)

