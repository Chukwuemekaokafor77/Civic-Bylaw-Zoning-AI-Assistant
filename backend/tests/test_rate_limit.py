"""Unit tests for app.services.rate_limit (Phase 5, Step 2)."""

from __future__ import annotations

import types

from app.config import Settings
from app.services.rate_limit import (
    MAX_SESSION_ID_LENGTH,
    RATE_LIMIT_MESSAGE,
    SESSION_HEADER,
    client_key,
    stream_limits,
)


def settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://test-placeholder.supabase.co",
        "supabase_service_role_key": "test-placeholder",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def request(ip: str = "203.0.113.7", session: str | None = None):
    headers = {SESSION_HEADER: session} if session else {}
    return types.SimpleNamespace(
        headers=headers,
        client=types.SimpleNamespace(host=ip),
        url=types.SimpleNamespace(path="/stream"),
        state=types.SimpleNamespace(),
        scope={"client": (ip, 1234), "headers": [], "type": "http"},
    )


# ---------------------------------------------------------------------
#  Key construction
# ---------------------------------------------------------------------


def test_ip_alone_is_the_key_without_a_session():
    assert client_key(request()) == "203.0.113.7"


def test_session_is_combined_with_the_ip():
    """Session alone would be bypassed by generating a new id per request."""
    key = client_key(request(session="tab-a"))
    assert key.startswith("203.0.113.7")
    assert "tab-a" in key


def test_two_tabs_on_one_ip_get_separate_budgets():
    """Otherwise everyone behind one municipal NAT shares a single limit."""
    assert client_key(request(session="tab-a")) != client_key(request(session="tab-b"))


def test_one_session_from_two_ips_is_not_one_budget():
    a = client_key(request(ip="203.0.113.7", session="same"))
    b = client_key(request(ip="198.51.100.4", session="same"))
    assert a != b


def test_session_id_is_length_capped():
    """It is untrusted client input that lands in a limiter key."""
    key = client_key(request(session="x" * 500))
    assert len(key) < 200
    assert key.count("x") == MAX_SESSION_ID_LENGTH


def test_blank_session_header_falls_back_to_ip():
    assert client_key(request(session="   ")) == "203.0.113.7"


# ---------------------------------------------------------------------
#  Limits and message
# ---------------------------------------------------------------------


def test_limits_come_from_settings():
    assert stream_limits(settings()) == "10/minute;200/day"


def test_limits_track_configuration_changes():
    configured = settings(rate_limit_per_minute=3, rate_limit_per_day=50)
    assert stream_limits(configured) == "3/minute;50/day"


def test_throttle_message_is_not_phrased_as_a_failure():
    """On a free tier this is the system working, not something broken."""
    lowered = RATE_LIMIT_MESSAGE.lower()
    assert "error" not in lowered
    assert "failed" not in lowered
    assert "wait a moment" in lowered
