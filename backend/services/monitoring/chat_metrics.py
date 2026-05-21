"""Métriques Prometheus pour le chat (Socket.IO + hub REST)."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram
except ImportError:
    Counter = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]

_CHAT_MESSAGES_SENT: Counter | None = None
_CHAT_MESSAGES_REJECTED: Counter | None = None
_CHAT_MESSAGES_DUPLICATE: Counter | None = None
_CHAT_PAYLOAD_VALIDATION_FAILED: Counter | None = None
_CHAT_SID_LOOKUP_FAILED: Counter | None = None
_CONVERSATION_ROOM_JOIN: Counter | None = None
_CONVERSATION_ROOM_JOIN_FAILED: Counter | None = None
_HUB_THREADS_DURATION: Histogram | None = None
_INBOX_BUILD_DURATION: Histogram | None = None
_ROUTE_DURATION: Histogram | None = None
_SOCKET_CONNECT_BACKFILL: Counter | None = None


def _sent() -> Counter | None:
    global _CHAT_MESSAGES_SENT
    if Counter is None:
        return None
    if _CHAT_MESSAGES_SENT is None:
        _CHAT_MESSAGES_SENT = Counter(
            "chat_messages_sent_total",
            "Messages chat enregistrés avec succès",
            ["channel", "thread_type"],
        )
    return _CHAT_MESSAGES_SENT


def _rejected() -> Counter | None:
    global _CHAT_MESSAGES_REJECTED
    if Counter is None:
        return None
    if _CHAT_MESSAGES_REJECTED is None:
        _CHAT_MESSAGES_REJECTED = Counter(
            "chat_messages_rejected_total",
            "Messages chat rejetés avant insertion",
            ["reason"],
        )
    return _CHAT_MESSAGES_REJECTED


def _duplicate() -> Counter | None:
    global _CHAT_MESSAGES_DUPLICATE
    if Counter is None:
        return None
    if _CHAT_MESSAGES_DUPLICATE is None:
        _CHAT_MESSAGES_DUPLICATE = Counter(
            "chat_messages_duplicate_total",
            "Retries idempotents (message déjà en base)",
            ["channel"],
        )
    return _CHAT_MESSAGES_DUPLICATE


def _validation_failed() -> Counter | None:
    global _CHAT_PAYLOAD_VALIDATION_FAILED
    if Counter is None:
        return None
    if _CHAT_PAYLOAD_VALIDATION_FAILED is None:
        _CHAT_PAYLOAD_VALIDATION_FAILED = Counter(
            "chat_payload_validation_failed_total",
            "Payloads Socket.IO invalides",
            ["event", "reason"],
        )
    return _CHAT_PAYLOAD_VALIDATION_FAILED


def _sid_lookup_failed() -> Counter | None:
    global _CHAT_SID_LOOKUP_FAILED
    if Counter is None:
        return None
    if _CHAT_SID_LOOKUP_FAILED is None:
        _CHAT_SID_LOOKUP_FAILED = Counter(
            "chat_sid_lookup_failed_total",
            "Claims socket introuvables (multi-worker / session expirée)",
            ["event"],
        )
    return _CHAT_SID_LOOKUP_FAILED


def _room_join() -> Counter | None:
    global _CONVERSATION_ROOM_JOIN
    if Counter is None:
        return None
    if _CONVERSATION_ROOM_JOIN is None:
        _CONVERSATION_ROOM_JOIN = Counter(
            "conversation_room_join_total",
            "Join dynamique conversation_* réussi",
        )
    return _CONVERSATION_ROOM_JOIN


def _room_join_failed() -> Counter | None:
    global _CONVERSATION_ROOM_JOIN_FAILED
    if Counter is None:
        return None
    if _CONVERSATION_ROOM_JOIN_FAILED is None:
        _CONVERSATION_ROOM_JOIN_FAILED = Counter(
            "conversation_room_join_failed_total",
            "Échecs join dynamique conversation_*",
            ["reason"],
        )
    return _CONVERSATION_ROOM_JOIN_FAILED


def inc_chat_message_sent(*, channel: str, thread_type: str = "unknown") -> None:
    c = _sent()
    if c is not None:
        c.labels(channel=channel, thread_type=thread_type).inc()


def inc_chat_message_rejected(reason: str) -> None:
    c = _rejected()
    if c is not None:
        c.labels(reason=reason).inc()


def inc_chat_message_duplicate(*, channel: str) -> None:
    c = _duplicate()
    if c is not None:
        c.labels(channel=channel).inc()


def inc_chat_payload_validation_failed(*, event: str, reason: str) -> None:
    c = _validation_failed()
    if c is not None:
        c.labels(event=event, reason=reason).inc()


def inc_chat_sid_lookup_failed(*, event: str) -> None:
    c = _sid_lookup_failed()
    if c is not None:
        c.labels(event=event).inc()


def inc_conversation_room_join() -> None:
    c = _room_join()
    if c is not None:
        c.inc()


def inc_conversation_room_join_failed(reason: str) -> None:
    c = _room_join_failed()
    if c is not None:
        c.labels(reason=reason).inc()


def _hub_threads_duration() -> Histogram | None:
    global _HUB_THREADS_DURATION
    if Histogram is None:
        return None
    if _HUB_THREADS_DURATION is None:
        _HUB_THREADS_DURATION = Histogram(
            "hub_threads_duration_ms",
            "Durée GET hub/threads",
            ["route"],
            buckets=(25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
        )
    return _HUB_THREADS_DURATION


def _inbox_build_duration() -> Histogram | None:
    global _INBOX_BUILD_DURATION
    if Histogram is None:
        return None
    if _INBOX_BUILD_DURATION is None:
        _INBOX_BUILD_DURATION = Histogram(
            "inbox_build_duration_ms",
            "Durée construction inbox",
            ["kind"],
            buckets=(25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
        )
    return _INBOX_BUILD_DURATION


def _route_duration() -> Histogram | None:
    global _ROUTE_DURATION
    if Histogram is None:
        return None
    if _ROUTE_DURATION is None:
        _ROUTE_DURATION = Histogram(
            "messaging_route_duration_ms",
            "Durée routes messaging",
            ["route"],
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
        )
    return _ROUTE_DURATION


def _socket_connect_backfill() -> Counter | None:
    global _SOCKET_CONNECT_BACKFILL
    if Counter is None:
        return None
    if _SOCKET_CONNECT_BACKFILL is None:
        _SOCKET_CONNECT_BACKFILL = Counter(
            "socket_connect_backfill_total",
            "Backfill déclenché au connect socket",
        )
    return _SOCKET_CONNECT_BACKFILL


def observe_hub_threads_duration_ms(duration_ms: int, route: str = "hub_threads") -> None:
    h = _hub_threads_duration()
    if h is not None:
        h.labels(route=route).observe(max(0, duration_ms))


def observe_inbox_build_duration_ms(duration_ms: int, kind: str) -> None:
    h = _inbox_build_duration()
    if h is not None:
        h.labels(kind=kind).observe(max(0, duration_ms))


def observe_route_duration_ms(route: str, duration_ms: int) -> None:
    h = _route_duration()
    if h is not None:
        h.labels(route=route).observe(max(0, duration_ms))


def inc_socket_connect_backfill() -> None:
    c = _socket_connect_backfill()
    if c is not None:
        c.inc()
