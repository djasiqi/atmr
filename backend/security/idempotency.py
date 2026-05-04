"""Idempotency middleware for driver booking status updates.

Uses Redis to cache responses keyed by X-Idempotency-Key header.
If the same key is seen within the TTL, the cached response is replayed.
"""

from __future__ import annotations

import json
import logging
from functools import wraps
from typing import Any, Callable

from flask import Response, jsonify, make_response, request

logger = logging.getLogger(__name__)

IDEMPOTENCY_TTL_SECONDS = 300  # 5 minutes


def idempotent(get_context_key: Callable[[], str | None] | None = None):
    """Decorator that adds idempotency to a Flask endpoint.

    Reads ``X-Idempotency-Key`` header. If present and a cached response
    exists in Redis for the composite key, the cached response is returned
    with ``X-Idempotency-Status: replay``.  Otherwise the wrapped handler
    executes normally and the response is cached with
    ``X-Idempotency-Status: new``.

    ``get_context_key`` is an optional callable returning extra context
    (e.g. ``driver_id:booking_id``) to namespace the key.
    """

    def decorator(fn: Callable[..., Any]):
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            idem_key = request.headers.get("X-Idempotency-Key")
            if not idem_key:
                return fn(*args, **kwargs)

            from ext import redis_client

            if redis_client is None:
                return fn(*args, **kwargs)

            ctx = ""
            if get_context_key:
                try:
                    ctx = get_context_key() or ""
                except Exception:
                    ctx = ""

            redis_key = (
                f"idempotency:{ctx}:{idem_key}" if ctx else f"idempotency:{idem_key}"
            )

            try:
                cached = redis_client.get(redis_key)
                if cached:
                    raw: str | bytes = (
                        cached.decode("utf-8")
                        if isinstance(cached, bytes)
                        else str(cached)
                    )
                    data = json.loads(raw)
                    cached_requested = data.get("requested_status")
                    current_body = request.get_json(silent=True) or {}
                    current_requested = (current_body.get("status") or "").upper()
                    if (
                        cached_requested
                        and current_requested
                        and cached_requested != current_requested
                    ):
                        logger.info(
                            "idempotency_conflict key=%s cached=%s requested=%s",
                            redis_key,
                            cached_requested,
                            current_requested,
                        )
                        conflict_resp = make_response(
                            jsonify(
                                {
                                    "error": "Idempotency conflict: same key used with different status",
                                    "cached_status": cached_requested,
                                    "requested_status": current_requested,
                                }
                            ),
                            400,
                        )
                        conflict_resp.headers["X-Idempotency-Status"] = "conflict"
                        try:
                            from services.monitoring.prometheus import (
                                track_driver_booking_status_update,
                            )

                            track_driver_booking_status_update("conflict")
                        except Exception:
                            pass
                        return conflict_resp
                    logger.info(
                        "idempotency_replay key=%s status=%s",
                        redis_key,
                        cached_requested or "?",
                    )
                    try:
                        from services.monitoring.prometheus import (
                            track_driver_booking_status_update,
                        )

                        track_driver_booking_status_update("replay")
                    except Exception:
                        pass
                    resp = make_response(jsonify(data["body"]), data["status_code"])
                    resp.headers["X-Idempotency-Status"] = "replay"
                    return resp
            except Exception as e:
                logger.warning("[Idempotency] Redis read failed: %s", e)

            result = fn(*args, **kwargs)

            # Normalize result to (response_body, status_code)
            resp_obj: Response
            if isinstance(result, tuple):
                resp_obj = make_response(jsonify(result[0]), result[1])
                body = result[0]
                status_code = result[1]
            elif isinstance(result, Response):
                resp_obj = result
                try:
                    body = result.get_json()
                except Exception:
                    body = {}
                status_code = result.status_code
            else:
                resp_obj = make_response(jsonify(result), 200)
                body = result
                status_code = 200

            try:
                req_body = request.get_json(silent=True) or {}
                requested_status = (req_body.get("status") or "").upper()
                cache_payload: dict[str, Any] = {
                    "status_code": status_code,
                    "body": body,
                }
                if requested_status:
                    cache_payload["requested_status"] = requested_status
                redis_client.setex(
                    redis_key,
                    IDEMPOTENCY_TTL_SECONDS,
                    json.dumps(cache_payload),
                )
            except Exception as e:
                logger.warning("[Idempotency] Redis write failed: %s", e)

            resp_obj.headers["X-Idempotency-Status"] = "new"
            try:
                from services.monitoring.prometheus import (
                    track_driver_booking_status_update,
                )

                track_driver_booking_status_update("new")
            except Exception:
                pass
            return resp_obj

        return wrapper

    return decorator
