#!/usr/bin/env python3
"""Canary C-LEDGER-SERVER isolé — 6 scénarios (HTTP + Redis + PG réels).

Exécuter dans le container API ::

    docker compose exec -T atmr_api python scripts/canary_ledger_server_p0c.py

Périmètre : SERVER Option B uniquement. Pas de purge Redis préventive globale.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = os.getenv("CANARY_API_URL", "http://127.0.0.1:5000").rstrip("/")
OUT_DIR = Path(
    os.getenv(
        "CANARY_OUT_DIR",
        "/app/../docs/ops/_c3_ledger_server_2026-08-14",
    )
)
# Volume backend = /app ; docs hors mount → écrire sous /tmp puis host copie
OUT_DIR = Path(os.getenv("CANARY_OUT_DIR", "/tmp/c3_ledger_server_canary"))
LAT = 46.2044
LON = 6.1432
EVENT_NS = os.getenv("DRIVER_LOCATION_REDIS_EVENT_NS", "atmr:driver_location:event")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _eid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _claim_key(driver_id: int, event_id: str) -> str:
    h = hashlib.sha256(event_id.strip().encode("utf-8")).hexdigest()[:32]
    return f"{EVENT_NS}:{driver_id}:{h}"


def _boot():
    from flask_jwt_extended import create_access_token
    from sqlalchemy import text

    from app import create_app

    app = create_app()
    ctx = app.app_context()
    ctx.push()
    from ext import db, redis_client
    from models import Driver, User

    preferred = int(os.getenv("CANARY_DRIVER_ID", "0") or 0)
    driver = None
    if preferred:
        driver = db.session.get(Driver, preferred)
    if driver is None:
        # Prefer driver historically used in GPS canaries if present, else first active
        for cand in (19, 1, 2):
            driver = db.session.get(Driver, cand)
            if driver is not None:
                break
    if driver is None:
        row = db.session.execute(
            text("SELECT id FROM driver WHERE is_active IS TRUE ORDER BY id LIMIT 1")
        ).scalar()
        driver = db.session.get(Driver, int(row)) if row else None
    if driver is None:
        raise SystemExit("Aucun driver disponible pour le canary")

    user = db.session.get(User, driver.user_id)
    if user is None:
        raise SystemExit(f"User manquant pour driver {driver.id}")

    claims = {
        "role": (
            user.role.value if hasattr(user.role, "value") else str(user.role)
        ),
        "roles": ["driver"],
        "driver_id": driver.id,
        "company_id": driver.company_id,
        "aud": "atmr-api",
        # Doit matcher User.token_version (0 inclus — ne pas faire ``or 1``).
        "token_version": int(getattr(user, "token_version", 0) or 0),
    }
    token = create_access_token(identity=str(user.public_id), additional_claims=claims)
    return {
        "app": app,
        "ctx": ctx,
        "db": db,
        "redis": redis_client,
        "driver": driver,
        "token": token,
    }


def _put(token: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{API}/api/v1/driver/me/location",
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Requested-With": "canary-ledger-server-p0c",
            "X-Forwarded-Proto": "https",
            "X-Idempotency-Key": str(payload.get("location_event_id") or uuid.uuid4()),
        },
    )
    try:
        with urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw[:500]}
            return int(resp.status), parsed
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw[:500]}
        return int(exc.code), parsed
    except (URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}


def _point(
    *,
    event_id: str,
    session_id: str,
    generation: int | None,
    sequence_id: int,
    mission_id: int | None = None,
    location_mode: str = "availability_presence",
    lat_off: float = 0.0,
) -> dict[str, Any]:
    now = _now()
    data: dict[str, Any] = {
        "latitude": LAT + lat_off + sequence_id * 0.00001,
        "longitude": LON + lat_off + sequence_id * 0.00001,
        "recorded_at": now,
        "sent_at": now,
        "location_event_id": event_id,
        "capture_id": event_id,
        "captureId": event_id,
        "tracking_session_id": session_id,
        "sequence_id": sequence_id,
        "location_mode": location_mode,
        "accuracy": 8.0,
        "speed": 0.0,
        "heading": 0.0,
        "is_background": False,
    }
    # Explicit null pour vieux client (clé présente, valeur null)
    data["session_generation"] = generation
    if mission_id is not None:
        data["mission_id"] = mission_id
        data["location_mode"] = "mission_live"
    return data


def _claim_present(redis, driver_id: int, event_id: str) -> bool:
    if redis is None:
        return False
    key = _claim_key(driver_id, event_id)
    return bool(redis.exists(key))


def _claim_ttl(redis, driver_id: int, event_id: str) -> int | None:
    if redis is None:
        return None
    key = _claim_key(driver_id, event_id)
    ttl = redis.ttl(key)
    if ttl is None or int(ttl) == -2:
        return None
    return int(ttl)


def _inject_orphan_claim(redis, driver_id: int, event_id: str, *, ttl: int = 600) -> str:
    key = _claim_key(driver_id, event_id)
    redis.set(key, "1", nx=False, ex=ttl)
    return key


def _pg_rows(db, driver_id: int, event_id: str) -> list[dict[str, Any]]:
    from sqlalchemy import text

    sql = text(
        """
        SELECT id, location_event_id, tracking_session_id, session_generation,
               sequence_id
        FROM driver_location_events
        WHERE driver_id = :did AND location_event_id = :eid
        ORDER BY id
        """
    )
    try:
        rows = db.session.execute(sql, {"did": driver_id, "eid": event_id}).mappings()
        return [dict(r) for r in rows]
    except Exception as exc:
        db.session.rollback()
        return [{"error": str(exc)}]


def _ingest_rows(db, driver_id: int, event_id: str) -> list[dict[str, Any]]:
    from sqlalchemy import text

    sql = text(
        """
        SELECT id, location_event_id, tracking_session_id, session_generation,
               sequence_id
        FROM tracking_ingest_events
        WHERE driver_id = :did AND location_event_id = :eid
        ORDER BY id
        """
    )
    try:
        rows = db.session.execute(sql, {"did": driver_id, "eid": event_id}).mappings()
        return [dict(r) for r in rows]
    except Exception as exc:
        db.session.rollback()
        return [{"error": str(exc)}]


def _record(
    results: list[dict[str, Any]],
    scenario: str,
    *,
    ok: bool,
    detail: dict[str, Any],
) -> None:
    entry = {"scenario": scenario, "pass": ok, **detail}
    results.append(entry)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {scenario}: {json.dumps(detail, default=str)[:500]}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boot = _boot()
    driver = boot["driver"]
    token = boot["token"]
    redis = boot["redis"]
    db = boot["db"]
    driver_id = int(driver.id)
    session_id = f"canary_srv_{uuid.uuid4().hex[:12]}"
    results: list[dict[str, Any]] = []
    metrics = {
        "orphan_claim_after_invalid_ids": 0,
        "duplicate_final_without_persistence": 0,
        "double_write": 0,
        "HOL_after_invalid_item": 0,
        "valid_loc_pg_progression": False,
        "old_client_422": False,
        "claim_release_after_invalid": False,
    }

    print(
        json.dumps(
            {
                "driver_id": driver_id,
                "company_id": driver.company_id,
                "session_id": session_id,
                "api": API,
            }
        )
    )

    # Pré-enregistrement session (FK tracking_ingest_events → tracking_sessions)
    from services.tracking.session_registry import register_tracking_session

    auth = register_tracking_session(
        db.session,
        driver_id=driver_id,
        company_id=int(driver.company_id),
        tracking_session_id=session_id,
        tracking_session_started_at=None,
    )
    db.session.commit()
    server_generation = int(auth["session_generation"])
    print(
        json.dumps(
            {
                "registered_session": session_id,
                "server_generation": server_generation,
            }
        )
    )

    # ------------------------------------------------------------------ S1
    eid1 = _eid("s1_valid")
    gen1 = server_generation
    seq1 = 1
    st, ack = _put(
        token,
        _point(
            event_id=eid1,
            session_id=session_id,
            generation=gen1,
            sequence_id=seq1,
            lat_off=0.001,
        ),
    )
    time.sleep(0.3)
    rows1 = _pg_rows(db, driver_id, eid1) or _ingest_rows(db, driver_id, eid1)
    claim1 = _claim_present(redis, driver_id, eid1)
    s1_ok = (
        st == 200
        and str(ack.get("ack_status") or "") in ("persisted", "accepted", "ingested")
        and (
            str(ack.get("durability") or "") == "persisted_sync"
            or ack.get("ledger_persisted") is True
            or len([r for r in rows1 if "error" not in r]) >= 1
        )
    )
    # Accept also persisted without durability label if PG row exists
    if not s1_ok and len([r for r in rows1 if "error" not in r]) >= 1:
        s1_ok = st in (200, 202) and ack.get("ok") is not False
    if s1_ok and len([r for r in rows1 if "error" not in r]) >= 1:
        metrics["valid_loc_pg_progression"] = True
    _record(
        results,
        "S1_normal_valid_client",
        ok=s1_ok,
        detail={
            "http": st,
            "ack_status": ack.get("ack_status"),
            "accept_reason": ack.get("accept_reason"),
            "durability": ack.get("durability"),
            "ledger_persisted": ack.get("ledger_persisted"),
            "claim_present": claim1,
            "claim_key": _claim_key(driver_id, eid1),
            "pg_rows": len([r for r in rows1 if "error" not in r]),
            "pg": rows1[:2],
            "event_id": eid1,
            "generation": gen1,
            "sequence": seq1,
        },
    )

    # ------------------------------------------------------------------ S2
    eid2 = _eid("s2_nullgen")
    st2, ack2 = _put(
        token,
        _point(
            event_id=eid2,
            session_id=session_id,
            generation=None,
            sequence_id=2,
            lat_off=0.002,
        ),
    )
    time.sleep(0.2)
    claim2 = _claim_present(redis, driver_id, eid2)
    rows2 = _pg_rows(db, driver_id, eid2)
    s2_ok = (
        st2 == 422
        and ack2.get("retryable") is False
        and str(ack2.get("accept_reason") or ack2.get("error_code") or "")
        in ("invalid_ledger_ids", "ledger_ids_missing")
        and claim2 is False
    )
    if claim2:
        metrics["orphan_claim_after_invalid_ids"] += 1
    else:
        metrics["claim_release_after_invalid"] = True
    if st2 == 422 and ack2.get("retryable") is False:
        metrics["old_client_422"] = True
    _record(
        results,
        "S2_old_client_generation_null",
        ok=s2_ok,
        detail={
            "http": st2,
            "ack_status": ack2.get("ack_status"),
            "accept_reason": ack2.get("accept_reason"),
            "error_code": ack2.get("error_code"),
            "retryable": ack2.get("retryable"),
            "claim_present_after": claim2,
            "claim_key": _claim_key(driver_id, eid2),
            "pg_rows": len([r for r in rows2 if "error" not in r]),
            "event_id": eid2,
            "generation": None,
            "sequence": 2,
            "ack": {k: ack2.get(k) for k in (
                "ack_status", "accept_reason", "error", "error_code", "retryable",
                "durability", "ledger_persisted",
            )},
        },
    )

    # ------------------------------------------------------------------ S3
    # Même event_id / generation=null ; lat distincte pour éviter proximity skip.
    time.sleep(1.2)
    st3, ack3 = _put(
        token,
        _point(
            event_id=eid2,
            session_id=session_id,
            generation=None,
            sequence_id=2,
            lat_off=0.012,
        ),
    )
    time.sleep(0.2)
    claim3 = _claim_present(redis, driver_id, eid2)
    reason3 = str(ack3.get("accept_reason") or ack3.get("error_code") or "")
    # Pas de cycle duplicate_unproven ↔ ledger_ids_missing
    cycle_poison = reason3 in (
        "duplicate_event_id_unproven",
        "ledger_ids_missing",
    ) and str(ack3.get("ack_status") or "") == "duplicate"
    s3_ok = (
        st3 == 422
        and ack3.get("retryable") is False
        and reason3 in ("invalid_ledger_ids", "ledger_ids_missing")
        and claim3 is False
        and not cycle_poison
    )
    if claim3:
        metrics["orphan_claim_after_invalid_ids"] += 1
    _record(
        results,
        "S3_retry_same_old_payload",
        ok=s3_ok,
        detail={
            "http": st3,
            "accept_reason": reason3,
            "ack_status": ack3.get("ack_status"),
            "retryable": ack3.get("retryable"),
            "claim_present_after": claim3,
            "cycle_poison": cycle_poison,
            "event_id": eid2,
        },
    )

    # ------------------------------------------------------------------ S4
    # Resend S1 event (valid persisted) → duplicate_persisted, no 2nd row
    before4 = len([r for r in _pg_rows(db, driver_id, eid1) if "error" not in r])
    st4, ack4 = _put(
        token,
        _point(
            event_id=eid1,
            session_id=session_id,
            generation=gen1,
            sequence_id=seq1,
            lat_off=0.001,
        ),
    )
    time.sleep(0.2)
    after4 = len([r for r in _pg_rows(db, driver_id, eid1) if "error" not in r])
    if after4 > max(before4, 1):
        metrics["double_write"] += 1
    ar4 = str(ack4.get("accept_reason") or "")
    as4 = str(ack4.get("ack_status") or "")
    dur4 = ack4.get("durability")
    no_double = after4 <= max(before4, 1) and after4 == before4
    s4_ok = no_double and before4 >= 1 and (
        ar4 == "duplicate_persisted"
        or (as4 == "duplicate" and dur4 == "persisted_sync")
        # same_event durable_ok (idempotence PG) si claim déjà expiré/libéré
        or (as4 == "persisted" and dur4 == "persisted_sync" and after4 == 1)
    )
    _record(
        results,
        "S4_duplicate_persisted",
        ok=s4_ok,
        detail={
            "http": st4,
            "accept_reason": ar4,
            "ack_status": ack4.get("ack_status"),
            "durability": ack4.get("durability"),
            "pg_before": before4,
            "pg_after": after4,
            "claim_present": _claim_present(redis, driver_id, eid1),
            "event_id": eid1,
        },
    )

    # ------------------------------------------------------------------ S5
    eid5 = _eid("s5_orphan")
    key5 = _inject_orphan_claim(redis, driver_id, eid5, ttl=600)
    # Age the claim beyond in-flight window by lowering TTL remaining
    # (age ≈ DEFAULT_TTL - ttl). Set ttl remaining low → age high → unproven.
    # Also test in_flight with fresh claim on another id.
    eid5b = _eid("s5_inflight")
    key5b = _inject_orphan_claim(redis, driver_id, eid5b, ttl=600)
    # Force stale on eid5: set short remaining TTL so age > 15s
    redis.set(key5, "1", ex=30)  # age ≈ 600-30 = 570 → unproven

    st5, ack5 = _put(
        token,
        _point(
            event_id=eid5,
            session_id=session_id,
            generation=server_generation,
            sequence_id=50,
            lat_off=0.005,
        ),
    )
    ar5 = str(ack5.get("accept_reason") or "")
    as5 = str(ack5.get("ack_status") or "")
    final_dup_bad = as5 == "duplicate" and ack5.get("durability") == "persisted_sync"
    if final_dup_bad and ar5 != "duplicate_persisted":
        metrics["duplicate_final_without_persistence"] += 1
    s5a_ok = (
        ar5 in ("duplicate_event_id_unproven", "claim_in_flight")
        and as5 == "ingested_non_persisted"
        and not final_dup_bad
    )

    st5b, ack5b = _put(
        token,
        _point(
            event_id=eid5b,
            session_id=session_id,
            generation=server_generation,
            sequence_id=51,
            lat_off=0.006,
        ),
    )
    ar5b = str(ack5b.get("accept_reason") or "")
    as5b = str(ack5b.get("ack_status") or "")
    final_dup_bad_b = (
        as5b == "duplicate" and ack5b.get("durability") == "persisted_sync"
    )
    if final_dup_bad_b and ar5b != "duplicate_persisted":
        metrics["duplicate_final_without_persistence"] += 1
    s5b_ok = (
        ar5b in ("claim_in_flight", "duplicate_event_id_unproven")
        and as5b == "ingested_non_persisted"
        and not final_dup_bad_b
    )
    s5_ok = s5a_ok and s5b_ok
    _record(
        results,
        "S5_claim_without_persistence_proof",
        ok=s5_ok,
        detail={
            "stale": {
                "http": st5,
                "accept_reason": ar5,
                "ack_status": as5,
                "durability": ack5.get("durability"),
                "claim_after": _claim_present(redis, driver_id, eid5),
                "event_id": eid5,
                "claim_key": key5,
            },
            "inflight": {
                "http": st5b,
                "accept_reason": ar5b,
                "ack_status": as5b,
                "durability": ack5b.get("durability"),
                "claim_after": _claim_present(redis, driver_id, eid5b),
                "event_id": eid5b,
                "claim_key": key5b,
            },
        },
    )

    # ------------------------------------------------------------------ S6
    eid6_poison = _eid("s6_poison")
    st6a, _ack6a = _put(
        token,
        _point(
            event_id=eid6_poison,
            session_id=session_id,
            generation=None,
            sequence_id=60,
            lat_off=0.007,
        ),
    )
    claim6a = _claim_present(redis, driver_id, eid6_poison)
    eid6_ok = _eid("s6_valid")
    gen6 = server_generation
    st6b, ack6b = _put(
        token,
        _point(
            event_id=eid6_ok,
            session_id=session_id,
            generation=gen6,
            sequence_id=61,
            lat_off=0.008,
        ),
    )
    time.sleep(0.3)
    rows6 = _pg_rows(db, driver_id, eid6_ok) or _ingest_rows(db, driver_id, eid6_ok)
    pg6 = len([r for r in rows6 if "error" not in r])
    hol = claim6a or (
        st6a == 422
        and st6b not in (200, 202)
        and pg6 == 0
        and str(ack6b.get("accept_reason") or "").endswith("unproven")
    )
    if hol:
        metrics["HOL_after_invalid_item"] += 1
    if pg6 >= 1:
        metrics["valid_loc_pg_progression"] = True
    s6_ok = (
        st6a == 422
        and claim6a is False
        and (
            (
                st6b == 200
                and (
                    ack6b.get("durability") == "persisted_sync"
                    or ack6b.get("ledger_persisted") is True
                    or pg6 >= 1
                )
            )
            or pg6 >= 1
        )
        and not hol
    )
    _record(
        results,
        "S6_progress_after_invalid",
        ok=s6_ok,
        detail={
            "poison_http": st6a,
            "poison_claim_after": claim6a,
            "valid_http": st6b,
            "valid_ack_status": ack6b.get("ack_status"),
            "valid_durability": ack6b.get("durability"),
            "valid_pg_rows": pg6,
            "valid_event_id": eid6_ok,
            "generation": gen6,
            "HOL": hol,
        },
    )

    # Blocking metrics
    blocking_ok = (
        metrics["orphan_claim_after_invalid_ids"] == 0
        and metrics["duplicate_final_without_persistence"] == 0
        and metrics["double_write"] == 0
        and metrics["HOL_after_invalid_item"] == 0
        and metrics["valid_loc_pg_progression"] is True
        and metrics["old_client_422"] is True
        and metrics["claim_release_after_invalid"] is True
    )
    all_scenarios = all(r["pass"] for r in results)
    verdict = "PASS" if (blocking_ok and all_scenarios) else "FAIL"

    report = {
        "verdict": verdict,
        "driver_id": driver_id,
        "company_id": driver.company_id,
        "session_id": session_id,
        "metrics": metrics,
        "scenarios": results,
        "blocking_ok": blocking_ok,
        "ts": _now(),
    }
    out_path = OUT_DIR / "canary_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("REPORT", out_path)
    print("VERDICT", verdict)
    print("METRICS", json.dumps(metrics))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
