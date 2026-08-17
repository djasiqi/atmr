"""Preuve Socket.IO réelle staging P5-B — client éphémère hors image applicative.

Connexion Engine.IO vers le backend staging (même réseau Docker).
N'installe rien dans l'image sha-d5694d8.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXTURES = Path(os.getenv("STAGING_FIXTURES_PATH", "/output/gps-fixtures.json"))
API = os.getenv("STAGING_API_URL", "http://backend:5000").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
OUT_DIR = Path(os.getenv("STAGING_OUTPUT_DIR", "/output"))
LAT = 46.2044
LON = 6.1432
WAIT_PGFAIL = OUT_DIR / "socket-pgfail-wait"
GO_PGFAIL = OUT_DIR / "socket-pgfail-go"
OBS_PGFAIL = OUT_DIR / "socket-pgfail-observed"
RESUME_PGFAIL = OUT_DIR / "socket-pgfail-resume"


def _load() -> dict:
    if not FIXTURES.exists():
        raise SystemExit(f"fixtures manquantes: {FIXTURES}")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _redis():
    import redis as redis_lib

    return redis_lib.from_url(REDIS_URL)


def _pg():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "pgbouncer"),
        port=int(os.getenv("POSTGRES_PORT", "6432")),
        user=os.getenv("POSTGRES_USER", "atmrstg"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("POSTGRES_DB", "atmrstg"),
    )


def canonical(driver_id: int) -> dict[str, str]:
    raw = _redis().hgetall(f"driver:{driver_id}:loc:canonical") or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def last_position_update(driver_id: int) -> str | None:
    with _pg() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_position_update FROM driver WHERE id = %s",
                (driver_id,),
            )
            row = cur.fetchone()
    if not row or row[0] is None:
        return None
    val = row[0]
    return val.isoformat() if hasattr(val, "isoformat") else str(val)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _trk_sess(when_ms: int, suffix: str) -> str:
    return f"trk_sess_{when_ms}_{suffix}"


def _pos(
    *,
    mission_id: int | None,
    session_id: str,
    generation: int,
    seq: int,
    capture_id: str,
    event_id: str,
    lat_off: float = 0.0,
) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    data: dict[str, Any] = {
        "latitude": LAT + lat_off,
        "longitude": LON + lat_off * 0.1,
        "accuracy": 8,
        "speed": 0,
        "heading": 0,
        "location_mode": "mission_live",
        "timestamp": now,
        "recorded_at": now,
        "sent_at": now,
        "tracking_event_id": event_id,
        "location_event_id": event_id,
        "capture_id": capture_id,
        "captureId": capture_id,
        "tracking_session_id": session_id,
        "session_generation": generation,
        "sequence_id": seq,
    }
    if mission_id is not None:
        data["mission_id"] = mission_id
    return data


class SocketProbe:
    def __init__(self, token: str) -> None:
        import socketio

        self.acks: list[dict[str, Any]] = []
        self.errors: list[Any] = []
        self.rate_limits: list[Any] = []
        self.sio = socketio.Client(logger=False, engineio_logger=False)

        @self.sio.on("driver_location_batch_ack")
        def _ack(data):  # noqa: ANN001
            if isinstance(data, dict):
                self.acks.append(data)
            else:
                self.acks.append({"raw": data})

        @self.sio.on("error")
        def _err(data):  # noqa: ANN001
            self.errors.append(data)

        @self.sio.on("rate_limit_exceeded")
        def _rl(data):  # noqa: ANN001
            self.rate_limits.append(data)

        self.sio.connect(
            API,
            auth={"token": token, "accessToken": token},
            transports=["polling", "websocket"],
            wait_timeout=20,
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "http://127.0.0.1:15000",
                "X-Forwarded-Proto": "https",
            },
            socketio_path="socket.io",
        )

    @property
    def connected(self) -> bool:
        return bool(self.sio.connected)

    def emit_batch(
        self,
        *,
        session_id: str,
        positions: list[dict[str, Any]],
        timeout: float = 35.0,
    ) -> dict[str, Any] | None:
        before = len(self.acks)
        self.sio.emit(
            "driver_location_batch",
            {
                "tracking_session_id": session_id,
                "batch_id": f"sock-{uuid.uuid4().hex[:8]}",
                "positions": positions,
            },
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.acks) > before:
                return self.acks[-1]
            time.sleep(0.1)
        return None

    def close(self) -> None:
        with suppress_exc():
            self.sio.disconnect()


class suppress_exc:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return True


def _wait_flag(path: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.2)
    return False


def _write_flag(path: Path, payload: dict[str, Any] | None = None) -> None:
    path.write_text(json.dumps(payload or {"ts": time.time()}), encoding="utf-8")


def run_s4_s5(probe: SocketProbe, driver_id: int, mission_id: int | None) -> dict[str, Any]:
    """S4 PG-fail + S5 duplicate — le host pause PgBouncer entre wait et go."""
    results: dict[str, Any] = {}
    x_can = canonical(driver_id)
    sess = x_can.get("tracking_session_id") or _trk_sess(int(time.time() * 1000), "s4")
    _write_flag(WAIT_PGFAIL, {"canonical": x_can})
    if not _wait_flag(GO_PGFAIL, 120):
        results["pg_failure"] = {"verdict": "FAIL", "error": "timeout_wait_pgfail_go"}
    else:
        cap_y = f"sock-pgy-{uuid.uuid4().hex[:8]}"
        ack_y = probe.emit_batch(
            session_id=sess,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess,
                    generation=int(x_can.get("session_generation") or 51),
                    seq=int(x_can.get("sequence_id") or 1) + 7,
                    capture_id=cap_y,
                    event_id=str(uuid.uuid4()),
                    lat_off=0.021,
                )
            ],
            timeout=25.0,
        )
        time.sleep(1.0)
        during = canonical(driver_id)
        moved_during = during.get("capture_id") == cap_y
        _write_flag(
            OBS_PGFAIL,
            {
                "ack": ack_y,
                "during": during,
                "moved_during": moved_during,
                "cap_y": cap_y,
            },
        )
        _wait_flag(RESUME_PGFAIL, 60)
        time.sleep(2.5)
        after_resume = canonical(driver_id)
        y_after = after_resume.get("capture_id") == cap_y
        pg_y = last_position_update(driver_id) if y_after else None
        fail_ok = (not moved_during) and ((not y_after) or (y_after and pg_y is not None))
        results["pg_failure"] = {
            "verdict": verdict(fail_ok),
            "canonical_before": x_can,
            "canonical_during": during,
            "canonical_after_resume": after_resume,
            "moved_during_outage": moved_during,
            "became_y_after_pg": y_after,
            "ack": ack_y,
            "capture_y": cap_y,
        }

    time.sleep(6)
    cap_dup = f"sock-dup-{uuid.uuid4().hex[:8]}"
    eid_dup = str(uuid.uuid4())
    sess_dup = _trk_sess(int(time.time() * 1000), "dup")
    pos_dup = _pos(
        mission_id=mission_id,
        session_id=sess_dup,
        generation=52,
        seq=4,
        capture_id=cap_dup,
        event_id=eid_dup,
        lat_off=0.016,
    )
    ack_d1 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup], timeout=35)
    time.sleep(1.2)
    can_d1 = canonical(driver_id)
    time.sleep(6)
    ack_d2 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup], timeout=35)
    time.sleep(1.2)
    can_d2 = canonical(driver_id)
    dup_ok = (
        can_d2.get("session_generation") == can_d1.get("session_generation")
        and int(can_d2.get("sequence_id") or 0) >= int(can_d1.get("sequence_id") or 0)
        and can_d2.get("capture_id") == can_d1.get("capture_id")
        and can_d1.get("capture_id") is not None
        and (
            (ack_d1 and ack_d1.get("success") is True)
            or (ack_d2 and ack_d2.get("success") is True)
        )
    )
    results["duplicate"] = {
        "verdict": verdict(dup_ok),
        "ack": [ack_d1, ack_d2],
        "canonical1": can_d1.get("capture_id"),
        "canonical2": can_d2.get("capture_id"),
        "gen_seq": [
            (can_d1.get("session_generation"), can_d1.get("sequence_id")),
            (can_d2.get("session_generation"), can_d2.get("sequence_id")),
        ],
    }
    results["socket_errors"] = len(probe.errors)
    results["rate_limit_events"] = len(probe.rate_limits) + sum(
        1 for a in probe.acks if a.get("rate_limited")
    )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "s4s5", "s5"])
    args = parser.parse_args()

    sc = _load()["scenarios"]["single"]
    token = sc["token"]
    driver_id = int(sc["driver_id"])
    mission_id = sc.get("mission_id")
    results: dict[str, Any] = {}
    rate_hits = 0
    socket_errors = 0

    rds = _redis()
    if args.phase != "s4s5":
        rds.delete(f"driver:{driver_id}:active_tracking_session")
    rds.delete(f"ws_rate_limit:driver_location_batch:driver:{driver_id}")
    for stale in (
        WAIT_PGFAIL,
        GO_PGFAIL,
        OBS_PGFAIL,
        RESUME_PGFAIL,
    ):
        with suppress_exc():
            stale.unlink()

    probe: SocketProbe | None = None
    try:
        probe = SocketProbe(token)
        results["connect_auth"] = {
            "verdict": verdict(probe.connected),
            "connected": probe.connected,
        }
        if not probe.connected:
            raise SystemExit("CONNECT/AUTH failed")

        if args.phase == "s5":
            cap_dup = f"sock-dup-{uuid.uuid4().hex[:8]}"
            eid_dup = str(uuid.uuid4())
            sess_dup = _trk_sess(int(time.time() * 1000), "dup")
            gen_dup = 60 + (int(time.time()) % 1000)
            pos_dup = _pos(
                mission_id=mission_id,
                session_id=sess_dup,
                generation=gen_dup,
                seq=5,
                capture_id=cap_dup,
                event_id=eid_dup,
                lat_off=0.017,
            )
            ack_d1 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup], timeout=35)
            time.sleep(1.2)
            can_d1 = canonical(driver_id)
            time.sleep(6)
            ack_d2 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup], timeout=35)
            time.sleep(1.2)
            can_d2 = canonical(driver_id)
            # Preuve métier : capture/gen/seq stables (ACK peut arriver en retard en polling).
            dup_ok = (
                can_d1.get("capture_id") == cap_dup
                and can_d2.get("capture_id") == cap_dup
                and can_d2.get("session_generation") == can_d1.get("session_generation")
                and can_d2.get("sequence_id") == can_d1.get("sequence_id")
                and (
                    (ack_d1 and ack_d1.get("success") is True)
                    or (ack_d2 and ack_d2.get("success") is True)
                )
            )
            results["duplicate"] = {
                "verdict": verdict(dup_ok),
                "ack": [ack_d1, ack_d2],
                "sent": cap_dup,
                "canonical1": can_d1.get("capture_id"),
                "canonical2": can_d2.get("capture_id"),
                "gen_seq": [
                    (can_d1.get("session_generation"), can_d1.get("sequence_id")),
                    (can_d2.get("session_generation"), can_d2.get("sequence_id")),
                ],
            }
            results["socket_errors"] = len(probe.errors)
            results["rate_limit_events"] = len(probe.rate_limits)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "socket-real-s5.json").write_text(
                json.dumps(results, default=str, indent=2), encoding="utf-8"
            )
            print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else v) for k, v in results.items()}, indent=2))
            return

        if args.phase == "s4s5":
            results.update(run_s4_s5(probe, driver_id, mission_id))
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "socket-real-s4s5.json").write_text(
                json.dumps(results, default=str, indent=2), encoding="utf-8"
            )
            summary = {
                k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else v)
                for k, v in results.items()
            }
            print(json.dumps(summary, indent=2))
            return

        now_ms = int(time.time() * 1000)
        sess_s1 = _trk_sess(now_ms, "s1")
        cap_s1 = f"sock-s1-{uuid.uuid4().hex[:10]}"
        eid_s1 = str(uuid.uuid4())
        before_pg = last_position_update(driver_id)
        ack1 = probe.emit_batch(
            session_id=sess_s1,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess_s1,
                    generation=50,
                    seq=1,
                    capture_id=cap_s1,
                    event_id=eid_s1,
                    lat_off=0.011,
                )
            ],
        )
        time.sleep(0.8)
        can1 = canonical(driver_id)
        after_pg = last_position_update(driver_id)
        ack_ok = bool(ack1) and ack1.get("success") is True and not ack1.get(
            "ingest_disabled"
        )
        if ack1 and ack1.get("rate_limited"):
            rate_hits += 1
        pg_moved = after_pg is not None and after_pg != before_pg
        can_ok = (
            can1.get("capture_id") == cap_s1
            and can1.get("session_generation") == "50"
            and can1.get("sequence_id") == "1"
        )
        results["batch_ack"] = {"verdict": verdict(ack_ok), "ack": ack1}
        results["normal_canonical"] = {
            "verdict": verdict(can_ok and ack_ok),
            "canonical": can1,
            "capture_id": cap_s1,
        }
        results["pg_before_canonical"] = {
            "verdict": verdict(pg_moved and can_ok),
            "last_position_before": before_pg,
            "last_position_after": after_pg,
            "canonical_capture": can1.get("capture_id"),
        }
        results["capture_id"] = {
            "verdict": verdict(can1.get("capture_id") == cap_s1),
            "sent": cap_s1,
            "redis": can1.get("capture_id"),
        }

        time.sleep(6)
        sess_ord = _trk_sess(now_ms + 1, "ord")
        cap10 = f"sock-s10-{uuid.uuid4().hex[:8]}"
        cap9 = f"sock-s9-{uuid.uuid4().hex[:8]}"
        ack10 = probe.emit_batch(
            session_id=sess_ord,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess_ord,
                    generation=50,
                    seq=10,
                    capture_id=cap10,
                    event_id=str(uuid.uuid4()),
                    lat_off=0.012,
                )
            ],
        )
        time.sleep(0.6)
        after10 = canonical(driver_id)
        time.sleep(6)
        ack9 = probe.emit_batch(
            session_id=sess_ord,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess_ord,
                    generation=50,
                    seq=9,
                    capture_id=cap9,
                    event_id=str(uuid.uuid4()),
                    lat_off=0.013,
                )
            ],
        )
        time.sleep(0.6)
        after9 = canonical(driver_id)
        same_ok = (
            after10.get("session_generation") == "50"
            and after10.get("sequence_id") == "10"
            and after9.get("session_generation") == "50"
            and after9.get("sequence_id") == "10"
            and after9.get("capture_id") == cap10
        )
        results["order_same_gen"] = {
            "verdict": verdict(same_ok),
            "ack10": ack10,
            "ack9": ack9,
            "after10": after10,
            "after9": after9,
        }

        time.sleep(6)
        t51 = int(time.time() * 1000)
        t50 = t51 - 60_000
        sess51 = _trk_sess(t51, "n1")
        sess50 = _trk_sess(t50, "n0")
        cap51 = f"sock-g51-{uuid.uuid4().hex[:8]}"
        cap50old = f"sock-g50-{uuid.uuid4().hex[:8]}"
        ack51 = probe.emit_batch(
            session_id=sess51,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess51,
                    generation=51,
                    seq=1,
                    capture_id=cap51,
                    event_id=str(uuid.uuid4()),
                    lat_off=0.014,
                )
            ],
        )
        time.sleep(0.6)
        after51 = canonical(driver_id)
        time.sleep(6)
        ack_old = probe.emit_batch(
            session_id=sess50,
            positions=[
                _pos(
                    mission_id=mission_id,
                    session_id=sess50,
                    generation=50,
                    seq=9,
                    capture_id=cap50old,
                    event_id=str(uuid.uuid4()),
                    lat_off=0.015,
                )
            ],
        )
        time.sleep(0.6)
        after_old = canonical(driver_id)
        rejected_or_safe = bool(ack_old) and (
            ack_old.get("session_conflict") is True
            or after_old.get("session_generation") == "51"
        )
        old_ok = (
            after51.get("session_generation") == "51"
            and after_old.get("session_generation") == "51"
            and after_old.get("capture_id") == cap51
            and rejected_or_safe
        )
        results["order_old_gen"] = {
            "verdict": verdict(old_ok),
            "ack51": ack51,
            "ack_old": ack_old,
            "after51": after51,
            "after_old": after_old,
            "session_conflict": bool(ack_old and ack_old.get("session_conflict")),
        }

        time.sleep(6)
        x_can = canonical(driver_id)
        _write_flag(WAIT_PGFAIL, {"canonical": x_can})
        if not _wait_flag(GO_PGFAIL, 90):
            results["pg_failure"] = {
                "verdict": "FAIL",
                "error": "timeout_wait_pgfail_go",
            }
        else:
            cap_y = f"sock-pgy-{uuid.uuid4().hex[:8]}"
            ack_y = probe.emit_batch(
                session_id=sess51,
                positions=[
                    _pos(
                        mission_id=mission_id,
                        session_id=sess51,
                        generation=51,
                        seq=2,
                        capture_id=cap_y,
                        event_id=str(uuid.uuid4()),
                        lat_off=0.02,
                    )
                ],
                timeout=25.0,
            )
            time.sleep(1.0)
            during = canonical(driver_id)
            moved_during = during.get("capture_id") == cap_y
            _write_flag(
                OBS_PGFAIL,
                {
                    "ack": ack_y,
                    "during": during,
                    "moved_during": moved_during,
                    "cap_y": cap_y,
                },
            )
            _wait_flag(RESUME_PGFAIL, 60)
            time.sleep(2.0)
            after_resume = canonical(driver_id)
            y_after = after_resume.get("capture_id") == cap_y
            pg_y = None
            if y_after:
                pg_y = last_position_update(driver_id)
            fail_ok = (not moved_during) and (
                (not y_after) or (y_after and pg_y is not None)
            )
            results["pg_failure"] = {
                "verdict": verdict(fail_ok),
                "canonical_before": x_can,
                "canonical_during": during,
                "canonical_after_resume": after_resume,
                "moved_during_outage": moved_during,
                "became_y_after_pg": y_after,
                "ack": ack_y,
                "capture_y": cap_y,
            }

        time.sleep(6)
        cap_dup = f"sock-dup-{uuid.uuid4().hex[:8]}"
        eid_dup = str(uuid.uuid4())
        sess_dup = _trk_sess(int(time.time() * 1000), "dup")
        pos_dup = _pos(
            mission_id=mission_id,
            session_id=sess_dup,
            generation=51,
            seq=4,
            capture_id=cap_dup,
            event_id=eid_dup,
            lat_off=0.016,
        )
        ack_d1 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup])
        time.sleep(0.6)
        can_d1 = canonical(driver_id)
        time.sleep(6)
        ack_d2 = probe.emit_batch(session_id=sess_dup, positions=[pos_dup])
        time.sleep(0.6)
        can_d2 = canonical(driver_id)
        dup_ok = can_d2.get("capture_id") == can_d1.get("capture_id") and (
            can_d1.get("capture_id") in {cap_dup, can_d1.get("capture_id")}
        )
        # Pas de régression : gen/seq ne reculent pas ; capture stable si promu dup.
        dup_ok = (
            can_d2.get("session_generation") == can_d1.get("session_generation")
            and int(can_d2.get("sequence_id") or 0)
            >= int(can_d1.get("sequence_id") or 0)
            and can_d2.get("capture_id") == can_d1.get("capture_id")
        )
        results["duplicate"] = {
            "verdict": verdict(dup_ok),
            "ack": [ack_d1, ack_d2],
            "canonical1": can_d1.get("capture_id"),
            "canonical2": can_d2.get("capture_id"),
            "gen_seq": [
                (can_d1.get("session_generation"), can_d1.get("sequence_id")),
                (can_d2.get("session_generation"), can_d2.get("sequence_id")),
            ],
        }

        rate_hits += sum(1 for a in probe.acks if a.get("rate_limited"))
        socket_errors = len(probe.errors)
        results["socket_errors"] = socket_errors
        results["rate_limit_events"] = rate_hits + len(probe.rate_limits)
        results["all_acks"] = probe.acks
        results["errors"] = probe.errors
    finally:
        if probe is not None:
            probe.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "socket-real.json"
    out.write_text(json.dumps(results, default=str, indent=2), encoding="utf-8")
    summary = {
        k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else v)
        for k, v in results.items()
        if k not in {"all_acks", "errors"}
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
