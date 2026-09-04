#!/usr/bin/env python3
"""CNY-02B — Idle réel 30 min, backend runtime (certification canary institution P0).

Harness :
  - une seule initialisation Flask / app_context pour toute la durée
  - HTTP réel vers Gunicorn (pas test_client)
  - drift guard UTC wall
  - crash / timeout / env → NOT_EXECUTED ou INVALID (jamais FAIL produit)

Exécution :
  docker compose exec -T atmr_api python /app/scripts/canary/run_cny02b.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

_APP_ROOT = Path(__file__).resolve().parents[2]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

CANARY_EMAIL = "canary-cny01@atmr.test"
CANARY_PASSWORD = "CanaryP0-CNY01!"
CANARY_INSTITUTION_NAME = "CANARY Institution P0"
API_BASE = os.environ.get("CANARY_API_BASE", "http://localhost:5000/api/v1")

IDLE_SECONDS = int(os.environ.get("INSTITUTION_IDLE_TIMEOUT_SECONDS", "1800"))
MAX_TIMING_DRIFT_SECONDS = int(os.environ.get("CANARY_MAX_TIMING_DRIFT", "15"))
WARNING_LEAD_SECONDS = 120
HTTP_TIMEOUT_NORMAL = int(os.environ.get("CANARY_HTTP_TIMEOUT_NORMAL", "30"))
HTTP_TIMEOUT_REFRESH = int(os.environ.get("CANARY_HTTP_TIMEOUT_REFRESH", "60"))
REFRESH_LATENCY_WARN_MS = int(os.environ.get("CANARY_REFRESH_LATENCY_WARN_MS", "30000"))

POLL_ROUTE = "/institutions/requests"
VERDICT_DIR = Path("/app/scripts/canary/verdicts")
BUILD_COMMITS = ("ae3caa25", "2fec3fa4", "03ae1364")


class HarnessStop(Exception):
    """Arrêt protocolaire — pas un échec produit."""

    def __init__(self, verdict: str, reason: str) -> None:
        self.verdict = verdict
        self.reason = reason
        super().__init__(reason)


@dataclass
class HttpResult:
    status: int
    body: dict
    scheduled_at: datetime
    request_started_at: datetime
    response_received_at: datetime
    latency_ms: float
    drift_start_seconds: float | None = None
    error: str | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "http_status": self.status,
            "error_code": self.body.get("error_code") if self.body else None,
            "scheduled_at": _iso(self.scheduled_at),
            "request_started_at": _iso(self.request_started_at),
            "response_received_at": _iso(self.response_received_at),
            "latency_ms": round(self.latency_ms, 1),
            "drift_start_seconds": (
                round(self.drift_start_seconds, 3)
                if self.drift_start_seconds is not None
                else None
            ),
            "error": self.error,
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _git_short_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_APP_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return os.environ.get("CANARY_BUILD_SHA", "").strip() or "unknown"


def _wait_until_utc(target: datetime) -> None:
    while True:
        now = _utcnow()
        if now >= target:
            return
        time.sleep(min(1.0, (target - now).total_seconds()))


def _drift_seconds(expected_at: datetime, observed_at: datetime) -> float:
    return (observed_at - expected_at).total_seconds()


class CanaryRuntime:
    """Contexte Flask unique — aucun create_app() répétitif."""

    def __init__(self) -> None:
        from app import create_app

        self.app = create_app()
        self._ctx = self.app.app_context()
        self._ctx.push()

    def close(self) -> None:
        from ext import db

        db.session.remove()
        self._ctx.pop()

    def ensure_canary_user(self) -> int:
        from ext import db
        from models import Institution, User
        from models.web_session import WebSession
        from security.web_session_service import revoke_web_session

        institution = Institution.query.filter_by(name=CANARY_INSTITUTION_NAME).first()
        if institution is None:
            raise HarnessStop(
                "NOT_EXECUTED",
                "Compte canary absent — exécuter CNY-01 d'abord",
            )
        user = User.query.filter_by(email=CANARY_EMAIL).first()
        if user is None:
            raise HarnessStop("NOT_EXECUTED", "Utilisateur canary absent")
        for ws in WebSession.query.filter_by(user_id=user.id).all():
            if ws.revoked_at is None:
                revoke_web_session(ws.id, reason="CNY-02B reset")
        db.session.commit()
        return int(user.id)

    def get_web_session(self, sid: str) -> dict | None:
        from ext import db
        from models.web_session import WebSession

        ws = db.session.get(WebSession, sid)
        if ws is None:
            return None
        return {
            "id": ws.id,
            "revoked_at": _iso(ws.revoked_at),
            "revoked_reason": ws.revoked_reason,
            "last_interactive_activity_at": _iso(ws.last_interactive_activity_at),
            "created_at": _iso(ws.created_at),
        }

    def decode_jwt(self, token: str) -> dict:
        from flask_jwt_extended import decode_token

        return dict(decode_token(token))


class CanaryClient:
    """Client HTTP vers Gunicorn — indépendant du contexte Flask."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.jar = CookieJar()
        self.opener = urllib_request.build_opener(
            urllib_request.HTTPCookieProcessor(self.jar)
        )
        self.access_token: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        bearer: str | None = None,
        timeout: int = HTTP_TIMEOUT_NORMAL,
        scheduled_at: datetime | None = None,
        deadline_at: datetime | None = None,
    ) -> HttpResult:
        url = f"{self.base}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CNY-02B-canary/2.0",
        }
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib_request.Request(url, data=data, headers=headers, method=method)

        sched = scheduled_at or _utcnow()
        started = _utcnow()
        drift_start = (
            _drift_seconds(deadline_at, started) if deadline_at is not None else None
        )

        try:
            with self.opener.open(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                finished = _utcnow()
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {"_raw": raw}
                return HttpResult(
                    status=resp.status,
                    body=payload,
                    scheduled_at=sched,
                    request_started_at=started,
                    response_received_at=finished,
                    latency_ms=(finished - started).total_seconds() * 1000,
                    drift_start_seconds=drift_start,
                )
        except HTTPError as exc:
            raw = exc.read().decode()
            finished = _utcnow()
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"_raw": raw}
            return HttpResult(
                status=exc.code,
                body=payload,
                scheduled_at=sched,
                request_started_at=started,
                response_received_at=finished,
                latency_ms=(finished - started).total_seconds() * 1000,
                drift_start_seconds=drift_start,
            )
        except (TimeoutError, URLError) as exc:
            finished = _utcnow()
            raise HarnessStop(
                "NOT_EXECUTED",
                f"HTTP {method} {path} timeout/erreur réseau: {exc}",
            ) from exc

    def _sync_tokens(self) -> None:
        for cookie in self.jar:
            if cookie.name == "access_token":
                self.access_token = cookie.value

    def login(self, email: str, password: str) -> HttpResult:
        result = self._request(
            "POST", "/auth/login", body={"email": email, "password": password}
        )
        self._sync_tokens()
        return result

    def refresh(
        self,
        *,
        scheduled_at: datetime | None = None,
        deadline_at: datetime | None = None,
    ) -> HttpResult:
        result = self._request(
            "POST",
            "/auth/refresh-token",
            body={},
            timeout=HTTP_TIMEOUT_REFRESH,
            scheduled_at=scheduled_at,
            deadline_at=deadline_at,
        )
        self._sync_tokens()
        return result

    def get_institution(self, path: str) -> HttpResult:
        if not self.access_token:
            raise HarnessStop(
                "NOT_EXECUTED", "access_token manquant pour GET institution"
            )
        return self._request("GET", path, bearer=self.access_token)


def _checkpoint_entry(
    phase: str,
    expected_at: datetime,
    *,
    event: str,
    observed_at: datetime | None = None,
    http: HttpResult | None = None,
    web_session: dict | None = None,
    extra: dict | None = None,
) -> dict:
    obs = observed_at or (http.response_received_at if http else _utcnow())
    entry: dict[str, Any] = {
        "phase": phase,
        "expected_at": _iso(expected_at),
        "observed_at": _iso(obs),
        "drift_seconds": round(_drift_seconds(expected_at, obs), 3),
        "event": event,
    }
    if http is not None:
        entry.update(http.to_log())
    if web_session is not None:
        entry["web_session"] = web_session
        entry["last_interactive_activity_at"] = web_session.get(
            "last_interactive_activity_at"
        )
        entry["revoked_at"] = web_session.get("revoked_at")
        entry["revoked_reason"] = web_session.get("revoked_reason")
    if extra:
        entry.update(extra)
    return entry


def _assert_schedule_drift(
    phase: str,
    expected_at: datetime,
    action_at: datetime,
) -> None:
    drift = abs(_drift_seconds(expected_at, action_at))
    if drift > MAX_TIMING_DRIFT_SECONDS:
        raise HarnessStop(
            "INVALID",
            f"{phase}: drift {drift:.1f}s > max {MAX_TIMING_DRIFT_SECONDS}s",
        )


def _emit_verdict(
    verdict: str,
    issues: list[str],
    timeline: list[dict],
    evidence: dict,
    *,
    sid: str | None = None,
) -> int:
    first_issue = issues[0] if issues else "—"
    ws_final = evidence.get("web_session_final") or {}

    lines = [
        f"CNY-02B = {verdict}",
        "CNY-02F = PENDING (vérif navigateur — warning ~T28, logout ~T30)",
        "CNY-02  = NOT PASS (nécessite CNY-02B PASS + CNY-02F PASS)",
        "",
        f"BUILD_SHA: {evidence.get('build_head')}",
        f"BUILD_REQUIRED: {' + '.join(BUILD_COMMITS)}",
        f"SID: {sid or evidence.get('sid', '—')}",
        f"WEB_SESSION: {json.dumps(ws_final, ensure_ascii=False)}",
        "FIRST_ACTION: login T00 → idle 30 min UTC wall — sans heartbeat humain",
        f"FIRST_ISSUE: {first_issue}",
        "OBSERVED:",
        *[
            f"  - {e.get('phase')}: drift={e.get('drift_seconds')}s "
            f"latency_ms={e.get('latency_ms', '—')} "
            f"status={e.get('http_status', '—')} {e.get('event')}"
            for e in timeline
        ],
        "EXPECTED:",
        f"  - drift_seconds <= {MAX_TIMING_DRIFT_SECONDS} au démarrage de chaque action",
        "  - last_interactive_activity_at inchangé (polling/refresh ≠ activité humaine)",
        "  - T29:30 refresh START dans tolérance ; 200 ; idle inchangé",
        "  - T30+ → 401 idle_timeout + web_session révoquée",
        "EVIDENCE:",
        json.dumps(evidence, indent=2, ensure_ascii=False, default=str),
        f"VERDICT: {verdict}",
    ]
    if verdict in ("INVALID", "NOT_EXECUTED"):
        lines.insert(4, f"HARNESS_REASON: {first_issue}")

    text = "\n".join(lines)
    print(text)

    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = VERDICT_DIR / f"CNY-02B_{stamp}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n[artefact] {out}")

    return {"PASS": 0, "FAIL": 1, "INVALID": 2, "NOT_EXECUTED": 3}.get(verdict, 1)


def run_cny02b() -> int:
    runtime = CanaryRuntime()
    failures: list[str] = []
    timeline: list[dict] = []
    anomalies: list[str] = []
    evidence: dict = {
        "build_head": _git_short_head(),
        "idle_seconds": IDLE_SECONDS,
        "max_timing_drift_seconds": MAX_TIMING_DRIFT_SECONDS,
        "http_timeout_normal": HTTP_TIMEOUT_NORMAL,
        "http_timeout_refresh": HTTP_TIMEOUT_REFRESH,
        "clock": "UTC wall",
        "harness_version": "2.0",
    }

    try:
        runtime.ensure_canary_user()
        client = CanaryClient(API_BASE)

        login_http = client.login(CANARY_EMAIL, CANARY_PASSWORD)
        if login_http.status >= 500:
            raise HarnessStop(
                "NOT_EXECUTED",
                f"login HTTP {login_http.status} (erreur serveur/environnement)",
            )
        if login_http.status != 200 or not client.access_token:
            raise HarnessStop(
                "NOT_EXECUTED",
                f"login HTTP {login_http.status} — authentification canary impossible",
            )

        sid = str(runtime.decode_jwt(client.access_token).get("sid") or "")
        evidence["sid"] = sid
        if not sid:
            raise HarnessStop("NOT_EXECUTED", "sid absent après login")

        t0 = _utcnow()
        ws_t00 = runtime.get_web_session(sid)
        activity_t00 = (ws_t00 or {}).get("last_interactive_activity_at")
        evidence["t00_utc"] = _iso(t0)
        evidence["last_interactive_activity_at_t00"] = activity_t00
        timeline.append(
            _checkpoint_entry(
                "T00",
                t0,
                event="login institution",
                observed_at=t0,
                http=login_http,
                web_session=ws_t00,
            )
        )

        for phase, delta in [
            ("T+5m", timedelta(minutes=5)),
            ("T+10m", timedelta(minutes=10)),
            ("T+15m", timedelta(minutes=15)),
            ("T+20m", timedelta(minutes=20)),
            ("T+25m", timedelta(minutes=25)),
        ]:
            expected_at = t0 + delta
            _wait_until_utc(expected_at)
            action_at = _utcnow()
            _assert_schedule_drift(phase, expected_at, action_at)

            poll_http = client.get_institution(POLL_ROUTE)
            ws = runtime.get_web_session(sid)
            timeline.append(
                _checkpoint_entry(
                    phase,
                    expected_at,
                    event=f"polling GET {POLL_ROUTE}",
                    observed_at=action_at,
                    http=poll_http,
                    web_session=ws,
                )
            )
            if (ws or {}).get("last_interactive_activity_at") != activity_t00:
                failures.append(f"{phase}: last_interactive_activity_at a bougé")
            err = poll_http.body.get("error_code")
            if poll_http.status != 200:
                failures.append(f"{phase}: attendu 200, reçu {poll_http.status}")
            if poll_http.status == 401 and err == "idle_timeout":
                failures.append(f"{phase}: idle_timeout prématuré")

        expected_t28 = t0 + timedelta(seconds=IDLE_SECONDS - WARNING_LEAD_SECONDS)
        _wait_until_utc(expected_t28)
        action_t28 = _utcnow()
        _assert_schedule_drift("~T28", expected_t28, action_t28)
        timeline.append(
            _checkpoint_entry(
                "~T28",
                expected_t28,
                event="marqueur warning frontend (CNY-02F)",
                observed_at=action_t28,
                extra={"cny02f_note": "vérif navigateur séparée"},
            )
        )

        expected_t2930 = t0 + timedelta(minutes=29, seconds=30)
        _wait_until_utc(expected_t2930)
        refresh_start = _utcnow()
        _assert_schedule_drift("T29:30", expected_t2930, refresh_start)

        refresh_http = client.refresh(
            scheduled_at=expected_t2930,
            deadline_at=expected_t2930,
        )
        ws_refresh = runtime.get_web_session(sid)
        timeline.append(
            _checkpoint_entry(
                "T29:30",
                expected_t2930,
                event="refresh-token (technique)",
                observed_at=refresh_start,
                http=refresh_http,
                web_session=ws_refresh,
            )
        )
        if refresh_http.latency_ms > REFRESH_LATENCY_WARN_MS:
            anomalies.append(
                f"T29:30 refresh latency {refresh_http.latency_ms:.0f}ms "
                f"> seuil anomalie {REFRESH_LATENCY_WARN_MS}ms"
            )
        refresh_err = refresh_http.body.get("error_code")
        if refresh_http.status != 200:
            failures.append(
                f"T29:30 refresh HTTP {refresh_http.status} ({refresh_err})"
            )
        if (ws_refresh or {}).get("last_interactive_activity_at") != activity_t00:
            failures.append("T29:30: refresh a prolongé last_interactive_activity_at")

        expected_t30 = t0 + timedelta(seconds=IDLE_SECONDS + 5)
        _wait_until_utc(expected_t30)
        expire_start = _utcnow()
        _assert_schedule_drift("T30+", expected_t30, expire_start)

        expire_http = client.get_institution("/institutions/me")
        ws_final = runtime.get_web_session(sid)
        expire_err = expire_http.body.get("error_code")
        timeline.append(
            _checkpoint_entry(
                "T30+",
                expected_t30,
                event="GET /institutions/me après deadline idle",
                observed_at=expire_start,
                http=expire_http,
                web_session=ws_final,
            )
        )
        if expire_http.status != 401 or expire_err != "idle_timeout":
            failures.append(
                f"T30+: attendu 401 idle_timeout, reçu {expire_http.status}/{expire_err}"
            )
        if ws_final and ws_final.get("revoked_at") is None:
            failures.append("T30+: web_session.revoked_at NULL")
        elif ws_final and ws_final.get("revoked_reason") != "idle_timeout":
            failures.append(f"T30+: revoked_reason={ws_final.get('revoked_reason')!r}")

        post_refresh = client.refresh(
            scheduled_at=expected_t30,
            deadline_at=expected_t30,
        )
        timeline.append(
            _checkpoint_entry(
                "T30+ refresh",
                expected_t30,
                event="refresh-token après idle",
                http=post_refresh,
                web_session=runtime.get_web_session(sid),
            )
        )
        if post_refresh.status == 200:
            failures.append("T30+: refresh OK après idle (attendu échec)")

        evidence["timeline"] = timeline
        evidence["web_session_final"] = ws_final
        evidence["latency_anomalies"] = anomalies
        verdict = "PASS" if not failures else "FAIL"
        return _emit_verdict(verdict, failures, timeline, evidence, sid=sid)

    except HarnessStop as stop:
        timeline.append(
            {
                "phase": "HARNESS_STOP",
                "event": stop.reason,
                "verdict": stop.verdict,
                "observed_at": _iso(_utcnow()),
            }
        )
        evidence["timeline"] = timeline
        return _emit_verdict(stop.verdict, [stop.reason], timeline, evidence)

    except Exception as exc:
        reason = f"exception harness: {type(exc).__name__}: {exc}"
        evidence["traceback"] = traceback.format_exc()
        timeline.append(
            {
                "phase": "HARNESS_CRASH",
                "event": reason,
                "observed_at": _iso(_utcnow()),
            }
        )
        evidence["timeline"] = timeline
        return _emit_verdict("NOT_EXECUTED", [reason], timeline, evidence)

    finally:
        runtime.close()


if __name__ == "__main__":
    print(
        f"[CNY-02B v2] Idle {IDLE_SECONDS}s UTC | drift≤{MAX_TIMING_DRIFT_SECONDS}s | "
        f"HTTP {HTTP_TIMEOUT_NORMAL}/{HTTP_TIMEOUT_REFRESH}s — machine éveillée",
        flush=True,
    )
    sys.exit(run_cny02b())
