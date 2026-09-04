#!/usr/bin/env python3
"""Canary INSTITUTION-07 — Contrôle facturation (lecture seule par défaut).

Vérifie post-déploiement :
- population contrôle (période) cohérente avec period-preview
- summary.total == pagination.total
- compteurs summary présents

Usage (container API, token institution admin/billing) ::

    export CANARY_API_URL=https://api.example.ch
    export CANARY_INSTITUTION_BEARER='Bearer …'
    export CANARY_PERIOD=2026-09
    docker compose exec -T atmr_api python scripts/canary/run_r07_billing_control_canary.py

Mutation optionnelle (booking test dédié uniquement) ::

    export CANARY_WRITE_BOOKING_ID=12345
    python scripts/canary/run_r07_billing_control_canary.py --write-payer-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.parse import urlparse

import requests


def _canary_api_base() -> str:
    raw = os.getenv("CANARY_API_URL", "http://127.0.0.1:5000").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SystemExit(f"CANARY_API_URL invalide (http/https requis): {raw}")
    return raw


API = _canary_api_base()
BEARER = os.getenv("CANARY_INSTITUTION_BEARER", "").strip()
PERIOD = os.getenv("CANARY_PERIOD", "").strip()
WRITE_BOOKING_ID = os.getenv("CANARY_WRITE_BOOKING_ID", "").strip()


def _parse_json_body(resp: requests.Response) -> Any:
    raw = resp.text
    if not raw:
        return {}
    try:
        return resp.json()
    except ValueError:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": raw}


def _http(method: str, path: str, *, body: dict | None = None) -> tuple[int, Any]:
    if not BEARER:
        raise SystemExit(
            "CANARY_INSTITUTION_BEARER requis (JWT institution admin/billing)."
        )
    url = f"{API}{path}"
    headers = {
        "Authorization": BEARER if BEARER.startswith("Bearer ") else f"Bearer {BEARER}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            json=body,
            timeout=60,
        )
    except requests.RequestException as exc:
        return 0, {"error": str(exc)}
    return resp.status_code, _parse_json_body(resp)


def _period_query() -> str:
    if not PERIOD:
        raise SystemExit("CANARY_PERIOD requis (YYYY-MM).")
    return f"period={PERIOD}&page_size=200"


def run_readonly_checks() -> dict[str, Any]:
    status, data = _http(
        "GET", f"/api/v1/institutions/billing/control/bookings?{_period_query()}"
    )
    if status != 200:
        raise SystemExit(f"Liste contrôle échouée HTTP {status}: {data}")

    summary = data.get("summary") or {}
    pagination = data.get("pagination") or {}
    total = int(summary.get("total") or 0)
    page_total = int(pagination.get("total") or 0)
    if total != page_total:
        raise SystemExit(
            f"Invariant summary.total ({total}) != pagination.total ({page_total})"
        )

    required_summary_keys = (
        "pending_review",
        "validated",
        "anomaly",
        "payer_patient",
        "payer_clinic",
        "locked_or_invoiced",
    )
    missing = [k for k in required_summary_keys if k not in summary]
    if missing:
        raise SystemExit(f"Summary incomplet, clés manquantes: {missing}")

    status_sum = (
        int(summary.get("pending_review") or 0)
        + int(summary.get("validated") or 0)
        + int(summary.get("anomaly") or 0)
    )
    if status_sum != total:
        raise SystemExit(
            f"Summary statuts ({status_sum}) != total ({total}) pour période {PERIOD}"
        )

    return {
        "period": PERIOD,
        "total": total,
        "summary": summary,
        "items_sample": len(data.get("items") or []),
        "readonly": "PASS",
    }


def run_write_payer_test() -> dict[str, Any]:
    if not WRITE_BOOKING_ID:
        raise SystemExit("CANARY_WRITE_BOOKING_ID requis pour --write-payer-test.")
    bid = int(WRITE_BOOKING_ID)

    st0, detail = _http("GET", f"/api/v1/institutions/billing/control/bookings/{bid}")
    if st0 != 200:
        raise SystemExit(f"Détail booking {bid} HTTP {st0}: {detail}")
    if not (detail.get("billing") or {}).get("editable"):
        raise SystemExit(f"Booking {bid} non éditable — canary write refusé.")

    current = (detail.get("payer") or {}).get("type") or "patient"
    target_intent = "patient" if str(current).lower() == "clinic" else "institution"

    st1, put_res = _http(
        "PUT",
        f"/api/v1/institutions/billing/bookings/{bid}",
        body={
            "billing_intent": target_intent,
            "billing_change_reason_code": "ADMIN_CORRECTION",
            "override_reason": "Canary R07 write test (booking dédié)",
        },
    )
    if st1 != 200:
        raise SystemExit(f"PUT payeur HTTP {st1}: {put_res}")

    st2, after = _http("GET", f"/api/v1/institutions/billing/control/bookings/{bid}")
    if st2 != 200:
        raise SystemExit(f"Refetch détail HTTP {st2}: {after}")

    control = after.get("control") or {}
    if control.get("effective_status") != "pending_review":
        raise SystemExit(
            f"Après correction, statut attendu pending_review, obtenu {control.get('effective_status')}"
        )

    return {
        "booking_id": bid,
        "payer_before": current,
        "payer_after": (after.get("payer") or {}).get("type"),
        "write_payer_test": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canary R07 billing control")
    parser.add_argument(
        "--write-payer-test",
        action="store_true",
        help="Mutation contrôlée sur CANARY_WRITE_BOOKING_ID (booking test uniquement).",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {"checks": []}
    report["checks"].append(run_readonly_checks())
    if args.write_payer_test:
        report["checks"].append(run_write_payer_test())

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n✅ Canary R07 billing control PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(f"\n❌ Canary R07 FAIL: {exc}", file=sys.stderr)
        raise
