"""Audit read-only de la topologie institution (TransportRequest / legs / bookings).

Usage (Docker obligatoire) :
  docker compose exec -T atmr_api python scripts/ops/audit_transport_topology_readonly.py
  docker compose exec -T atmr_api python scripts/ops/audit_transport_topology_readonly.py --sample 10

Aucune mutation. Code de sortie 0 si aucune anomalie bloquante, 1 sinon.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text

from app import create_app
from ext import db


def _rows(sql: str, **params: Any) -> list[dict[str, Any]]:
    result = db.session.execute(text(sql), params)
    return [dict(row._mapping) for row in result]


def _count(sql: str, **params: Any) -> int:
    row = db.session.execute(text(sql), params).first()
    if row is None:
        return 0
    val = row[0]
    return int(val or 0)


def run_audit(*, sample: int, production_scope: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "read_only",
        "production_scope": production_scope,
        "sections": {},
    }
    tr_status_filter = (
        "AND tr.status IN ('CONVERTED', 'SENT', 'ACCEPTED')"
        if production_scope
        else ""
    )
    blocking_total = 0

    def section(
        key: str,
        count: int,
        samples: list[dict[str, Any]],
        *,
        is_blocking: bool = True,
    ) -> None:
        nonlocal blocking_total
        if is_blocking:
            blocking_total += count
        report["sections"][key] = {
            "count": count,
            "blocking": is_blocking,
            "samples": samples[:sample],
        }

    # 1) Return legs sans booking
    c1 = _count(
        f"""
        SELECT COUNT(*)
        FROM transport_request_legs l
        JOIN transport_requests tr ON tr.id = l.transport_request_id
        WHERE l.is_return_stop = TRUE AND l.booking_id IS NULL
        {tr_status_filter}
        """
    )
    s1 = _rows(
        f"""
        SELECT l.id AS leg_id, l.transport_request_id, tr.external_reference,
               tr.status, tr.route_group_id
        FROM transport_request_legs l
        JOIN transport_requests tr ON tr.id = l.transport_request_id
        WHERE l.is_return_stop = TRUE AND l.booking_id IS NULL
        {tr_status_filter}
        ORDER BY l.id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("return_legs_without_booking_id", c1, s1)

    # 2) Plusieurs is_return_stop par TR
    c2 = _count(
        """
        SELECT COUNT(*) FROM (
            SELECT transport_request_id
            FROM transport_request_legs
            WHERE is_return_stop = TRUE
            GROUP BY transport_request_id
            HAVING COUNT(*) > 1
        ) x
        """
    )
    s2 = _rows(
        """
        SELECT tr.id AS transport_request_id, tr.external_reference, tr.route_group_id,
               COUNT(*) AS return_stop_count
        FROM transport_request_legs l
        JOIN transport_requests tr ON tr.id = l.transport_request_id
        WHERE l.is_return_stop = TRUE
        GROUP BY tr.id, tr.external_reference, tr.route_group_id
        HAVING COUNT(*) > 1
        ORDER BY return_stop_count DESC, tr.id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("multiple_return_stops_per_tr", c2, s2)

    # 3) TR multi_stop/return_to_institution avec legs mais leg(s) sans booking (hors return-only)
    c3 = _count(
        """
        SELECT COUNT(*)
        FROM transport_request_legs l
        JOIN transport_requests tr ON tr.id = l.transport_request_id
        WHERE l.booking_id IS NULL
          AND tr.status IN ('CONVERTED', 'SENT', 'ACCEPTED')
          AND (tr.multi_stop = TRUE OR tr.return_to_institution = TRUE)
        """
    )
    s3 = _rows(
        """
        SELECT l.id AS leg_id, l.transport_request_id, l.sequence_index,
               l.is_return_stop, tr.external_reference, tr.status, tr.route_group_id
        FROM transport_request_legs l
        JOIN transport_requests tr ON tr.id = l.transport_request_id
        WHERE l.booking_id IS NULL
          AND tr.status IN ('CONVERTED', 'SENT', 'ACCEPTED')
          AND (tr.multi_stop = TRUE OR tr.return_to_institution = TRUE)
        ORDER BY l.id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("active_tr_legs_missing_booking_id", c3, s3)

    # 4) Bookings institution (route_group_id) sans leg associé
    c4 = _count(
        """
        SELECT COUNT(*)
        FROM booking b
        WHERE b.route_group_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM transport_request_legs l WHERE l.booking_id = b.id
          )
        """
    )
    s4 = _rows(
        """
        SELECT b.id AS booking_id, b.route_group_id, b.route_sequence_number,
               b.is_return, b.parent_booking_id, b.status, b.created_at
        FROM booking b
        WHERE b.route_group_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM transport_request_legs l WHERE l.booking_id = b.id
          )
        ORDER BY b.id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("bookings_with_route_group_without_leg", c4, s4)

    # 5) route_group_id partagé par plusieurs TransportRequest
    c5 = _count(
        """
        SELECT COUNT(*) FROM (
            SELECT route_group_id
            FROM transport_requests
            WHERE route_group_id IS NOT NULL
            GROUP BY route_group_id
            HAVING COUNT(*) > 1
        ) x
        """
    )
    s5 = _rows(
        """
        SELECT route_group_id, COUNT(*) AS tr_count,
               ARRAY_AGG(id ORDER BY id) AS transport_request_ids
        FROM transport_requests
        WHERE route_group_id IS NOT NULL
        GROUP BY route_group_id
        HAVING COUNT(*) > 1
        ORDER BY tr_count DESC, route_group_id
        LIMIT :lim
        """,
        lim=sample,
    )
    section("duplicate_route_group_id_on_transport_requests", c5, s5)

    # 6) Incohérence séquence leg vs booking.route_sequence_number
    c6 = _count(
        """
        SELECT COUNT(*)
        FROM transport_request_legs l
        JOIN booking b ON b.id = l.booking_id
        WHERE l.route_sequence_number IS DISTINCT FROM b.route_sequence_number
        """
    )
    s6 = _rows(
        """
        SELECT l.id AS leg_id, l.transport_request_id, l.route_sequence_number AS leg_seq,
               b.id AS booking_id, b.route_sequence_number AS booking_seq, b.route_group_id
        FROM transport_request_legs l
        JOIN booking b ON b.id = l.booking_id
        WHERE l.route_sequence_number IS DISTINCT FROM b.route_sequence_number
        ORDER BY l.id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("leg_booking_route_sequence_mismatch", c6, s6, is_blocking=False)

    # 7) Doublon classique + return leg (pattern 38907/39042)
    c7 = _count(
        """
        WITH return_leg AS (
            SELECT tr.id AS tr_id, tr.route_group_id, l.booking_id AS return_booking_id
            FROM transport_request_legs l
            JOIN transport_requests tr ON tr.id = l.transport_request_id
            WHERE l.is_return_stop = TRUE AND l.booking_id IS NOT NULL
        ),
        outbound AS (
            SELECT tr.id AS tr_id, tr.route_group_id, MIN(l.booking_id) AS outbound_booking_id
            FROM transport_request_legs l
            JOIN transport_requests tr ON tr.id = l.transport_request_id
            WHERE l.is_return_stop = FALSE AND l.booking_id IS NOT NULL
            GROUP BY tr.id, tr.route_group_id
        ),
        dup AS (
            SELECT rl.tr_id, rl.route_group_id, rl.return_booking_id,
                   o.outbound_booking_id, cb.id AS classic_return_id
            FROM return_leg rl
            JOIN outbound o ON o.tr_id = rl.tr_id
            JOIN booking cb ON cb.parent_booking_id = o.outbound_booking_id
                           AND cb.is_return = TRUE
            WHERE cb.id <> rl.return_booking_id
        )
        SELECT COUNT(*) FROM dup
        """
    )
    s7 = _rows(
        """
        WITH return_leg AS (
            SELECT tr.id AS tr_id, tr.external_reference, tr.route_group_id,
                   l.booking_id AS return_booking_id, l.id AS return_leg_id
            FROM transport_request_legs l
            JOIN transport_requests tr ON tr.id = l.transport_request_id
            WHERE l.is_return_stop = TRUE AND l.booking_id IS NOT NULL
        ),
        outbound AS (
            SELECT tr.id AS tr_id, MIN(l.booking_id) AS outbound_booking_id
            FROM transport_request_legs l
            JOIN transport_requests tr ON tr.id = l.transport_request_id
            WHERE l.is_return_stop = FALSE AND l.booking_id IS NOT NULL
            GROUP BY tr.id
        ),
        dup AS (
            SELECT rl.tr_id, rl.external_reference, rl.route_group_id,
                   rl.return_leg_id, rl.return_booking_id,
                   o.outbound_booking_id, cb.id AS classic_return_id,
                   cb.status AS classic_status, cb.scheduled_time AS classic_scheduled_time,
                   cb.created_at AS classic_created_at
            FROM return_leg rl
            JOIN outbound o ON o.tr_id = rl.tr_id
            JOIN booking cb ON cb.parent_booking_id = o.outbound_booking_id
                           AND cb.is_return = TRUE
            WHERE cb.id <> rl.return_booking_id
        )
        SELECT * FROM dup
        ORDER BY classic_created_at DESC NULLS LAST, tr_id DESC
        LIMIT :lim
        """,
        lim=sample,
    )
    section("classic_return_duplicate_with_return_leg", c7, s7)

    # 8) Cas référence prod connus (si présents)
    ref_ids = (38906, 38907, 39042, 4464)
    s8 = _rows(
        """
        SELECT 'booking' AS kind, b.id, b.is_return, b.parent_booking_id,
               b.route_group_id, b.route_sequence_number, b.status::text AS status,
               b.scheduled_time, b.time_confirmed, b.driver_id, b.created_at
        FROM booking b
        WHERE b.id = ANY(:ids)
        UNION ALL
        SELECT 'transport_request' AS kind, tr.id, NULL::boolean, tr.booking_id,
               tr.route_group_id, NULL::integer, tr.status::text,
               tr.scheduled_time, tr.pickup_time_confirmed, NULL::integer, tr.created_at
        FROM transport_requests tr
        WHERE tr.id = ANY(:ids)
        ORDER BY kind, id
        """,
        ids=list(ref_ids),
    )
    report["sections"]["reference_ids_snapshot"] = {
        "count": len(s8),
        "blocking": False,
        "samples": s8,
    }

    report["summary"] = {
        "blocking_anomaly_count": blocking_total,
        "status": "OK" if blocking_total == 0 else "ANOMALIES",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit topologie transport (read-only)")
    parser.add_argument("--sample", type=int, default=15, help="Nombre d'exemples par section")
    parser.add_argument(
        "--production-scope",
        action="store_true",
        help="Limite aux TR CONVERTED/SENT/ACCEPTED (recommandé prod)",
    )
    parser.add_argument("--json", action="store_true", help="Sortie JSON brute")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        report = run_audit(sample=max(1, args.sample), production_scope=args.production_scope)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print("=== Audit topologie transport (READ-ONLY) ===")
        print(f"Généré : {report['generated_at']}\n")
        for key, section in report["sections"].items():
            if key == "reference_ids_snapshot":
                print(f"--- {key} ---")
                for row in section["samples"]:
                    print(f"  {row}")
                print()
                continue
            flag = "BLOCK" if section.get("blocking") else "INFO"
            print(f"[{flag}] {key}: {section['count']}")
            for row in section["samples"]:
                print(f"  {row}")
            print()

        print("=== Résumé ===")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    return 0 if report["summary"]["blocking_anomaly_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
