#!/usr/bin/env python3
"""Smoke test charge chat minimal (idempotence + compteurs).

Usage (depuis backend/, avec app Flask configurée) :
  python scripts/chat_load_smoke.py --messages 100

Critères documentés dans le plan Sprint A :
  - 0 doublon pour un client_message_id réutilisé
  - helper find_idempotent_message opérationnel
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ajouter backend au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat load smoke (DB idempotence)")
    parser.add_argument("--messages", type=int, default=100)
    args = parser.parse_args()

    from datetime import UTC, datetime

    from app import create_app
    from ext import db
    from models import Message, SenderRole
    from services.messaging.message_idempotence import find_idempotent_message

    app = create_app()
    duplicates = 0
    latencies: list[float] = []

    with app.app_context():
        sender_id = int(os.getenv("CHAT_SMOKE_SENDER_ID", "1"))
        company_id = int(os.getenv("CHAT_SMOKE_COMPANY_ID", "1"))
        for i in range(args.messages):
            cid = f"smoke-{i}"
            t0 = time.perf_counter()
            existing = find_idempotent_message(sender_id, cid)
            if existing:
                duplicates += 1
                latencies.append(time.perf_counter() - t0)
                continue
            msg = Message(
                sender_id=sender_id,
                company_id=company_id,
                sender_role=SenderRole.DRIVER,
                content=f"smoke {i}",
                timestamp=datetime.now(UTC),
                client_message_id=cid,
                thread_id="dispatch",
            )
            db.session.add(msg)
            db.session.commit()
            latencies.append(time.perf_counter() - t0)

        # Retry idempotent
        raced = find_idempotent_message(sender_id, "smoke-0")
        if raced is None:
            print("FAIL: idempotence retry smoke-0")
            return 1

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95) - 1
    p95 = latencies[max(0, p95_idx)] if latencies else 0.0
    print(f"messages={args.messages} duplicates={duplicates} p95_s={p95:.4f}")
    if p95 > 2.0:
        print("WARN: p95 > 2s (seuil plan staging)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
