#!/usr/bin/env python3
"""Récupère les receipts Expo pour diagnostiquer les push non reçues (app killed).

Usage:
    python -m scripts.fetch_expo_receipts TICKET_ID1 TICKET_ID2 ...
    python -m scripts.fetch_expo_receipts XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

Les receipts sont disponibles ~15 min après l'envoi. Si pas encore dispo, réessayer.

Interprétation:
    - status=ok → Expo a donné à FCM/APNS (si rien sur device → Android/permission/channel)
    - DeviceNotRegistered → token invalide (réinstall app, etc.)
    - MessageTooBig → payload > 4KB
    - InvalidCredentials → config FCM/APNS
"""
from __future__ import annotations

import json
import sys

import requests


def fetch_receipts(ticket_ids: list[str]) -> dict:
    """Appelle Expo getReceipts API."""
    resp = requests.post(
        "https://exp.host/--/api/v2/push/getReceipts",
        json={"ids": ticket_ids},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        print("\nExemple: python -m scripts.fetch_expo_receipts XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX", file=sys.stderr)
        return 1

    ticket_ids = [a.strip() for a in sys.argv[1:] if a.strip()]
    if not ticket_ids:
        print("Aucun ticket ID fourni.", file=sys.stderr)
        return 1

    try:
        data = fetch_receipts(ticket_ids)
    except requests.RequestException as e:
        print(f"Erreur réseau: {e}", file=sys.stderr)
        return 1

    receipts = data.get("data") or {}
    errors = data.get("errors") or []

    if errors:
        print("Erreurs globales:", json.dumps(errors, indent=2))

    print("\n=== PUSH_PROOF receipts ===\n")
    for tid in ticket_ids:
        rec = receipts.get(tid)
        if rec is None:
            print(f"{tid}: (pas encore dispo — réessayer dans 15 min)")
            continue
        status = rec.get("status", "?")
        msg = rec.get("message", "")
        details = rec.get("details", {}) or {}
        err = details.get("error", "")
        print(f"{tid}:")
        print(f"  status={status}")
        if msg:
            print(f"  message={msg}")
        if err:
            print(f"  details.error={err}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
