#!/usr/bin/env python3
"""Affiche l'effet effectif de DRIVER_LOCATION_V21_* pour un tenant (runbook SRE).

Lecture **variables d'environnement uniquement** (pas d'appel API / DB).
Utile après ``docker compose exec backend env`` ou pour valider un manifeste déployé.

Usage::

    python scripts/print_driver_location_v21_status.py 42
    set COMPANY_ID=42
    python scripts/print_driver_location_v21_status.py
"""

from __future__ import annotations

import os
import sys


def _parse_overrides(raw: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        try:
            cid = int(left.strip())
        except ValueError:
            continue
        out[cid] = right.strip().lower() in {"1", "true", "yes", "on"}
    return out


def effective_v21_for_company(
    *, global_enabled: bool, company_id: int | None, overrides: dict[int, bool]
) -> bool:
    if not global_enabled:
        return False
    if company_id is None:
        return global_enabled
    if company_id in overrides:
        return overrides[company_id]
    return global_enabled


def main() -> int:
    raw_global = os.getenv("DRIVER_LOCATION_V21_ENABLED", "true")
    global_on = raw_global.lower() != "false"
    raw_ov = os.getenv("DRIVER_LOCATION_V21_TENANT_OVERRIDES", "").strip()
    overrides = _parse_overrides(raw_ov)

    cid_arg: int | None = None
    if len(sys.argv) >= 2:
        try:
            cid_arg = int(sys.argv[1].strip())
        except ValueError:
            print(f"company_id invalide: {sys.argv[1]!r}", file=sys.stderr)
            return 2
    else:
        raw_env = os.getenv("COMPANY_ID", "").strip()
        if raw_env:
            try:
                cid_arg = int(raw_env)
            except ValueError:
                print(f"COMPANY_ID invalide: {raw_env!r}", file=sys.stderr)
                return 2

    lines = [
        "DRIVER_LOCATION_V21_ENABLED="
        f"{raw_global!r} -> global_effective={'on' if global_on else 'off'}",
        f"DRIVER_LOCATION_V21_TENANT_OVERRIDES={raw_ov!r}",
    ]
    if overrides:
        lines.append("  overrides_parsed: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in sorted(overrides.items())))
    else:
        lines.append("  overrides_parsed: (vide)")

    if cid_arg is not None:
        eff = effective_v21_for_company(
            global_enabled=global_on, company_id=cid_arg, overrides=overrides
        )
        lines.append(f"company_id={cid_arg} -> v21_effective={'on' if eff else 'off'}")
    else:
        lines.append(
            "(passer un company_id en argument ou COMPANY_ID= pour le statut effectif tenant)"
        )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
