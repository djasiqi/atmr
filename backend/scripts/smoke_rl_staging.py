#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Smoke staging — axe A (suggestions RL).

Exécute les vérifications HTTP automatisables de la checklist
``docs/RL_STAGING_VALIDATION_CHECKLIST.md`` :

- GET ``/api/v1/company_dispatch/rl/status`` (JWT company)
- GET ``/api/v1/company_dispatch/rl/suggestions`` (même auth)
- Vocabulaire ``meta.model_source`` / ``meta.fallback_reason`` lorsque ``meta`` est présent
- Deuxième appel suggestions pour tenter un cache Redis (TTL 30s)

Variables d'environnement (alternative aux options CLI) :

- ``RL_STAGING_BASE_URL`` : ex. ``https://staging.example.com`` (sans slash final)
- ``RL_STAGING_JWT`` : token Bearer (rôle company)

Exemple :

.. code-block:: bash

   set RL_STAGING_BASE_URL=https://api.staging.example.com
   set RL_STAGING_JWT=eyJ...
   python scripts/smoke_rl_staging.py --for-date 2025-03-26

Les logs ``RL_POSTOPT_SKIPPED`` et le panneau semi-auto restent manuels (voir message en fin d'exécution).

Protocole pas à pas et scénarios ``S1_no_model`` … ``S4_manual_ui_and_logs`` :
``docs/RL_STAGING_EXECUTION_PROTOCOL.md``.

Sortie : code 0 si toutes les assertions passent, 1 sinon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

ALLOWED_MODEL_SOURCES = frozenset({"dqn", "basic_fallback", "cache"})
ALLOWED_FALLBACK_REASONS = frozenset({None, "model_missing"})


def _json_loads_maybe(s: str | bytes) -> Any:
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    if not s.strip():
        return None
    return json.loads(s)


def http_get_json(
    base_url: str,
    path: str,
    token: str,
    *,
    query: str = "",
    timeout: float = 45.0,
) -> tuple[int, Any | None, str | None]:
    """GET JSON ; retourne (status, body_json|None, erreur_texte)."""
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query.lstrip('?')}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.getcode() or 200
            return status, _json_loads_maybe(raw), None
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = _json_loads_maybe(body)
        except json.JSONDecodeError:
            parsed = body.decode("utf-8", errors="replace") if body else None
        return e.code, parsed, str(e)
    except urllib.error.URLError as e:
        return 0, None, str(e.reason)


def validate_meta(meta: Any, *, label: str) -> list[str]:
    """Retourne une liste d'erreurs (vide si OK)."""
    errs: list[str] = []
    if not isinstance(meta, dict):
        return [f"{label}: meta doit être un objet JSON"]
    src = meta.get("model_source")
    if src not in ALLOWED_MODEL_SOURCES:
        errs.append(
            f"{label}: meta.model_source={src!r} "
            f"(attendu un de {sorted(ALLOWED_MODEL_SOURCES)})"
        )
    fr = meta.get("fallback_reason")
    if fr not in ALLOWED_FALLBACK_REASONS:
        # API peut renvoyer null JSON -> None en Python
        errs.append(
            f"{label}: meta.fallback_reason={fr!r} "
            f"(attendu null ou 'model_missing')"
        )
    dm = meta.get("duration_ms")
    if dm is not None and not isinstance(dm, (int, float)):
        errs.append(f"{label}: meta.duration_ms doit être un nombre, got {type(dm).__name__}")
    elif dm is not None and float(dm) < 0:
        errs.append(f"{label}: meta.duration_ms < 0")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke staging RL (suggestions + status)")
    p.add_argument(
        "--base-url",
        default=os.environ.get("RL_STAGING_BASE_URL", ""),
        help="URL de base API (sans /api/v1). Env: RL_STAGING_BASE_URL",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("RL_STAGING_JWT", ""),
        help="JWT Bearer (rôle company). Env: RL_STAGING_JWT",
    )
    p.add_argument(
        "--for-date",
        default=None,
        help="Date dispatch YYYY-MM-DD (for_date). Requis sauf avec --print-report-template",
    )
    p.add_argument(
        "--skip-second-call",
        action="store_true",
        help="Ne pas refaire GET suggestions (test cache Redis)",
    )
    p.add_argument(
        "--expect-model-source",
        choices=sorted(ALLOWED_MODEL_SOURCES),
        default=None,
        help="Si meta présent, impose meta.model_source (sinon ignoré)",
    )
    p.add_argument(
        "--print-report-template",
        action="store_true",
        help="Affiche un gabarit de compte-rendu (scénario / attendu / observé / statut / note / action) puis quitte",
    )
    args = p.parse_args()

    if args.print_report_template:
        print(
            "| Scénario | Attendu | Observé | OK/KO | Note | Action |\n"
            "|----------|---------|---------|-------|------|--------|\n"
            "| S1_no_model | 200, meta basic_fallback + model_missing | | | | |\n"
            "| S2_with_model | 200, meta dqn, fallback_reason null | | | | |\n"
            "| S3_cache_repeat | 2e GET, meta.model_source=cache | | | | |\n"
            "| S4_manual_ui_and_logs | UI semi-auto + RL_POSTOPT_SKIPPED | | | | manuel |\n"
        )
        return 0

    if not args.for_date:
        p.error("--for-date est requis (sauf avec --print-report-template)")

    if not args.base_url or not args.token:
        print(
            "Erreur: --base-url et --token requis "
            "(ou RL_STAGING_BASE_URL / RL_STAGING_JWT).",
            file=sys.stderr,
        )
        return 1

    base = args.base_url.rstrip("/")
    token = args.token.strip()
    q_suggestions = (
        f"for_date={args.for_date}&min_confidence=0&limit=20"
    )

    failures: list[str] = []

    # --- Status ---
    st_status, body_status, err_status = http_get_json(
        base, "/api/v1/company_dispatch/rl/status", token
    )
    print(f"[1] GET /company_dispatch/rl/status -> HTTP {st_status}")
    if err_status and st_status == 0:
        failures.append(f"status: {err_status}")
        print("  ", err_status, file=sys.stderr)
    elif st_status != 200:
        failures.append(f"status: HTTP {st_status}")
        print("  body:", body_status)
    elif not isinstance(body_status, dict):
        failures.append("status: corps JSON invalide")
    else:
        if "available" not in body_status:
            failures.append("status: champ 'available' manquant")
        print("  available:", body_status.get("available"), "loaded:", body_status.get("loaded"))

    # --- Suggestions (1er appel) ---
    st1, body1, err1 = http_get_json(
        base,
        "/api/v1/company_dispatch/rl/suggestions",
        token,
        query=q_suggestions,
    )
    print(f"[2] GET /company_dispatch/rl/suggestions (1) -> HTTP {st1}")
    if err1 and st1 == 0:
        failures.append(f"suggestions[1]: {err1}")
        print("  ", err1, file=sys.stderr)
    elif st1 != 200:
        failures.append(f"suggestions[1]: HTTP {st1} (attendu 200 même sans modèle)")
        print("  body:", body1)
    elif body1 is None:
        failures.append("suggestions[1]: corps vide")
    elif not isinstance(body1, dict):
        failures.append("suggestions[1]: JSON doit être un objet")
    else:
        meta = body1.get("meta")
        if meta is None:
            print(
                "  meta: absent (normal si aucun assignment actif / pas de génération — "
                "voir message API)",
            )
        else:
            failures.extend(validate_meta(meta, label="suggestions[1]"))
            print(
                "  meta:",
                {
                    "model_source": meta.get("model_source"),
                    "fallback_reason": meta.get("fallback_reason"),
                    "duration_ms": meta.get("duration_ms"),
                },
            )
            if args.expect_model_source and meta.get("model_source") != args.expect_model_source:
                failures.append(
                    f"suggestions[1]: meta.model_source={meta.get('model_source')!r} "
                    f"!= --expect-model-source={args.expect_model_source!r}"
                )
        if body1.get("error") and st1 == 200:
            # Erreur métier dans 200 — inattendu pour ce smoke
            failures.append(f"suggestions[1]: champ error présent: {body1.get('error')!r}")

    # --- Suggestions (2e appel = cache si TTL et 1er non-cache avec données cachées) ---
    if not args.skip_second_call and st1 == 200 and isinstance(body1, dict):
        st2, body2, err2 = http_get_json(
            base,
            "/api/v1/company_dispatch/rl/suggestions",
            token,
            query=q_suggestions,
        )
        print(f"[3] GET /company_dispatch/rl/suggestions (2) -> HTTP {st2}")
        if err2 and st2 == 0:
            failures.append(f"suggestions[2]: {err2}")
        elif st2 != 200:
            failures.append(f"suggestions[2]: HTTP {st2}")
        elif isinstance(body2, dict):
            m2 = body2.get("meta")
            if m2 and m2.get("model_source") == "cache":
                print("  cache: hit (meta.model_source=cache)")
            else:
                print(
                    "  cache: pas de hit ou meta différent — "
                    "normal si pas de suggestions en cache, Redis absent, ou TTL expiré",
                )
            if m2 is not None:
                failures.extend(validate_meta(m2, label="suggestions[2]"))

    print()
    print("--- Étapes manuelles — scénario S4_manual_ui_and_logs ---")
    print("- Vérifier logs RL_POSTOPT_SKIPPED (dispatch auto) : fast_mode | feature_disabled |")
    print("  model_unavailable | import_error")
    print("- UI : SemiAutoPanel + useRLSuggestions (0 / 1 / N suggestions)")
    print("- Parité : ce script ne couvre que l’URL /api/v1/company_dispatch/ ;")
    print("  valider l’legacy si encore exposé dans votre déploiement.")
    print()

    if failures:
        print("ÉCHECS :")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("OK — smoke RL staging terminé sans erreur détectée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
