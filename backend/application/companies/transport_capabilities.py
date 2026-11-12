"""Déclarations de transport validées au niveau entreprise.

Une valeur absente reste inconnue et n'ouvre aucune éligibilité.
``false`` signifie que l'entreprise refuse explicitement la prestation.
``true`` est une déclaration volontaire, jamais un remplissage automatique.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ext import db
from security.audit_log import AuditLog

STANDARD_ASSISTANCE_DEFINITION = (
    "Aide non médicale au déplacement et à l'installation du passager, "
    "dans les limites du service proposé. Cela n'inclut pas le portage "
    "dans les escaliers, les soins ni l'accompagnement médical spécialisé."
)
MANUAL_WHEELCHAIR_DEFINITION = (
    "L'entreprise dispose d'un fauteuil manuel adapté à la prestation "
    "déclarée et peut effectivement le fournir. Le nombre indiqué est le "
    "nombre de fauteuils réellement disponibles. Une déclaration sans "
    "fauteuil disponible ne rend pas l'entreprise éligible."
)

_ORDINARY_ASSISTANCE_TYPES = frozenset(
    {
        "",
        "ordinaire",
        "assistance ordinaire",
        "aide ordinaire",
        "aide non medicale",
        "aide non médicale",
    }
)


def declared_standard_assistance(company: Any) -> bool:
    """Seule une déclaration volontaire vraie est éligible."""
    return getattr(company, "provides_standard_assistance", None) is True


def declared_manual_wheelchair(company: Any) -> bool:
    """Vrai seulement si la déclaration et un stock exploitable coexistent."""
    if getattr(company, "provides_manual_wheelchair", None) is not True:
        return False
    count = getattr(company, "manual_wheelchair_count", None)
    return type(count) is int and count >= 1


def specialized_escort_requested(mobility: dict[str, Any]) -> bool:
    """Un libellé autre que l'assistance ordinaire reste non démontré."""
    if not mobility.get("needs_assistance"):
        return False
    kind = str(mobility.get("assistance_type") or "").strip().lower()
    return kind not in _ORDINARY_ASSISTANCE_TYPES


def capability_view(company: Any) -> dict[str, Any]:
    return {
        "provides_standard_assistance": getattr(
            company, "provides_standard_assistance", None
        ),
        "provides_manual_wheelchair": getattr(
            company, "provides_manual_wheelchair", None
        ),
        "manual_wheelchair_count": getattr(company, "manual_wheelchair_count", None),
        "definitions": {
            "standard_assistance": STANDARD_ASSISTANCE_DEFINITION,
            "manual_wheelchair": MANUAL_WHEELCHAIR_DEFINITION,
        },
    }


def apply_transport_capabilities(
    company: Any,
    payload: dict[str, Any],
    *,
    user_id: int | None,
) -> dict[str, Any]:
    """Enregistre les deux déclarations et trace le changement dans la même transaction."""
    assistance = _tri_state(payload, "provides_standard_assistance")
    wheelchair = _tri_state(payload, "provides_manual_wheelchair")
    count = _wheelchair_count(wheelchair, payload.get("manual_wheelchair_count"))
    before = {
        "provides_standard_assistance": getattr(
            company, "provides_standard_assistance", None
        ),
        "provides_manual_wheelchair": getattr(company, "provides_manual_wheelchair", None),
        "manual_wheelchair_count": getattr(company, "manual_wheelchair_count", None),
    }
    company.provides_standard_assistance = assistance
    company.provides_manual_wheelchair = wheelchair
    company.manual_wheelchair_count = count
    after = {
        "provides_standard_assistance": assistance,
        "provides_manual_wheelchair": wheelchair,
        "manual_wheelchair_count": count,
    }
    audit = AuditLog()
    audit.user_id = user_id
    audit.user_type = "company"
    audit.action_type = "company_transport_capabilities_updated"
    audit.action_category = "company"
    audit.action_details = json.dumps({"before": before, "after": after})
    audit.result_status = "success"
    audit.result_message = "Déclarations de transport mises à jour"
    audit.company_id = getattr(company, "id", None)
    audit.resource_type = "company"
    audit.resource_id = str(getattr(company, "id", "") or "")
    audit.additional_metadata = json.dumps({})
    audit.created_at = datetime.now(UTC)
    db.session.add(audit)
    return capability_view(company)


def _tri_state(payload: dict[str, Any], key: str) -> bool | None:
    if key not in payload:
        raise ValueError("La déclaration est incomplète.")
    value = payload[key]
    if value is None or type(value) is bool:
        return value
    raise ValueError("La déclaration doit être vraie, fausse ou non renseignée.")


def _wheelchair_count(declared: bool | None, raw: Any) -> int | None:
    if declared is True:
        if type(raw) is not int or raw < 1:
            raise ValueError(
                "Indiquez le nombre de fauteuils manuels réellement disponibles "
                "(au moins 1)."
            )
        return raw
    if raw not in (None, 0):
        raise ValueError(
            "Le nombre de fauteuils ne s'enregistre que lorsque la prestation est déclarée."
        )
    return None
