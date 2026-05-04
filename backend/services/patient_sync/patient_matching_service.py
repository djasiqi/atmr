# services/patient_sync/patient_matching_service.py
"""Service de matching patient sans AVS — suggestions par nom + prénom + DOB.

Règle absolue : jamais d'auto-link sans confirmation humaine.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import func

from models.institution_patient import InstitutionPatient
from models.patient_identity import (
    PatientIdentity,
    PatientMatchRejection,
)

logger = logging.getLogger(__name__)


def find_potential_matches(
    patient_id: int,
    first_name: str,
    last_name: str,
    dob: date | None,
    city: str | None = None,
    phone: str | None = None,
) -> list[dict[str, Any]]:
    """Retourne des suggestions de correspondance avec score et signaux.

    Exclut automatiquement les identités déjà rejetées pour ce patient.

    Args:
        patient_id: ID du patient source (pour exclure les rejets)
        first_name: Prénom
        last_name: Nom
        dob: Date de naissance (optionnel mais fortement recommandé)
        city: Ville (optionnel, améliore le score)
        phone: Téléphone (optionnel, améliore le score)

    Returns:
        Liste de matchs triés par score décroissant
    """
    results: list[dict[str, Any]] = []

    # Identités déjà rejetées pour ce patient
    rejected_ids = {
        r.identity_id
        for r in PatientMatchRejection.query.filter_by(patient_id=patient_id).all()
    }

    # 1. Correspondances dans le Master Index (PatientIdentity)
    identity_query = PatientIdentity.query.filter(
        func.lower(PatientIdentity.canonical_first_name) == first_name.lower(),
        func.lower(PatientIdentity.canonical_last_name) == last_name.lower(),
    )
    if dob:
        identity_query = identity_query.filter(PatientIdentity.canonical_dob == dob)

    for identity in identity_query.all():
        if identity.id in rejected_ids:
            continue

        score = 85 if dob else 50
        signals = ["name_exact"]
        if dob:
            signals.append("dob_exact")

        active_links = [lnk for lnk in identity.links if lnk.is_active]

        results.append(
            {
                "type": "identity",
                "identity_id": identity.id,
                "match_score": score,
                "signals": signals,
                "confidence": identity.confidence_level,
                "avs_last4": identity.avs_last4,
                "linked_entities_count": len(active_links),
            }
        )

    # 2. Si rien dans l'index, chercher dans les patients d'autres institutions
    if not results:
        patient_source = InstitutionPatient.query.get(patient_id)
        source_institution_id = (
            patient_source.institution_id if patient_source else None
        )

        cross_query = InstitutionPatient.query.filter(
            func.lower(InstitutionPatient.first_name) == first_name.lower(),
            func.lower(InstitutionPatient.last_name) == last_name.lower(),
        )
        if dob:
            cross_query = cross_query.filter(InstitutionPatient.dob == dob)

        if source_institution_id:
            cross_query = cross_query.filter(
                InstitutionPatient.institution_id != source_institution_id
            )

        for p in cross_query.limit(20).all():
            score = 70 if dob else 40
            signals = ["name_exact"]
            if dob:
                signals.append("dob_exact")
            if city and p.city and p.city.lower() == city.lower():
                score += 5
                signals.append("city_match")
            if phone and p.phone and p.phone == phone:
                score += 10
                signals.append("phone_match")

            results.append(
                {
                    "type": "cross_patient",
                    "patient_id": p.id,
                    "institution_id": p.institution_id,
                    "institution_name": (p.institution.name if p.institution else "?"),
                    "match_score": score,
                    "signals": signals,
                }
            )

    return sorted(results, key=lambda r: r["match_score"], reverse=True)
