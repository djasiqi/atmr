"""Résolution en masse de ``Booking.institution_patient_id`` depuis ``TransportRequest``.

Les bookings créés avant l'ajout de la colonne n'ont pas de patient institutionnel
rattaché : chacun retombe alors sur la clé ``legacy-institution-booking:{id}``, ce qui
produit une opportunité de facturation par transport au lieu d'une par patient.

Ce module reconstruit le lien en quatre passes (demande directe, parent A/R,
``route_group_id``, puis ``BillingParty.external_ref``) et peut le persister. Il est
partagé par le script de backfill et par la lecture des opportunités, pour garantir
une règle unique.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _patient_ids_from_billing_parties(
    billing_party_ids: set[int],
) -> dict[int, int]:
    """``billing_party_id -> institution_patient_id`` via ``external_ref``.

    Les ``BillingParty`` de type patient sont créés avec
    ``external_ref = "patient:{InstitutionPatient.id}"``, ce qui en fait un lien
    d'identité fiable même quand la demande de transport source a disparu.
    """
    from models import BillingParty, InstitutionPatient

    if not billing_party_ids:
        return {}

    rows = (
        BillingParty.query.with_entities(BillingParty.id, BillingParty.external_ref)
        .filter(
            BillingParty.id.in_(sorted(billing_party_ids)),
            BillingParty.external_ref.like("patient:%"),
        )
        .all()
    )
    candidates: dict[int, int] = {}
    for billing_party_id, external_ref in rows:
        try:
            patient_id = int(str(external_ref).split(":", 1)[1])
        except (IndexError, TypeError, ValueError):
            continue
        candidates[int(billing_party_id)] = patient_id

    if not candidates:
        return {}

    # Ne jamais écrire une FK vers un patient supprimé.
    existing = {
        int(row[0])
        for row in InstitutionPatient.query.with_entities(InstitutionPatient.id)
        .filter(InstitutionPatient.id.in_(sorted(set(candidates.values()))))
        .all()
    }
    return {
        billing_party_id: patient_id
        for billing_party_id, patient_id in candidates.items()
        if patient_id in existing
    }


def build_institution_patient_mapping(
    booking_ids: Iterable[int],
    *,
    parent_ids_by_booking: dict[int, int] | None = None,
    route_group_by_booking: dict[int, str] | None = None,
    billing_party_by_booking: dict[int, int] | None = None,
) -> tuple[dict[int, int], set[int]]:
    """Construit ``booking_id -> institution_patient_id``.

    Retourne le mapping résolu et l'ensemble des bookings ambigus
    (plusieurs patients candidats), qui ne doivent jamais être écrits.
    """
    from models import TransportRequest

    ids = {int(b) for b in booking_ids}
    if not ids:
        return {}, set()

    parents = parent_ids_by_booking or {}
    groups = route_group_by_booking or {}
    billing_parties = billing_party_by_booking or {}

    candidates: dict[int, set[int]] = defaultdict(set)

    # 1) Demande rattachée directement au booking (ou à son parent A/R)
    lookup_ids = ids | {int(p) for p in parents.values() if p is not None}
    direct_rows = (
        TransportRequest.query.with_entities(
            TransportRequest.booking_id, TransportRequest.patient_id
        )
        .filter(
            TransportRequest.booking_id.in_(lookup_ids),
            TransportRequest.patient_id.isnot(None),
        )
        .all()
    )
    by_source_booking: dict[int, set[int]] = defaultdict(set)
    for source_booking_id, patient_id in direct_rows:
        by_source_booking[int(source_booking_id)].add(int(patient_id))

    for booking_id in ids:
        direct = by_source_booking.get(booking_id)
        if direct:
            candidates[booking_id].update(direct)
            continue
        parent_id = parents.get(booking_id)
        if parent_id is not None:
            inherited = by_source_booking.get(int(parent_id))
            if inherited:
                candidates[booking_id].update(inherited)

    # 2) route_group_id : un groupe multi-étapes concerne un seul patient
    unresolved_groups = {
        str(groups[bid]) for bid in ids if bid not in candidates and groups.get(bid)
    }
    if unresolved_groups:
        group_rows = (
            TransportRequest.query.with_entities(
                TransportRequest.route_group_id, TransportRequest.patient_id
            )
            .filter(
                TransportRequest.route_group_id.in_(sorted(unresolved_groups)),
                TransportRequest.patient_id.isnot(None),
            )
            .all()
        )
        patients_by_group: dict[str, set[int]] = defaultdict(set)
        for route_group_id, patient_id in group_rows:
            patients_by_group[str(route_group_id)].add(int(patient_id))
        for booking_id in ids:
            if booking_id in candidates:
                continue
            group_id = groups.get(booking_id)
            if not group_id:
                continue
            found = patients_by_group.get(str(group_id))
            if found:
                candidates[booking_id].update(found)

    # 3) BillingParty patient : dernier lien fiable quand la demande a disparu
    unresolved_parties = {
        int(billing_parties[bid])
        for bid in ids
        if bid not in candidates and billing_parties.get(bid) is not None
    }
    if unresolved_parties:
        patient_by_party = _patient_ids_from_billing_parties(unresolved_parties)
        for booking_id in ids:
            if booking_id in candidates:
                continue
            party_id = billing_parties.get(booking_id)
            if party_id is None:
                continue
            found = patient_by_party.get(int(party_id))
            if found is not None:
                candidates[booking_id].add(found)

    resolved: dict[int, int] = {}
    ambiguous: set[int] = set()
    for booking_id, patient_ids in candidates.items():
        if len(patient_ids) == 1:
            resolved[booking_id] = next(iter(patient_ids))
        else:
            ambiguous.add(booking_id)

    return resolved, ambiguous


def resolve_missing_institution_patient_ids(
    bookings: list[Any],
    *,
    persist: bool = True,
) -> int:
    """Complète ``institution_patient_id`` sur les bookings fournis.

    Les objets en mémoire sont toujours mis à jour ; ``persist`` déclenche en plus
    l'écriture en base (rattrapage idempotent du backfill). Retourne le nombre de
    bookings résolus.
    """
    targets = [
        b for b in bookings if getattr(b, "institution_patient_id", None) is None
    ]
    if not targets:
        return 0

    parents = {
        int(b.id): int(b.parent_booking_id)
        for b in targets
        if getattr(b, "parent_booking_id", None) is not None
    }
    groups = {
        int(b.id): str(b.route_group_id)
        for b in targets
        if getattr(b, "route_group_id", None)
    }
    billing_parties = {
        int(b.id): int(b.billing_party_id)
        for b in targets
        if getattr(b, "billing_party_id", None) is not None
    }

    resolved, ambiguous = build_institution_patient_mapping(
        [int(b.id) for b in targets],
        parent_ids_by_booking=parents,
        route_group_by_booking=groups,
        billing_party_by_booking=billing_parties,
    )
    if ambiguous:
        logger.warning(
            "institution_patient_resolution: %s booking(s) ambigus non résolus: %s",
            len(ambiguous),
            sorted(ambiguous)[:20],
        )
    if not resolved:
        return 0

    for booking in targets:
        patient_id = resolved.get(int(booking.id))
        if patient_id is not None:
            booking.institution_patient_id = patient_id

    if persist:
        from ext import db

        try:
            db.session.commit()
        except Exception:  # pragma: no cover - rattrapage best-effort
            logger.exception(
                "institution_patient_resolution: échec de persistance, "
                "poursuite en mémoire"
            )
            db.session.rollback()
            for booking in targets:
                patient_id = resolved.get(int(booking.id))
                if patient_id is not None:
                    booking.institution_patient_id = patient_id

    logger.info(
        "institution_patient_resolution: %s booking(s) rattachés (persist=%s)",
        len(resolved),
        persist,
    )
    return len(resolved)
