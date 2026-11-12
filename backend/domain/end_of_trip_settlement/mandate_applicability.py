"""Le mandat de démonstration couvre-t-il cette course ?

Aucune facture n'est autorisée. Une signature fictive ne devient pas
un mandat juridique.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class MandateCoverage:
    kind: str
    label: str
    billing_authorized: bool
    amendment_required: bool


@dataclass(frozen=True, slots=True)
class TechnicalApplicability:
    """Couverture technique. Ni valeur juridique, ni droit de facturer."""

    code: str
    label: str
    avenant_requis: bool
    motif: str | None
    billing_authorized: bool
    legally_effective: bool


def assess_mandate(
    *,
    mandate_company_id: int | None,
    booking_company_id: int,
    status: str,
    covered_tariff_versions: tuple[str, ...],
    booking_tariff_version: str | None,
    valid_from: date | None,
    valid_until: date | None,
    on_date: date,
    billed_to_type: str,
    already_paid: bool,
) -> MandateCoverage:
    """Décide la situation. N'écrit rien."""
    payer = (billed_to_type or "").strip().lower()
    if payer == "clinic" or already_paid:
        return MandateCoverage(
            kind="not_concerned",
            label="Situation non concernée par un mandat patient",
            billing_authorized=False,
            amendment_required=False,
        )
    if mandate_company_id is None or status in {"", "demo_absent"}:
        return MandateCoverage(
            kind="absent",
            label="Aucun mandat",
            billing_authorized=False,
            amendment_required=False,
        )
    if int(mandate_company_id) != int(booking_company_id):
        return MandateCoverage(
            kind="other_company",
            label="Mandat d'une autre entreprise",
            billing_authorized=False,
            amendment_required=False,
        )
    if valid_until is not None and on_date > valid_until:
        return MandateCoverage(
            kind="expired",
            label="Mandat expiré",
            billing_authorized=False,
            amendment_required=False,
        )
    if valid_from is not None and on_date < valid_from:
        return MandateCoverage(
            kind="prepared",
            label="Mandat à signer",
            billing_authorized=False,
            amendment_required=False,
        )
    if booking_tariff_version and booking_tariff_version not in covered_tariff_versions:
        return MandateCoverage(
            kind="amendment_required",
            label="Avenant requis — version tarifaire non couverte.",
            billing_authorized=False,
            amendment_required=True,
        )
    if status == "demo_draft":
        return MandateCoverage(
            kind="prepared",
            label="Mandat à signer",
            billing_authorized=False,
            amendment_required=False,
        )
    if status == "demo_specimen_signed":
        return MandateCoverage(
            kind="specimen",
            label="Mandat signé",
            billing_authorized=False,
            amendment_required=False,
        )
    if status == "demo_valid_simulated":
        return MandateCoverage(
            kind="applicable_demonstration",
            label="Mandat signé",
            billing_authorized=False,
            amendment_required=False,
        )
    return MandateCoverage(
        kind="prepared",
        label="Mandat à signer",
        billing_authorized=False,
        amendment_required=False,
    )


def decide_coverage(
    *,
    mandate_client_id: int | None,
    booking_client_id: int | None,
    mandate_company_id: int | None,
    booking_company_id: int,
    active: bool,
    covered_tariffs: tuple[str, ...],
    booking_tariff: str | None,
    covered_services: tuple[str, ...],
    booking_service: str | None,
) -> TechnicalApplicability:
    """Compare la version proposée au transport. N'écrit rien et n'émet rien."""
    if mandate_client_id is None or not active:
        return TechnicalApplicability(
            code="mandat_absent",
            label="Aucun mandat actif",
            avenant_requis=False,
            motif=None,
            billing_authorized=False,
            legally_effective=False,
        )
    if booking_client_id is None or int(mandate_client_id) != int(booking_client_id):
        return TechnicalApplicability(
            code="autre_patient",
            label="Mandat d'un autre patient",
            avenant_requis=False,
            motif=None,
            billing_authorized=False,
            legally_effective=False,
        )
    if mandate_company_id is None or int(mandate_company_id) != int(booking_company_id):
        return TechnicalApplicability(
            code="autre_entreprise",
            label="Mandat d'une autre entreprise",
            avenant_requis=False,
            motif=None,
            billing_authorized=False,
            legally_effective=False,
        )
    service = (booking_service or "").strip()
    if service and covered_services and service not in covered_services:
        return TechnicalApplicability(
            code="avenant_requis",
            label="Avenant requis — prestation non couverte.",
            avenant_requis=True,
            motif="prestation",
            billing_authorized=False,
            legally_effective=False,
        )
    tariff = (booking_tariff or "").strip()
    if tariff and tariff not in covered_tariffs:
        return TechnicalApplicability(
            code="avenant_requis",
            label="Avenant requis — tarif non couvert.",
            avenant_requis=True,
            motif="tarif",
            billing_authorized=False,
            legally_effective=False,
        )
    return TechnicalApplicability(
        code="applicable",
        label="Mandat applicable techniquement",
        avenant_requis=False,
        motif=None,
        billing_authorized=False,
        legally_effective=False,
    )


def real_invoice_blocked(
    *,
    kind: str,
    legally_effective: bool,
    real_signature_collected: bool,
) -> str:
    """Refuse toute facture réelle. Aucune preuve juridique n'est acceptée ici."""
    if legally_effective or real_signature_collected:
        return "preuve_juridique_absente"
    return {
        "absent": "convention_inexistante",
        "prepared": "convention_preparee",
        "specimen": "mandat_demonstration",
        "applicable_demonstration": "mandat_demonstration",
        "expired": "convention_expiree",
        "amendment_required": "avenant_non_signe",
        "other_company": "autre_entreprise",
        "not_concerned": "reservation_non_concernee",
    }.get(kind, "convention_incomplete")
