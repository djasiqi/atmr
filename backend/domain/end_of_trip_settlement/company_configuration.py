"""Modalités de règlement définies par l'entreprise, pas par le formulaire.

Le payeur reste celui de la réservation. Cette préparation ne persiste rien
et n'active aucun encaissement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

_PAYER = frozenset({"patient", "institution", "third_party"})


@dataclass(frozen=True, slots=True)
class CompanySettlementSettings:
    """Réglages entreprise. Faux : aucun moyen nouveau n'est proposé."""

    tap_to_pay_enabled: bool = False
    cash_enabled: bool = False
    direct_patient_invoice_enabled: bool = False
    tap_to_pay_preferred: bool = False
    cash_preferred: bool = False
    direct_patient_invoice_preferred: bool = False
    periodic_accounts_enabled: bool = False
    config_version: str = "inactive"
    invoice_signature_approved: bool = False


@dataclass(frozen=True, slots=True)
class PatientNotice:
    """Déclaration d'un collaborateur. Ce n'est pas une preuve de réception."""

    confirmed: bool
    author_user_id: int | None
    confirmed_at: datetime | None
    conditions_version: str | None
    amount_chf: Decimal | None
    debtor_label: str | None
    channel: str | None
    auto_checked: bool


@dataclass(frozen=True, slots=True)
class AcceptanceSnapshot:
    """Décision préparée à l'attribution. Aucun paiement n'est lancé."""

    record: bool
    modality: str | None
    modes: frozenset[str]
    reason: str
    administrative_review: bool
    collection_allowed: bool
    financial_lock_required: bool
    lock_applied: bool
    config_version: str | None
    patient_notice_required: bool
    institution_modes_ignored: bool


def settings_for_acceptance(
    *,
    preferences: CompanySettlementSettings,
    authorized: CompanySettlementSettings,
) -> CompanySettlementSettings:
    """Une préférence enregistrée ne rend pas le moyen disponible."""
    return CompanySettlementSettings(
        tap_to_pay_enabled=authorized.tap_to_pay_enabled,
        cash_enabled=authorized.cash_enabled,
        direct_patient_invoice_enabled=authorized.direct_patient_invoice_enabled,
        tap_to_pay_preferred=preferences.tap_to_pay_preferred,
        cash_preferred=preferences.cash_preferred,
        direct_patient_invoice_preferred=preferences.direct_patient_invoice_preferred,
        periodic_accounts_enabled=False,
        config_version=preferences.config_version or authorized.config_version,
        invoice_signature_approved=authorized.invoice_signature_approved,
    )


def acceptance_write_plan(decision: AcceptanceSnapshot) -> str:
    """Aucune politique active ni aucun verrou pendant cette étape."""
    if decision.collection_allowed or decision.lock_applied or decision.record:
        return "refused"
    return "no_policy_row"


def declare_patient_notice(
    *,
    confirmed: bool,
    auto_checked: bool,
    author_user_id: int | None,
    confirmed_at: datetime | None,
    conditions_version: str | None,
    amount_chf: Decimal | None,
    debtor_label: str | None,
    channel: str | None,
) -> PatientNotice:
    """Une case précochée ou incomplète ne confirme rien."""
    complete = (
        confirmed
        and not auto_checked
        and author_user_id is not None
        and author_user_id > 0
        and confirmed_at is not None
        and confirmed_at.tzinfo is not None
        and bool((conditions_version or "").strip())
        and amount_chf is not None
        and bool((debtor_label or "").strip())
    )
    return PatientNotice(
        confirmed=complete,
        author_user_id=author_user_id,
        confirmed_at=confirmed_at,
        conditions_version=conditions_version,
        amount_chf=amount_chf,
        debtor_label=debtor_label,
        channel=channel,
        auto_checked=auto_checked,
    )


def prepare_acceptance_snapshot(
    *,
    company_assigned: bool,
    payer: str,
    created_via: str,
    financial_responsibility_locked: bool,
    client_has_established_account: bool,
    settings: CompanySettlementSettings,
    already_paid: bool,
    invoice_issued: bool,
    is_round_trip: bool,
    round_trip_collection_approved: bool,
    return_time_defined: bool | None,
    patient_notice: PatientNotice | None,
    announced_amount_chf: Decimal | None = None,
    institution_requested_modes: frozenset[str] = frozenset(),
    driver_requested_modes: frozenset[str] = frozenset(),
) -> AcceptanceSnapshot:
    """Calcule les moyens compatibles. N'écrit ni payeur ni politique."""
    ignored = bool(institution_requested_modes or driver_requested_modes)
    if already_paid or invoice_issued:
        return _snap(
            record=False,
            modality="prepaid" if already_paid else None,
            modes=frozenset(),
            reason="already_settled" if already_paid else "invoice_already_issued",
            review=False,
            allowed=False,
            lock_required=False,
            version=None,
            notice_required=False,
            ignored=ignored,
        )
    if not company_assigned:
        return _snap(
            record=False,
            modality=None,
            modes=frozenset(),
            reason="company_not_assigned",
            review=True,
            allowed=False,
            lock_required=False,
            version=None,
            notice_required=False,
            ignored=ignored,
        )
    if payer not in _PAYER:
        return _blocked("payer_not_explicit", ignored, settings, lock_required=False)
    if payer == "institution":
        return _snap(
            record=False,
            modality="institution_account",
            modes=frozenset(),
            reason="institution_payer",
            review=False,
            allowed=False,
            lock_required=False,
            version=settings.config_version,
            notice_required=False,
            ignored=ignored,
        )
    if payer == "third_party":
        return _blocked("third_party_convention", ignored, settings, lock_required=False)
    if client_has_established_account:
        return _snap(
            record=False,
            modality="established_account",
            modes=frozenset(),
            reason="established_account",
            review=False,
            allowed=False,
            lock_required=False,
            version=settings.config_version,
            notice_required=False,
            ignored=ignored,
        )
    lock_required = created_via == "institution_portal"
    if lock_required and not financial_responsibility_locked:
        return _blocked("payer_still_mutable", ignored, settings, lock_required=True)
    if is_round_trip and return_time_defined is False:
        return _blocked(
            "return_time_missing",
            ignored,
            settings,
            lock_required=lock_required,
        )
    if is_round_trip and not round_trip_collection_approved:
        return _blocked(
            "round_trip_collection_unapproved",
            ignored,
            settings,
            lock_required=lock_required,
        )
    modes = _company_modes(settings)
    if not modes:
        return _blocked(
            "company_has_no_enabled_mode",
            ignored,
            settings,
            lock_required=lock_required,
        )
    notice_ok = patient_notice is not None and patient_notice.confirmed
    if (
        notice_ok
        and announced_amount_chf is not None
        and patient_notice is not None
        and patient_notice.amount_chf != announced_amount_chf
    ):
        return _snap(
            record=False,
            modality=None,
            modes=frozenset(),
            reason="patient_notice_stale",
            review=True,
            allowed=False,
            lock_required=lock_required,
            version=settings.config_version,
            notice_required=True,
            ignored=ignored,
        )
    if not notice_ok:
        return _snap(
            record=False,
            modality=None,
            modes=frozenset(),
            reason="patient_notice_missing",
            review=True,
            allowed=False,
            lock_required=lock_required,
            version=settings.config_version,
            notice_required=True,
            ignored=ignored,
        )
    if modes == frozenset({"invoice"}):
        modality = "patient_invoice_after"
    elif "invoice" not in modes:
        modality = "on_site"
    else:
        modality = None
    return _snap(
        record=False,
        modality=modality,
        modes=modes,
        reason="compatible_modes_prepared",
        review=False,
        allowed=False,
        lock_required=lock_required,
        version=settings.config_version,
        notice_required=True,
        ignored=ignored,
    )


def capability_change_conflict(
    confirmed_modes: frozenset[str],
    settings: CompanySettlementSettings,
) -> str | None:
    """Une désactivation ultérieure ne remplace pas un moyen déjà confirmé."""
    current = _company_modes(settings)
    missing = confirmed_modes - current
    if not missing:
        return None
    return "capability_withdrawn_needs_review"


def _company_modes(settings: CompanySettlementSettings) -> frozenset[str]:
    modes: set[str] = set()
    if settings.tap_to_pay_enabled:
        modes.add("tap_to_pay")
    if settings.cash_enabled:
        modes.add("cash")
    if settings.direct_patient_invoice_enabled and settings.invoice_signature_approved:
        modes.add("invoice")
    return frozenset(modes)


def _blocked(
    reason: str,
    ignored: bool,
    settings: CompanySettlementSettings,
    *,
    lock_required: bool,
) -> AcceptanceSnapshot:
    return _snap(
        record=False,
        modality=None,
        modes=frozenset(),
        reason=reason,
        review=True,
        allowed=False,
        lock_required=lock_required,
        version=settings.config_version,
        notice_required=False,
        ignored=ignored,
    )


def _snap(
    *,
    record: bool,
    modality: str | None,
    modes: frozenset[str],
    reason: str,
    review: bool,
    allowed: bool,
    lock_required: bool,
    version: str | None,
    notice_required: bool,
    ignored: bool,
) -> AcceptanceSnapshot:
    return AcceptanceSnapshot(
        record=record,
        modality=modality,
        modes=modes,
        reason=reason,
        administrative_review=review,
        collection_allowed=allowed,
        financial_lock_required=lock_required,
        lock_applied=False,
        config_version=version,
        patient_notice_required=notice_required,
        institution_modes_ignored=ignored,
    )
