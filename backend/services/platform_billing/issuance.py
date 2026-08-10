"""Émission de factures légales plateforme (PDF + QR 0.01)."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ext import db
from models import Company
from models.billing_profile import CompanyBillingProfile
from models.enums import (
    PlatformBillingPeriodStatus,
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
    PlatformStatementStatus,
)
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformBillingCreditor,
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformIssuedInvoice,
    PlatformSupportEntry,
)
from services.platform_billing.errors import BillingInvariantError
from services.platform_billing.invoice_pdf import (
    build_platform_qrr_reference,
    build_platform_scor_reference,
    generate_platform_invoice_pdf_bytes,
    publish_platform_invoice_pdf,
)
from services.platform_billing.money import money_round_chf
from services.platform_billing.readiness import (
    validate_platform_invoice_creditor,
    validate_platform_invoice_debtor,
)
from services.platform_billing.swiss_qr import (
    QrParty,
    SwissQrBillPayload,
    platform_qr_amount,
    render_swiss_qr_bill,
    resolve_platform_reference_mode,
)
from services.platform_billing.time_bounds import (
    billing_period_has_ended,
    next_month_start_zurich_utc,
)

logger = logging.getLogger(__name__)


def _next_invoice_number(year: int, month: int) -> str:
    """Numérotation atomique LIRIE-YYYY-MM-NNNN via séquence dédiée."""
    result = db.session.execute(
        text(
            """
            INSERT INTO platform_invoice_number_sequence
                (billing_year, billing_month, next_value)
            VALUES (:year, :month, 1)
            ON CONFLICT (billing_year, billing_month)
            DO UPDATE SET
                next_value = platform_invoice_number_sequence.next_value + 1,
                updated_at = NOW()
            RETURNING next_value
            """
        ),
        {"year": int(year), "month": int(month)},
    )
    seq = int(result.scalar_one())
    return f"LIRIE-{year:04d}-{month:02d}-{seq:04d}"


def _debtor_party(
    company: Company, profile: CompanyBillingProfile | None
) -> tuple[QrParty, dict[str, Any]]:
    if profile is not None:
        party = QrParty(
            name=profile.legal_name or company.name or "",
            street=profile.street_name or "",
            building_number=profile.building_number,
            postal_code=profile.postal_code or "",
            city=profile.city or "",
            country_code=profile.country_code or "CH",
        )
        snap = {
            "legal_name": profile.legal_name,
            "street_name": profile.street_name,
            "building_number": profile.building_number,
            "postal_code": profile.postal_code,
            "city": profile.city,
            "country_code": profile.country_code,
            "billing_email": profile.billing_email,
            "uid_ide": profile.uid_ide,
            "vat_number": profile.vat_number,
        }
        return party, snap
    party = QrParty(
        name=company.name or "",
        street=getattr(company, "domicile_address_line1", None) or "",
        building_number=getattr(company, "domicile_address_line2", None),
        postal_code=getattr(company, "domicile_zip", None) or "",
        city=getattr(company, "domicile_city", None) or "",
        country_code=getattr(company, "domicile_country", None) or "CH",
    )
    snap = {
        "legal_name": company.name,
        "street_name": party.street,
        "building_number": party.building_number,
        "postal_code": party.postal_code,
        "city": party.city,
        "country_code": party.country_code,
        "billing_email": getattr(company, "billing_email", None),
        "uid_ide": getattr(company, "uid_ide", None),
        "vat_number": None,
    }
    return party, snap


def _creditor_party(
    creditor: PlatformBillingCreditor,
) -> tuple[QrParty, dict[str, Any], str]:
    party = QrParty(
        name=creditor.legal_name,
        street=creditor.street_name,
        building_number=creditor.building_number,
        postal_code=creditor.postal_code,
        city=creditor.city,
        country_code=creditor.country_code or "CH",
    )
    snap = {
        "legal_name": creditor.legal_name,
        "street_name": creditor.street_name,
        "building_number": creditor.building_number,
        "postal_code": creditor.postal_code,
        "city": creditor.city,
        "country_code": creditor.country_code,
        "uid_ide": creditor.uid_ide,
        "vat_number": creditor.vat_number,
        "iban": creditor.iban,
        "qr_iban": creditor.qr_iban,
        "payment_reference_mode": creditor.payment_reference_mode,
    }
    iban = (creditor.qr_iban or creditor.iban or "").replace(" ", "")
    return party, snap, iban


def statement_issuance_ready(statement: PlatformInvoice) -> tuple[bool, list[str]]:
    """Diagnostic d'émission (raisons textuelles ; n'élève pas d'exception)."""
    errors: list[str] = []
    status = statement.statement_status or PlatformStatementStatus.DRAFT.value
    if status != PlatformStatementStatus.LOCKED.value:
        errors.append(
            "Le relevé doit être verrouillé avant l’émission "
            "(clôturez d’abord la période)."
        )
    period = statement.period
    if period is None:
        errors.append("Période introuvable pour ce relevé")
    else:
        if period.status != PlatformBillingPeriodStatus.LOCKED.value:
            errors.append("La période doit être verrouillée avant l’émission.")
        year, month = int(period.billing_year), int(period.billing_month)
        if not billing_period_has_ended(year, month):
            ends_at = next_month_start_zurich_utc(year, month)
            errors.append(
                f"La période {month:02d}/{year} n’est pas terminée "
                f"(fin : {ends_at.isoformat()})."
            )
    company = db.session.get(Company, statement.company_id)
    profile = CompanyBillingProfile.query.filter_by(
        company_id=statement.company_id
    ).first()
    debtor_ok, debtor_errors = validate_platform_invoice_debtor(profile, company)
    if not debtor_ok:
        errors.extend(debtor_errors)
    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    cred = validate_platform_invoice_creditor(creditor)
    if not cred["creditor_profile_ready"]:
        errors.extend(cred["creditor_errors"])
    if statement.total_amount is None:
        errors.append("Totaux non figés")
    return len(errors) == 0, errors


def statement_qr_ready(statement: PlatformInvoice) -> tuple[bool, list[str]]:
    ok, errors = statement_issuance_ready(statement)
    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    cred = validate_platform_invoice_creditor(creditor)
    if not cred["creditor_qr_ready"]:
        errors.extend([e for e in cred["creditor_errors"] if e not in errors])
        ok = False
    if statement.currency != "CHF":
        errors.append("Devise QR doit être CHF")
        ok = False
    if Decimal(str(statement.total_amount or 0)) <= 0:
        errors.append("Montant total doit être > 0 pour QR")
        ok = False
    company = db.session.get(Company, statement.company_id)
    profile = CompanyBillingProfile.query.filter_by(
        company_id=statement.company_id
    ).first()
    debtor_ok, debtor_errors = validate_platform_invoice_debtor(profile, company)
    if not debtor_ok:
        ok = False
        for e in debtor_errors:
            if e not in errors:
                errors.append(e)
    return ok, errors


def _support_qty_unit_from_snapshot(snap: dict[str, Any]) -> tuple[Any, Any]:
    """Durée / tarif depuis le snapshot, ou hydratation via entry_ids (anciens relevés)."""
    hours_raw = snap.get("duration_hours")
    minutes = snap.get("duration_minutes")
    rate_raw = snap.get("hourly_rate")
    if hours_raw is None and minutes is not None:
        hours_raw = Decimal(str(minutes)) / Decimal(60)
    if hours_raw is not None and rate_raw is not None:
        return hours_raw, rate_raw

    entry_ids = snap.get("entry_ids") or []
    if not entry_ids:
        return hours_raw, rate_raw
    try:
        ids = [int(x) for x in entry_ids]
    except (TypeError, ValueError):
        return hours_raw, rate_raw
    rows = PlatformSupportEntry.query.filter(PlatformSupportEntry.id.in_(ids)).all()
    if not rows:
        return hours_raw, rate_raw
    total_minutes = sum(int(r.duration_minutes or 0) for r in rows)
    hours = (Decimal(total_minutes) / Decimal(60)).quantize(Decimal("0.01"))
    rates = {
        str(r.hourly_rate_snapshot) for r in rows if r.hourly_rate_snapshot is not None
    }
    rate = rows[0].hourly_rate_snapshot if len(rates) == 1 else None
    return hours, rate


def resolve_line_qty_unit(ln: Any) -> dict[str, Any]:
    """
    Quantité / prix unitaire / taux % pour affichage et PDF.
    Complète les colonnes absentes via snapshot (et entrées support si besoin).
    """
    snap = ln.snapshot_json or {}
    line_type = (ln.line_type or "").lower()
    qty: Any = ln.quantity
    unit: Any = ln.unit_amount
    rate_percent: Decimal | None = None

    if "commission" in line_type:
        if qty is None and snap.get("booking_count") is not None:
            qty = snap.get("booking_count")
        rate_raw = snap.get("commission_rate")
        if rate_raw is not None:
            try:
                rate_percent = (Decimal(str(rate_raw)) * Decimal("100")).quantize(
                    Decimal("0.0001")
                )
            except Exception:
                rate_percent = None
        if unit is None and qty is not None and ln.amount is not None:
            try:
                q = Decimal(str(qty))
                if q > 0:
                    unit = money_round_chf(Decimal(str(ln.amount)) / q)
            except Exception:
                pass
    elif "support" in line_type:
        hours_raw, rate_raw = _support_qty_unit_from_snapshot(snap)
        if qty is None:
            qty = hours_raw
        if unit is None:
            unit = rate_raw
    elif "subscription" in line_type:
        if qty is None and snap.get("volume_count") is not None:
            qty = snap.get("volume_count")
        # Abonnement = forfait mensuel : P.U. = montant de la ligne
        if unit is None and ln.amount is not None:
            unit = ln.amount
    else:
        if qty is None and snap.get("booking_count") is not None:
            qty = snap.get("booking_count")
        if qty is None and snap.get("volume_count") is not None:
            qty = snap.get("volume_count")
        if qty is None and snap.get("duration_hours") is not None:
            qty = snap.get("duration_hours")

    return {
        "quantity": qty,
        "unit_amount": unit,
        "unit_rate_percent": rate_percent,
    }


def _enrich_line_label_for_pdf(ln: Any) -> str:
    """Complète le libellé (commission % / nb, support heures)."""
    label = (ln.label or "").strip() or str(ln.line_type or "Prestation")
    snap = ln.snapshot_json or {}
    line_type = (ln.line_type or "").lower()
    resolved = resolve_line_qty_unit(ln)
    # Retirer le jargon technique (created_at) des libellés client
    label = re.sub(
        r"\s*\(\s*created_at\s*\)",
        "",
        label,
        flags=re.IGNORECASE,
    ).strip()
    if "subscription" in line_type or label.lower().startswith("abonnement"):
        label = re.sub(r"\s*,\s*", ", ", label)
        if "created_at" in label.lower():
            label = "Abonnement plateforme — volume sur courses créées, hors annulés"
        return label
    if "commission" in line_type or "commission" in label.lower():
        rate_raw = snap.get("commission_rate")
        # Retirer nb transports / anciens suffixes — le nb va en colonne Qté
        base = re.sub(
            r"\s*[—\-]\s*\d+\s+transports?\s*$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()
        base = re.sub(
            r"\s*\(\s*[\d.,]+\s*%\s*\)\s*$",
            "",
            base,
            flags=re.IGNORECASE,
        ).strip()
        base = re.sub(
            r"\s*\(taux\s+[\d.,]+\s*%\s*\)\s*$",
            "",
            base,
            flags=re.IGNORECASE,
        ).strip()
        if not base or base.lower().startswith("commission transports"):
            base = "Commission LIRIE sur transports marketplace acceptés"
        if rate_raw is not None:
            try:
                rate = Decimal(str(rate_raw))
                pct_s = f"{(rate * Decimal('100')):.4f}".rstrip("0").rstrip(".")
                return f"{base} (taux {pct_s} %)"
            except Exception:
                pass
        return base
    if "support" in line_type or label.lower().startswith("support"):
        hours_raw = resolved.get("quantity")
        rate_raw = resolved.get("unit_amount")
        if hours_raw is None:
            return label
        hours = Decimal(str(hours_raw)).quantize(Decimal("0.01"))
        hours_s = f"{hours:.2f}".rstrip("0").rstrip(".")
        if rate_raw is not None:
            rate_s = f"{Decimal(str(rate_raw)):.2f}".rstrip("0").rstrip(".")
            return f"Support plateforme — {hours_s} h à {rate_s} CHF/h"
        return f"Support plateforme — {hours_s} h"
    return label


def _statement_line_dicts(statement: PlatformInvoice) -> list[dict[str, Any]]:
    lines = sorted(statement.lines, key=lambda x: (x.sort_order, x.id))
    result: list[dict[str, Any]] = []
    for ln in lines:
        resolved = resolve_line_qty_unit(ln)
        qty = resolved["quantity"]
        unit = resolved["unit_amount"]
        mode = "UNIT_PRICE" if qty is not None and unit is not None else "FIXED_AMOUNT"
        result.append(
            {
                "line_type": ln.line_type,
                "label": _enrich_line_label_for_pdf(ln),
                "amount": str(money_round_chf(Decimal(str(ln.amount)))),
                "quantity": str(qty) if qty is not None else None,
                "unit_amount": str(unit) if unit is not None else None,
                "calculation_mode": mode,
            }
        )
    return result


def _lines_for_pdf(
    inv: PlatformIssuedInvoice, statement: PlatformInvoice | None
) -> list[dict[str, Any]]:
    """Préfère lines_snapshot ; fallback relevé (legacy)."""
    from services.platform_billing.invoice_replace import sync_derived_line_label

    snap = inv.lines_snapshot
    if isinstance(snap, list) and snap:
        out: list[dict[str, Any]] = []
        for raw in snap:
            if not isinstance(raw, dict):
                continue
            amount = raw.get("amount")
            qty = (
                Decimal(str(raw["quantity"]))
                if raw.get("quantity") is not None
                else None
            )
            unit = (
                Decimal(str(raw["unit_amount"]))
                if raw.get("unit_amount") is not None
                else None
            )
            line_type = raw.get("line_type") or "ADJUSTMENT"
            label = sync_derived_line_label(
                label=str(raw.get("label") or line_type or "Ligne"),
                line_type=str(line_type),
                quantity=qty,
                unit_amount=unit,
            )
            out.append(
                {
                    "line_type": line_type,
                    "label": label,
                    "amount": Decimal(str(amount or 0)),
                    "quantity": qty,
                    "unit_amount": unit,
                }
            )
        return out
    if statement is not None:
        return [
            {
                **ln,
                "label": sync_derived_line_label(
                    label=str(ln.get("label") or ln.get("line_type") or "Ligne"),
                    line_type=str(ln.get("line_type") or "ADJUSTMENT"),
                    quantity=(
                        Decimal(str(ln["quantity"]))
                        if ln.get("quantity") is not None
                        else None
                    ),
                    unit_amount=(
                        Decimal(str(ln["unit_amount"]))
                        if ln.get("unit_amount") is not None
                        else None
                    ),
                ),
                "amount": Decimal(str(ln["amount"])),
                "quantity": (
                    Decimal(str(ln["quantity"]))
                    if ln.get("quantity") is not None
                    else None
                ),
                "unit_amount": (
                    Decimal(str(ln["unit_amount"]))
                    if ln.get("unit_amount") is not None
                    else None
                ),
            }
            for ln in _statement_line_dicts(statement)
        ]
    return []


def _build_and_store_pdf(
    *,
    inv: PlatformIssuedInvoice,
    statement: PlatformInvoice | None,
    creditor: PlatformBillingCreditor,
    debtor_snap: dict[str, Any],
    creditor_snap: dict[str, Any],
    iban: str,
    payment_terms_days: int,
) -> None:
    period = statement.period if statement is not None else None
    year = period.billing_year if period else int(inv.billing_year or 0)
    month = period.billing_month if period else int(inv.billing_month or 0)
    pdf_bytes = generate_platform_invoice_pdf_bytes(
        invoice_number=inv.invoice_number,
        issued_at=inv.issued_at,
        due_at=inv.due_at,
        period_year=year,
        period_month=month,
        creditor_snap=creditor_snap,
        debtor_snap=debtor_snap,
        lines=_lines_for_pdf(inv, statement),
        subtotal=money_round_chf(Decimal(str(inv.subtotal_amount))),
        tax_rate=Decimal(str(inv.tax_rate)),
        tax_amount=money_round_chf(Decimal(str(inv.tax_amount))),
        total=money_round_chf(Decimal(str(inv.total_amount))),
        qr_amount=money_round_chf(Decimal(str(inv.qr_amount))),
        qr_reference=inv.qr_reference,
        payment_reference_mode=creditor.payment_reference_mode or "QRR",
        iban=iban,
        payment_terms_days=payment_terms_days,
    )
    pdf_key, checksum = publish_platform_invoice_pdf(
        inv.invoice_number,
        pdf_bytes,
        previous_path=inv.pdf_storage_key,
    )
    inv.pdf_storage_key = pdf_key
    inv.pdf_checksum = checksum


def _active_issued_for_statement(statement_id: int) -> PlatformIssuedInvoice | None:
    inactive = {
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    }
    rows = (
        PlatformIssuedInvoice.query.filter_by(statement_id=int(statement_id))
        .filter(
            PlatformIssuedInvoice.document_type
            == PlatformIssuedDocumentType.INVOICE.value
        )
        .order_by(PlatformIssuedInvoice.id.desc())
        .all()
    )
    for row in rows:
        if row.status not in inactive:
            return row
    return None


def _cancelled_issued_for_replace(statement_id: int) -> PlatformIssuedInvoice | None:
    """Dernière facture annulée du relevé (pour replaces_*), sans détacher statement_id."""
    return (
        PlatformIssuedInvoice.query.filter_by(statement_id=int(statement_id))
        .filter(
            PlatformIssuedInvoice.document_type
            == PlatformIssuedDocumentType.INVOICE.value,
            PlatformIssuedInvoice.status == PlatformIssuedInvoiceStatus.CANCELLED.value,
        )
        .order_by(PlatformIssuedInvoice.id.desc())
        .first()
    )


def issue_platform_invoice(
    statement_id: int,
    *,
    now_utc: datetime | None = None,
) -> PlatformIssuedInvoice:
    from services.platform_billing.engine import assert_billing_period_has_ended

    statement = db.session.get(PlatformInvoice, statement_id)
    if not statement:
        raise BillingInvariantError(
            "STATEMENT_NOT_FOUND",
            "Relevé introuvable",
            status_code=404,
            details={"statement_id": statement_id},
        )
    existing = _active_issued_for_statement(statement_id)
    if existing and existing.status not in (PlatformIssuedInvoiceStatus.DRAFT.value,):
        raise BillingInvariantError(
            "INVOICE_ALREADY_ISSUED",
            "Facture déjà émise pour ce relevé",
            details={"statement_id": statement_id, "issued_id": existing.id},
        )

    cancelled = _cancelled_issued_for_replace(statement_id)
    replaces_id = cancelled.id if cancelled is not None else None
    # Réutiliser un brouillon éventuel ; sinon créer une nouvelle ligne (1:N)
    draft = (
        existing
        if existing and existing.status == PlatformIssuedInvoiceStatus.DRAFT.value
        else None
    )
    existing = draft

    period = statement.period
    if period is None:
        raise BillingInvariantError(
            "PERIOD_NOT_FOUND",
            "Période introuvable pour ce relevé",
            status_code=404,
            details={"statement_id": statement_id},
        )
    assert_billing_period_has_ended(
        int(period.billing_year),
        int(period.billing_month),
        now_utc=now_utc,
    )
    if period.status != PlatformBillingPeriodStatus.LOCKED.value:
        raise BillingInvariantError(
            "PERIOD_NOT_LOCKED",
            "La période doit être verrouillée avant l’émission.",
            details={
                "period_id": period.id,
                "period_status": period.status,
            },
        )
    status = statement.statement_status or PlatformStatementStatus.DRAFT.value
    if status != PlatformStatementStatus.LOCKED.value:
        raise BillingInvariantError(
            "STATEMENT_NOT_LOCKED",
            "Le relevé doit être verrouillé avant l’émission.",
            details={
                "statement_id": statement_id,
                "statement_status": status,
            },
        )

    qr_ok, qr_errors = statement_qr_ready(statement)
    if not qr_ok:
        raise BillingInvariantError(
            "STATEMENT_QR_NOT_READY",
            "Émission QR impossible: " + "; ".join(qr_errors),
            details={"errors": qr_errors},
        )

    company = db.session.get(Company, statement.company_id)
    if not company:
        raise ValueError("Entreprise introuvable")
    profile = CompanyBillingProfile.query.filter_by(company_id=company.id).first()
    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    if not creditor:
        raise ValueError("Créancier LIRIE manquant")

    year, month = period.billing_year, period.billing_month
    invoice_number = _next_invoice_number(year, month)

    debtor_party, debtor_snap = _debtor_party(company, profile)
    creditor_party, creditor_snap, iban = _creditor_party(creditor)

    subtotal = money_round_chf(Decimal(str(statement.subtotal_amount)))
    tax_rate = Decimal(str(statement.tax_rate or creditor.default_tax_rate))
    tax_amount = money_round_chf(
        Decimal(str(statement.tax_amount or 0))
        if statement.tax_amount is not None
        else subtotal * tax_rate / Decimal("100")
    )
    total = money_round_chf(Decimal(str(statement.total_amount)))
    qr_amount = platform_qr_amount(total)

    cfg = None
    if statement.contract_id:
        cfg = db.session.get(CompanyPlatformBillingConfig, statement.contract_id)
    days = (
        (cfg.payment_terms_days if cfg and cfg.payment_terms_days else None)
        or creditor.payment_terms_days_default
        or 30
    )
    now = datetime.now(UTC)
    due = now + timedelta(days=int(days))

    from models.enums import PartnerAgreementStatus
    from models.platform_billing import PlatformPartnerAgreement
    from services.platform_billing.dunning_policy import (
        build_dunning_policy_snapshot,
        is_dunning_authorized_at_issuance,
    )

    agreement = None
    if cfg is not None:
        agreement = (
            PlatformPartnerAgreement.query.filter_by(
                billing_config_id=cfg.id,
                status=PartnerAgreementStatus.SIGNED.value,
            )
            .order_by(PlatformPartnerAgreement.id.desc())
            .first()
        )
    dunning_snap = build_dunning_policy_snapshot(cfg) if cfg else None
    dunning_authorized = is_dunning_authorized_at_issuance(
        cfg=cfg, agreement=agreement, today=now.date()
    )

    # Colonnes NOT NULL : tout doit être renseigné avant le premier flush/INSERT.
    issued_fields = {
        "invoice_number": invoice_number,
        "document_type": PlatformIssuedDocumentType.INVOICE.value,
        "status": PlatformIssuedInvoiceStatus.ISSUED.value,
        "currency": "CHF",
        "subtotal_amount": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total_amount": total,
        "qr_amount": qr_amount,
        "issued_at": now,
        "due_at": due,
        "debtor_snapshot": debtor_snap,
        "creditor_snapshot": creditor_snap,
        "billing_config_id": cfg.id if cfg else None,
        "partner_agreement_id": agreement.id if agreement else None,
        "dunning_policy_snapshot": dunning_snap,
        "dunning_automation_authorized_at_issuance": dunning_authorized,
        "billing_year": int(year),
        "billing_month": int(month),
        "period_id": period.id,
        "replaces_issued_invoice_id": replaces_id,
        "lines_snapshot": _statement_line_dicts(statement),
    }

    if existing:
        inv = existing
        for key, value in issued_fields.items():
            setattr(inv, key, value)
        db.session.flush()
    else:
        inv = PlatformIssuedInvoice(
            statement_id=statement.id,
            company_id=company.id,
            **issued_fields,
        )
        db.session.add(inv)
        db.session.flush()  # besoin de inv.id pour QRR

    ref_mode = resolve_platform_reference_mode(
        iban, creditor.payment_reference_mode or "QRR"
    )
    if ref_mode == "QRR":
        qr_reference = build_platform_qrr_reference(
            invoice_number=invoice_number,
            issued_id=inv.id,
            creditor_reference_base=getattr(creditor, "creditor_reference_base", None)
            or "21",
        )
    elif ref_mode == "SCOR":
        qr_reference = build_platform_scor_reference(invoice_number=invoice_number)
    else:
        qr_reference = None
    inv.qr_reference = qr_reference

    # Validation QR avant stockage PDF
    payload = SwissQrBillPayload(
        creditor=creditor_party,
        debtor=debtor_party,
        iban=iban,
        reference_type=ref_mode,
        reference=qr_reference,
        amount=qr_amount,
        currency="CHF",
        additional_information=invoice_number,
    )
    try:
        render_swiss_qr_bill(payload)
    except Exception as e:
        db.session.rollback()
        raise ValueError(f"Émission QR impossible: {e}") from e

    _build_and_store_pdf(
        inv=inv,
        statement=statement,
        creditor=creditor,
        debtor_snap=debtor_snap,
        creditor_snap=creditor_snap,
        iban=iban,
        payment_terms_days=int(days),
    )
    db.session.commit()
    db.session.refresh(inv)
    return inv


def _heal_lines_snapshot_labels(inv: PlatformIssuedInvoice) -> None:
    """Resynchronise les libellés auto du snapshot (support h) avec qté/prix."""
    from services.platform_billing.invoice_replace import sync_derived_line_label

    snap = inv.lines_snapshot
    if not isinstance(snap, list) or not snap:
        return
    healed: list[dict[str, Any]] = []
    changed = False
    for raw in snap:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        qty = Decimal(str(row["quantity"])) if row.get("quantity") is not None else None
        unit = (
            Decimal(str(row["unit_amount"]))
            if row.get("unit_amount") is not None
            else None
        )
        line_type = str(row.get("line_type") or "ADJUSTMENT")
        synced = sync_derived_line_label(
            label=str(row.get("label") or line_type or "Ligne"),
            line_type=line_type,
            quantity=qty,
            unit_amount=unit,
        )
        if synced != row.get("label"):
            row["label"] = synced
            changed = True
        healed.append(row)
    if changed:
        inv.lines_snapshot = healed


def regenerate_issued_invoice_pdf(
    issued_id: int, *, commit: bool = True
) -> PlatformIssuedInvoice:
    """Régénère le PDF d'une facture déjà émise (vrai template + QR).

    Interdit après envoi (PDF immuable).
    """
    inv = db.session.get(PlatformIssuedInvoice, issued_id)
    if not inv:
        raise ValueError("Facture émise introuvable")
    if inv.sent_at is not None:
        raise ValueError(
            "PDF immuable après envoi — prolongation d'échéance sans régénération"
        )
    _heal_lines_snapshot_labels(inv)
    statement = (
        db.session.get(PlatformInvoice, inv.statement_id) if inv.statement_id else None
    )
    if not statement and not inv.lines_snapshot:
        raise ValueError("Relevé lié introuvable — régénération impossible")

    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    if not creditor:
        raise ValueError("Créancier LIRIE manquant")
    _, creditor_snap, iban = _creditor_party(creditor)
    debtor_snap = inv.debtor_snapshot or {}
    if not debtor_snap.get("street_name"):
        company = db.session.get(Company, inv.company_id)
        profile = CompanyBillingProfile.query.filter_by(
            company_id=inv.company_id
        ).first()
        if company:
            _, debtor_snap = _debtor_party(company, profile)

    days = creditor.payment_terms_days_default or 30
    ref_mode = resolve_platform_reference_mode(
        iban, creditor.payment_reference_mode or "QRR"
    )
    if ref_mode == "QRR":
        inv.qr_reference = build_platform_qrr_reference(
            invoice_number=inv.invoice_number,
            issued_id=inv.id,
            creditor_reference_base=getattr(creditor, "creditor_reference_base", None)
            or "21",
        )
    elif ref_mode == "SCOR":
        inv.qr_reference = build_platform_scor_reference(
            invoice_number=inv.invoice_number
        )
    else:
        inv.qr_reference = None

    _build_and_store_pdf(
        inv=inv,
        statement=statement,
        creditor=creditor,
        debtor_snap=debtor_snap,
        creditor_snap=creditor_snap or inv.creditor_snapshot or {},
        iban=iban,
        payment_terms_days=int(days),
    )
    if commit:
        db.session.commit()
        db.session.refresh(inv)
    else:
        db.session.flush()
    return inv


def read_issued_invoice_pdf(issued_id: int) -> tuple[bytes, str]:
    """Lit le PDF stocké ; régénère avant envoi si stub / manquant."""
    inv = db.session.get(PlatformIssuedInvoice, issued_id)
    if not inv:
        raise ValueError("Facture émise introuvable")
    key = (inv.pdf_storage_key or "").strip()
    path = Path(key) if key else None
    filename = f"{(inv.invoice_number or f'facture-{issued_id}').replace('/', '_')}.pdf"

    needs_regen = (
        path is None
        or not path.exists()
        or path.suffix.lower() != ".pdf"
        or path.stat().st_size < 15000  # stub / PDF sans page QR
        or inv.sent_at is None  # avant envoi : coller au snapshot (libellés sync)
    )
    if needs_regen:
        if inv.sent_at is not None:
            if path is None or not path.exists():
                raise ValueError(
                    "PDF introuvable — document immuable après envoi, "
                    "régénération interdite"
                )
        else:
            inv = regenerate_issued_invoice_pdf(issued_id)
            path = Path(inv.pdf_storage_key)
            filename = path.name

    assert path is not None
    return path.read_bytes(), filename


def issue_ready_for_period(period_id: int) -> dict[str, Any]:
    """Émet toutes les factures prêtes de la période (idempotent, partial success)."""
    from services.platform_billing.dossier_status import dossier_key

    period = db.session.get(PlatformBillingPeriod, period_id)
    if not period:
        raise ValueError("Période introuvable")

    issued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    statements = PlatformInvoice.query.filter_by(period_id=period_id).all()
    for statement in statements:
        key = dossier_key(period_id, statement.company_id)
        st = statement.statement_status or PlatformStatementStatus.DRAFT.value
        if st != PlatformStatementStatus.LOCKED.value:
            skipped.append(
                {
                    "dossier_key": key,
                    "code": "NOT_LOCKED",
                    "message": "Relevé non verrouillé",
                }
            )
            continue
        existing = _active_issued_for_statement(statement.id)
        if existing and existing.status not in (
            PlatformIssuedInvoiceStatus.DRAFT.value,
        ):
            skipped.append(
                {
                    "dossier_key": key,
                    "code": "ALREADY_ISSUED",
                    "message": "Facture déjà émise",
                    "issued_invoice_id": existing.id,
                    "invoice_number": existing.invoice_number,
                }
            )
            continue
        qr_ok, qr_errors = statement_qr_ready(statement)
        if not qr_ok:
            failed.append(
                {
                    "dossier_key": key,
                    "code": "NOT_ISSUABLE",
                    "message": "; ".join(qr_errors) or "Émission non prête",
                }
            )
            continue
        try:
            inv = issue_platform_invoice(statement.id)
            issued.append(
                {
                    "dossier_key": key,
                    "issued_invoice_id": inv.id,
                    "invoice_number": inv.invoice_number,
                }
            )
        except BillingInvariantError as e:
            failed.append(
                {
                    "dossier_key": key,
                    "code": e.code,
                    "message": e.message,
                }
            )
        except Exception as e:
            logger.exception("issue-ready failed for %s", key)
            failed.append(
                {
                    "dossier_key": key,
                    "code": "ISSUE_ERROR",
                    "message": str(e),
                }
            )

    return {
        "period_id": period_id,
        "issued": issued,
        "skipped": skipped,
        "failed": failed,
    }
