"""Lifecycle des accords juridiques partenaires (DOCX + PDF signé)."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select, text

from ext import db
from models.billing_profile import CompanyBillingProfile
from models.company import Company
from models.enums import PartnerAgreementStatus, SubscriptionPricingMode
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformBillingCreditor,
    PlatformPartnerAgreement,
)
from security.audit_log import AuditLogger
from services.platform_billing.decimal_json import decimal_to_str
from services.platform_billing.partner_agreement_docx import (
    GENERATOR_VERSION,
    TEMPLATE_VERSION,
    build_partner_agreement_docx_bytes,
    template_sha256,
)
from services.platform_billing.partner_identity import (
    resolve_operator_contract_identity,
    resolve_partner_contract_identity,
)
from shared.upload_path_resolver import get_uploads_base
from shared.upload_write import ensure_writable_dir

logger = logging.getLogger(__name__)
_ZURICH = ZoneInfo("Europe/Zurich")

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME = "application/pdf"
ACTIVE_STATUSES = (
    PartnerAgreementStatus.DRAFT.value,
    PartnerAgreementStatus.SENT.value,
    PartnerAgreementStatus.SIGNED.value,
)
FROZEN_STATUSES = (
    PartnerAgreementStatus.SENT.value,
    PartnerAgreementStatus.SIGNED.value,
)


class PartnerAgreementError(Exception):
    """Erreur métier accord partenaire (message FR)."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def write_upload_bytes_atomic(filepath: Path, data: bytes) -> None:
    """Écriture atomique (temp + os.replace) sous le dossier cible."""
    ensure_writable_dir(filepath.parent)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_", suffix=filepath.suffix, dir=str(filepath.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _best_effort_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("Suppression fichier orphelin impossible: %s (%s)", path, exc)


def config_is_commercially_frozen(billing_config_id: int) -> bool:
    row = (
        db.session.execute(
            select(PlatformPartnerAgreement.id)
            .where(
                PlatformPartnerAgreement.billing_config_id == billing_config_id,
                PlatformPartnerAgreement.status.in_(FROZEN_STATUSES),
            )
            .limit(1)
        )
        .scalars()
        .first()
    )
    return row is not None


def assert_config_mutable(billing_config_id: int) -> None:
    if config_is_commercially_frozen(billing_config_id):
        raise PartnerAgreementError(
            "Configuration commerciale gelée : un accord est envoyé ou signé. "
            "Créez une nouvelle version de contrat commercial.",
            status_code=409,
        )


def _allocate_reference() -> str:
    now = datetime.now(_ZURICH)
    year_month = f"{now.year:04d}-{now.month:02d}"
    result = db.session.execute(
        text(
            """
            INSERT INTO platform_partner_agreement_sequence (year_month, last_value)
            VALUES (:year_month, 1)
            ON CONFLICT (year_month)
            DO UPDATE SET
                last_value = platform_partner_agreement_sequence.last_value + 1,
                updated_at = NOW()
            RETURNING last_value
            """
        ),
        {"year_month": year_month},
    )
    last_value = int(result.scalar_one())
    return f"LIRIE/PART/{year_month}/{last_value:03d}"


def _lock_active_agreement(
    billing_config_id: int,
) -> PlatformPartnerAgreement | None:
    return (
        db.session.execute(
            select(PlatformPartnerAgreement)
            .where(
                PlatformPartnerAgreement.billing_config_id == billing_config_id,
                PlatformPartnerAgreement.status.in_(ACTIVE_STATUSES),
            )
            .with_for_update()
        )
        .scalars()
        .first()
    )


def _lock_agreement(agreement_id: int) -> PlatformPartnerAgreement:
    agr = (
        db.session.execute(
            select(PlatformPartnerAgreement)
            .where(PlatformPartnerAgreement.id == agreement_id)
            .with_for_update()
        )
        .scalars()
        .first()
    )
    if not agr:
        raise PartnerAgreementError("Accord introuvable", status_code=404)
    return agr


def _next_revision_number(billing_config_id: int) -> int:
    current = (
        db.session.execute(
            select(PlatformPartnerAgreement.revision_number)
            .where(
                PlatformPartnerAgreement.billing_config_id == billing_config_id
            )
            .order_by(PlatformPartnerAgreement.revision_number.desc())
            .limit(1)
            .with_for_update()
        )
        .scalars()
        .first()
    )
    return int(current or 0) + 1


def _storage_dir(company_id: int, agreement_id: int) -> Path:
    return (
        get_uploads_base()
        / "platform_agreements"
        / str(company_id)
        / str(agreement_id)
    )


def _storage_key(company_id: int, agreement_id: int, filename: str) -> str:
    return f"platform_agreements/{company_id}/{agreement_id}/{filename}"


def _build_commercial_snapshot(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    return {
        "billing_config_id": cfg.id,
        "is_billing_enabled": bool(cfg.is_billing_enabled),
        "own_portfolio_billing_enabled": bool(cfg.own_portfolio_billing_enabled),
        "lirie_commission_enabled": bool(cfg.lirie_commission_enabled),
        "support_enabled": bool(cfg.support_enabled),
        "subscription_pricing_mode": cfg.subscription_pricing_mode,
        "custom_subscription_amount": decimal_to_str(cfg.custom_subscription_amount),
        "commission_rate": decimal_to_str(cfg.commission_rate, places=6),
        "commission_cancellation_policy": cfg.commission_cancellation_policy,
        "free_license_max_months": cfg.free_license_max_months,
        "statement_dispute_days": cfg.statement_dispute_days
        if cfg.statement_dispute_days is not None
        else 10,
        "payment_terms_days": cfg.payment_terms_days,
        "support_hourly_rate_default": decimal_to_str(
            cfg.support_hourly_rate_default
        ),
        "effective_from": cfg.effective_from.isoformat()
        if cfg.effective_from
        else None,
        "effective_to": cfg.effective_to.isoformat() if cfg.effective_to else None,
    }


def _effective_from_date(cfg: CompanyPlatformBillingConfig) -> date:
    if cfg.effective_from is None:
        return datetime.now(_ZURICH).date()
    dt = cfg.effective_from
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_ZURICH).date()


def serialize_agreement(agr: PlatformPartnerAgreement) -> dict[str, Any]:
    return {
        "id": agr.id,
        "billing_config_id": agr.billing_config_id,
        "company_id": agr.company_id,
        "revision_number": agr.revision_number,
        "reference": agr.reference,
        "status": agr.status,
        "generated_storage_key": agr.generated_storage_key,
        "generated_sha256": agr.generated_sha256,
        "generated_size_bytes": agr.generated_size_bytes,
        "generated_content_type": agr.generated_content_type,
        "has_generated_docx": bool(agr.generated_storage_key),
        "signed_storage_key": agr.signed_storage_key,
        "signed_sha256": agr.signed_sha256,
        "signed_size_bytes": agr.signed_size_bytes,
        "signed_content_type": agr.signed_content_type,
        "signed_original_filename": agr.signed_original_filename,
        "has_signed_pdf": bool(agr.signed_storage_key),
        "parties_snapshot": agr.parties_snapshot,
        "commercial_snapshot": agr.commercial_snapshot,
        "generation_snapshot": agr.generation_snapshot,
        "generated_at": agr.generated_at.isoformat() if agr.generated_at else None,
        "sent_at": agr.sent_at.isoformat() if agr.sent_at else None,
        "signed_file_uploaded_at": (
            agr.signed_file_uploaded_at.isoformat()
            if agr.signed_file_uploaded_at
            else None
        ),
        "agreement_signed_on": (
            agr.agreement_signed_on.isoformat() if agr.agreement_signed_on else None
        ),
        "agreement_effective_from": (
            agr.agreement_effective_from.isoformat()
            if agr.agreement_effective_from
            else None
        ),
        "void_reason": agr.void_reason,
        "generated_by_user_id": agr.generated_by_user_id,
        "sent_by_user_id": agr.sent_by_user_id,
        "signed_uploaded_by_user_id": agr.signed_uploaded_by_user_id,
        "voided_by_user_id": agr.voided_by_user_id,
    }


def list_agreements_for_config(
    billing_config_id: int,
) -> list[PlatformPartnerAgreement]:
    return (
        PlatformPartnerAgreement.query.filter_by(
            billing_config_id=billing_config_id
        )
        .order_by(PlatformPartnerAgreement.revision_number.desc())
        .all()
    )


def get_active_agreement(
    billing_config_id: int,
) -> PlatformPartnerAgreement | None:
    return (
        PlatformPartnerAgreement.query.filter(
            PlatformPartnerAgreement.billing_config_id == billing_config_id,
            PlatformPartnerAgreement.status.in_(ACTIVE_STATUSES),
        )
        .order_by(PlatformPartnerAgreement.revision_number.desc())
        .first()
    )


def generate_agreement(
    billing_config_id: int,
    *,
    user_id: int | None,
) -> PlatformPartnerAgreement:
    cfg = db.session.get(CompanyPlatformBillingConfig, billing_config_id)
    if not cfg:
        raise PartnerAgreementError("Contrat commercial introuvable", status_code=404)

    company = db.session.get(Company, cfg.company_id)
    if not company:
        raise PartnerAgreementError("Entreprise introuvable", status_code=404)

    profile = CompanyBillingProfile.query.filter_by(company_id=company.id).first()
    partner = resolve_partner_contract_identity(company, profile)
    if not partner.is_complete:
        raise PartnerAgreementError(
            "Identité contractuelle partenaire incomplète : "
            + ", ".join(partner.missing_fields),
            status_code=400,
        )

    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    operator = resolve_operator_contract_identity(creditor)
    if operator is None or not operator.is_complete:
        missing = operator.missing_fields if operator else ["creditor"]
        raise PartnerAgreementError(
            "Identité juridique LIRIE (créancier) incomplète : "
            + ", ".join(missing),
            status_code=400,
        )

    active = _lock_active_agreement(billing_config_id)
    created_path: Path | None = None
    try:
        if active is None:
            revision = _next_revision_number(billing_config_id)
            reference = _allocate_reference()
            agr = PlatformPartnerAgreement(
                billing_config_id=billing_config_id,
                company_id=company.id,
                revision_number=revision,
                reference=reference,
                status=PartnerAgreementStatus.DRAFT.value,
            )
            db.session.add(agr)
            db.session.flush()
        elif active.status == PartnerAgreementStatus.DRAFT.value:
            agr = active
        else:
            raise PartnerAgreementError(
                "Un accord est déjà envoyé ou signé pour cette configuration. "
                "Annulez-le (void) ou créez une nouvelle version commerciale.",
                status_code=409,
            )

        parties_snapshot = {
            "operator": operator.to_snapshot_dict(),
            "partner": partner.to_snapshot_dict(),
        }
        commercial_snapshot = _build_commercial_snapshot(cfg)
        effective_from = _effective_from_date(cfg)
        generation_snapshot = {
            "template_version": TEMPLATE_VERSION,
            "template_sha256": template_sha256(),
            "generator_version": GENERATOR_VERSION,
        }

        docx_bytes = build_partner_agreement_docx_bytes(
            reference=agr.reference,
            parties=parties_snapshot,
            commercial=commercial_snapshot,
            agreement_effective_from=effective_from.isoformat(),
        )
        digest = hashlib.sha256(docx_bytes).hexdigest()
        filename = f"agreement-r{agr.revision_number}.docx"
        rel_key = _storage_key(company.id, agr.id, filename)
        abs_path = get_uploads_base() / rel_key
        write_upload_bytes_atomic(abs_path, docx_bytes)
        created_path = abs_path

        agr.parties_snapshot = parties_snapshot
        agr.commercial_snapshot = commercial_snapshot
        agr.generation_snapshot = generation_snapshot
        agr.agreement_effective_from = effective_from
        agr.generated_storage_key = rel_key
        agr.generated_sha256 = digest
        agr.generated_size_bytes = len(docx_bytes)
        agr.generated_content_type = DOCX_MIME
        agr.generated_at = datetime.now(UTC)
        agr.generated_by_user_id = user_id
        agr.status = PartnerAgreementStatus.DRAFT.value

        db.session.commit()
        created_path = None  # commit OK : ne plus supprimer le fichier
        db.session.refresh(agr)
    except Exception:
        db.session.rollback()
        _best_effort_unlink(created_path)
        raise

    try:
        AuditLogger.log_action(
            action_type="partner_agreement_generated",
            action_category="platform_billing",
            user_id=user_id,
            user_type="admin",
            company_id=company.id,
            action_details={
                "agreement_id": agr.id,
                "billing_config_id": billing_config_id,
                "reference": agr.reference,
                "revision_number": agr.revision_number,
            },
            resource_type="platform_partner_agreement",
            resource_id=str(agr.id),
        )
    except Exception as audit_exc:
        logger.warning(
            "Audit partner_agreement_generated échoué agreement_id=%s: %s",
            agr.id,
            audit_exc,
        )
    return agr


def mark_agreement_sent(
    agreement_id: int, *, user_id: int | None
) -> PlatformPartnerAgreement:
    agr = _lock_agreement(agreement_id)
    if agr.status != PartnerAgreementStatus.DRAFT.value:
        raise PartnerAgreementError(
            "Seul un brouillon peut être marqué comme envoyé",
            status_code=409,
        )
    if not agr.generated_storage_key or not agr.generated_sha256:
        raise PartnerAgreementError(
            "Aucun document généré à envoyer",
            status_code=400,
        )
    agr.status = PartnerAgreementStatus.SENT.value
    agr.sent_at = datetime.now(UTC)
    agr.sent_by_user_id = user_id
    db.session.commit()
    db.session.refresh(agr)
    AuditLogger.log_action(
        action_type="partner_agreement_marked_sent",
        action_category="platform_billing",
        user_id=user_id,
        user_type="admin",
        company_id=agr.company_id,
        action_details={
            "agreement_id": agr.id,
            "reference": agr.reference,
        },
        resource_type="platform_partner_agreement",
        resource_id=str(agr.id),
    )
    return agr


def void_agreement(
    agreement_id: int,
    *,
    reason: str,
    user_id: int | None,
) -> PlatformPartnerAgreement:
    agr = _lock_agreement(agreement_id)
    if agr.status == PartnerAgreementStatus.SIGNED.value:
        raise PartnerAgreementError(
            "Un accord signé ne peut pas être annulé",
            status_code=409,
        )
    if agr.status == PartnerAgreementStatus.VOID.value:
        raise PartnerAgreementError("Accord déjà annulé", status_code=409)
    if agr.status not in (
        PartnerAgreementStatus.DRAFT.value,
        PartnerAgreementStatus.SENT.value,
    ):
        raise PartnerAgreementError("Statut incompatible avec l'annulation", status_code=409)
    reason_clean = (reason or "").strip()
    if not reason_clean:
        raise PartnerAgreementError("Motif d'annulation obligatoire")
    agr.status = PartnerAgreementStatus.VOID.value
    agr.void_reason = reason_clean
    agr.voided_by_user_id = user_id
    db.session.commit()
    db.session.refresh(agr)
    AuditLogger.log_action(
        action_type="partner_agreement_voided",
        action_category="platform_billing",
        user_id=user_id,
        user_type="admin",
        company_id=agr.company_id,
        action_details={
            "agreement_id": agr.id,
            "reference": agr.reference,
            "void_reason": reason_clean,
        },
        resource_type="platform_partner_agreement",
        resource_id=str(agr.id),
    )
    return agr


def upload_signed_pdf(
    agreement_id: int,
    *,
    content: bytes,
    original_filename: str | None,
    agreement_signed_on: date,
    user_id: int | None,
) -> PlatformPartnerAgreement:
    if not content.startswith(b"%PDF"):
        raise PartnerAgreementError(
            "Le contrat signé doit être un PDF (signature %PDF)",
            status_code=400,
        )
    if len(content) > 10 * 1024 * 1024:
        raise PartnerAgreementError(
            "Fichier trop volumineux (max 10 Mo)",
            status_code=400,
        )

    agr = _lock_agreement(agreement_id)
    if agr.status != PartnerAgreementStatus.SENT.value:
        raise PartnerAgreementError(
            "L'upload du PDF signé n'est autorisé qu'après marquage « envoyé »",
            status_code=409,
        )

    created_path: Path | None = None
    try:
        digest = hashlib.sha256(content).hexdigest()
        filename = f"agreement-r{agr.revision_number}-signed.pdf"
        rel_key = _storage_key(agr.company_id, agr.id, filename)
        abs_path = get_uploads_base() / rel_key
        write_upload_bytes_atomic(abs_path, content)
        created_path = abs_path

        agr.signed_storage_key = rel_key
        agr.signed_sha256 = digest
        agr.signed_size_bytes = len(content)
        agr.signed_content_type = PDF_MIME
        agr.signed_original_filename = (original_filename or "")[:255] or None
        agr.agreement_signed_on = agreement_signed_on
        agr.signed_file_uploaded_at = datetime.now(UTC)
        agr.signed_uploaded_by_user_id = user_id
        agr.status = PartnerAgreementStatus.SIGNED.value

        db.session.commit()
        created_path = None
        db.session.refresh(agr)
    except Exception:
        db.session.rollback()
        _best_effort_unlink(created_path)
        raise

    try:
        AuditLogger.log_action(
            action_type="partner_agreement_signed_uploaded",
            action_category="platform_billing",
            user_id=user_id,
            user_type="admin",
            company_id=agr.company_id,
            action_details={
                "agreement_id": agr.id,
                "reference": agr.reference,
                "agreement_signed_on": agreement_signed_on.isoformat(),
                "sha256": digest,
            },
            resource_type="platform_partner_agreement",
            resource_id=str(agr.id),
        )
    except Exception as audit_exc:
        logger.warning(
            "Audit partner_agreement_signed_uploaded échoué agreement_id=%s: %s",
            agr.id,
            audit_exc,
        )
    return agr


def default_free_license_months(mode: str | None) -> int | None:
    if mode == SubscriptionPricingMode.FREE.value:
        return 60
    return None
