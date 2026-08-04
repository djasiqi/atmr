"""Lifecycle des accords juridiques partenaires (pack PDF + ZIP de remise)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, date, datetime
from io import BytesIO
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
from services.platform_billing.partner_agreement_canonical import (
    CanonicalDocumentError,
    ensure_canonical_documents,
)
from services.platform_billing.partner_agreement_compliance import (
    compliance_snapshot,
)
from services.platform_billing.partner_agreement_manifest_pdf import (
    build_delivery_manifest_pdf_bytes,
)
from services.platform_billing.partner_agreement_package import (
    build_delivery_zip_bytes,
    delivery_zip_filename,
    sha256_bytes,
)
from services.platform_billing.partner_agreement_particular_content import (
    build_particular_agreement_content,
)
from services.platform_billing.partner_agreement_particular_docx import (
    build_particular_docx_bytes,
)
from services.platform_billing.partner_agreement_particular_pdf import (
    PartnerAgreementLayoutError,
    build_particular_pdf_bytes,
    count_pdf_pages,
)
from services.platform_billing.partner_agreement_preview import (
    apply_draft_watermark,
)
from services.platform_billing.partner_agreement_versions import (
    COMMERCIAL_SNAPSHOT_SCHEMA_VERSION,
    DPA_VERSION,
    GENERAL_TERMS_VERSION,
    GENERATOR_VERSION,
    PACK_SCHEMA_VERSION,
    PARTICULAR_VERSION,
    PENALTY_CALCULATION_VERSION,
    PENALTY_CURRENCY,
    PENALTY_MINIMUM_CHF,
    PENALTY_MULTIPLIER,
    RETENTION_POLICY_VERSION,
    SPECIAL_CONDITIONS_MAX_LENGTH,
    SUBPROCESSORS_VERSION,
    TEMPLATE_VERSION,
)
from services.platform_billing.partner_identity import (
    resolve_operator_contract_identity,
    resolve_partner_contract_identity,
)
from services.platform_billing.subscription_pricing_resolver import (
    ensure_contract_pricing_grid,
    resolve_subscription_pricing,
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
MIGRATE_VOID_REASON = "migrated_to_partner_pack_v1"
# Compat anciens messages / routes
LEGAL_TEXT_VERSION = PARTICULAR_VERSION


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


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def legal_text_sha256() -> str:
    """Empreinte des versions pack + artefacts canoniques vérifiés."""
    try:
        canon = ensure_canonical_documents()
        terms_sha = canon["general_terms"].sha256
        dpa_sha = canon["dpa"].sha256
    except CanonicalDocumentError:
        terms_sha = "missing"
        dpa_sha = "missing"
    blob = (
        f"{PACK_SCHEMA_VERSION}|{PARTICULAR_VERSION}|{GENERATOR_VERSION}|"
        f"{GENERAL_TERMS_VERSION}|{terms_sha}|"
        f"{DPA_VERSION}|{dpa_sha}|"
        f"{RETENTION_POLICY_VERSION}|{SUBPROCESSORS_VERSION}"
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _fmt_dt_fr(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    local = dt.astimezone(_ZURICH) if dt.tzinfo else dt.replace(tzinfo=UTC).astimezone(_ZURICH)
    return local.strftime("%d.%m.%Y %H:%M %Z")


def _partner_display_name(parties: dict[str, Any] | None) -> str:
    partner = (parties or {}).get("partner") or {}
    return (partner.get("legal_name") or "").strip() or "—"


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


def _storage_key(company_id: int, agreement_id: int, filename: str) -> str:
    return f"platform_agreements/{company_id}/{agreement_id}/{filename}"


def _dispatch_mode_for_company(cfg: CompanyPlatformBillingConfig) -> str:
    company = db.session.get(Company, cfg.company_id)
    override = getattr(cfg, "dispatch_mode_override", None)
    if override:
        return str(override)
    if company and getattr(company, "dispatch_mode", None):
        return str(company.dispatch_mode)
    return "manual"


def _effective_from_date(cfg: CompanyPlatformBillingConfig) -> date:
    if cfg.effective_from is None:
        return datetime.now(_ZURICH).date()
    dt = cfg.effective_from
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_ZURICH).date()


def _period_start_for_cfg(cfg: CompanyPlatformBillingConfig) -> datetime:
    d = _effective_from_date(cfg)
    local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=_ZURICH)
    return local.astimezone(UTC)


def _normalize_special_conditions(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) > SPECIAL_CONDITIONS_MAX_LENGTH:
        raise PartnerAgreementError(
            "Conditions particulières : maximum "
            f"{SPECIAL_CONDITIONS_MAX_LENGTH} caractères.",
            status_code=400,
        )
    return text


def validate_signatory_authority_verification(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PartnerAgreementError(
            "Attestation du pouvoir de signature (Registre du commerce) obligatoire.",
            status_code=400,
        )
    attested = bool(payload.get("attested"))
    if not attested:
        raise PartnerAgreementError(
            "Le pouvoir de signature doit être attesté "
            "sur la base du Registre du commerce.",
            status_code=400,
        )
    mode = str(payload.get("signature_mode") or "individual").strip().lower()
    if mode not in {"individual", "collective"}:
        raise PartnerAgreementError(
            "Mode de signature invalide (individual ou collective).",
            status_code=400,
        )
    signatory_name = str(payload.get("signatory_name") or "").strip()
    signatory_function = str(payload.get("signatory_function") or "").strip()
    if not signatory_name or not signatory_function:
        raise PartnerAgreementError(
            "Nom et fonction du signataire requis pour l'attestation RC.",
            status_code=400,
        )
    co_name = str(payload.get("co_signatory_name") or "").strip() or None
    co_function = str(payload.get("co_signatory_function") or "").strip() or None
    if mode == "collective" and (not co_name or not co_function):
        raise PartnerAgreementError(
            "Signature collective : second signataire (nom et fonction) obligatoire.",
            status_code=400,
        )
    return {
        "source": str(payload.get("source") or "registre_du_commerce").strip(),
        "register_name": str(
            payload.get("register_name")
            or "Registre du commerce / Zefix"
        ).strip(),
        "checked_at": str(
            payload.get("checked_at") or datetime.now(_ZURICH).isoformat()
        ),
        "company_uid": str(payload.get("company_uid") or "").strip() or None,
        "signatory_name": signatory_name,
        "signatory_function": signatory_function,
        "signature_mode": mode,
        "co_signatory_required": mode == "collective",
        "co_signatory_name": co_name if mode == "collective" else None,
        "co_signatory_function": co_function if mode == "collective" else None,
        "attested": True,
    }


def build_commercial_snapshot(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    """Snapshot commercial déterministe (partagé génération / contrôle d'intégrité)."""
    from services.platform_billing.dunning_policy import serialize_dunning_fields

    mode = cfg.subscription_pricing_mode or SubscriptionPricingMode.VOLUME.value
    period_start = _period_start_for_cfg(cfg)
    dm = _dispatch_mode_for_company(cfg)
    resolution = resolve_subscription_pricing(
        cfg,
        period_start=period_start,
        pricing_mode=mode,
        dispatch_mode=dm,
        own_portfolio_enabled=bool(cfg.own_portfolio_billing_enabled),
    )
    special = _normalize_special_conditions(
        getattr(cfg, "contract_special_conditions", None)
    )
    return {
        "commercial_snapshot_schema_version": COMMERCIAL_SNAPSHOT_SCHEMA_VERSION,
        "billing_config_id": cfg.id,
        "is_billing_enabled": bool(cfg.is_billing_enabled),
        "own_portfolio_billing_enabled": bool(cfg.own_portfolio_billing_enabled),
        "lirie_commission_enabled": bool(cfg.lirie_commission_enabled),
        "support_enabled": bool(cfg.support_enabled),
        "subscription_pricing_mode": mode,
        "custom_subscription_amount": decimal_to_str(cfg.custom_subscription_amount),
        "use_global_pricing_grid": bool(getattr(cfg, "use_global_pricing_grid", True)),
        "pricing_grid_id": getattr(cfg, "pricing_grid_id", None),
        "commission_rate": decimal_to_str(cfg.commission_rate, places=6),
        "commission_cancellation_policy": cfg.commission_cancellation_policy,
        "commission_due_if_customer_unpaid": True,
        "free_license_max_months": cfg.free_license_max_months,
        "no_automatic_post_free_transition": True,
        "statement_dispute_days": cfg.statement_dispute_days
        if cfg.statement_dispute_days is not None
        else 10,
        "payment_terms_days": cfg.payment_terms_days,
        "support_hourly_rate_default": decimal_to_str(
            cfg.support_hourly_rate_default
        ),
        "amounts_are_tax_inclusive": bool(
            getattr(cfg, "amounts_are_tax_inclusive", False)
        ),
        "tax_rate_override": decimal_to_str(
            getattr(cfg, "tax_rate_override", None), places=4
        ),
        "contract_special_conditions": special,
        "penalty": {
            "multiplier": PENALTY_MULTIPLIER,
            "minimum": PENALTY_MINIMUM_CHF,
            "currency": PENALTY_CURRENCY,
            "calculation_version": PENALTY_CALCULATION_VERSION,
        },
        "subscription_pricing": resolution.to_snapshot_dict(),
        "compliance": compliance_snapshot(),
        "effective_from": cfg.effective_from.isoformat()
        if cfg.effective_from
        else None,
        "effective_to": cfg.effective_to.isoformat() if cfg.effective_to else None,
        **serialize_dunning_fields(cfg),
    }


# Alias rétrocompat tests / imports internes
_build_commercial_snapshot = build_commercial_snapshot


def default_free_license_months(mode: str) -> int | None:
    if mode == SubscriptionPricingMode.FREE.value:
        return 60
    return None


def serialize_agreement(agr: PlatformPartnerAgreement) -> dict[str, Any]:
    gen = agr.generation_snapshot or {}
    internal = gen.get("internal_docx") or {}
    has_pdf = bool(
        agr.generated_storage_key
        and (agr.generated_content_type or "").startswith("application/pdf")
    )
    # Anciens brouillons DOCX : content_type DOCX ou clé .docx
    has_legacy_docx_as_generated = bool(
        agr.generated_storage_key
        and not has_pdf
        and (
            (agr.generated_content_type or "") == DOCX_MIME
            or str(agr.generated_storage_key).endswith(".docx")
        )
    )
    has_internal_docx = bool(internal.get("storage_key")) or has_legacy_docx_as_generated
    is_sent_or_signed = agr.status in (
        PartnerAgreementStatus.SENT.value,
        PartnerAgreementStatus.SIGNED.value,
    )
    pack_ok = (gen.get("pack_schema_version") or "") == PACK_SCHEMA_VERSION
    return {
        "id": agr.id,
        "billing_config_id": agr.billing_config_id,
        "company_id": agr.company_id,
        "revision_number": agr.revision_number,
        "reference": agr.reference,
        "status": agr.status,
        "template_version": gen.get("pack_schema_version")
        or gen.get("template_version"),
        "pack_schema_version": gen.get("pack_schema_version"),
        "main_contract_version": gen.get("main_contract_version"),
        "generator_version": gen.get("generator_version"),
        "generated_storage_key": agr.generated_storage_key,
        "generated_sha256": agr.generated_sha256,
        "generated_size_bytes": agr.generated_size_bytes,
        "generated_content_type": agr.generated_content_type,
        "has_generated_particular_pdf": has_pdf,
        "has_internal_docx": has_internal_docx,
        "has_delivery_package": bool(gen.get("delivery_zip_key")) and is_sent_or_signed,
        "particular_pdf_available_for_signature": has_pdf and is_sent_or_signed,
        # Alias déprécié : pointe vers le DOCX interne (plus le fichier généré principal).
        "has_generated_docx": has_internal_docx,
        "signed_storage_key": agr.signed_storage_key,
        "signed_sha256": agr.signed_sha256,
        "signed_size_bytes": agr.signed_size_bytes,
        "signed_content_type": agr.signed_content_type,
        "signed_original_filename": agr.signed_original_filename,
        "has_signed_pdf": bool(agr.signed_storage_key),
        "signed_document_validation": gen.get("signed_document_validation"),
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
        "needs_v120_migration": (
            agr.status == PartnerAgreementStatus.DRAFT.value
            and bool(agr.generated_storage_key)
            and not pack_ok
        ),
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


def _pin_contract_grid_if_volume(
    cfg: CompanyPlatformBillingConfig,
    *,
    reference: str,
    revision_number: int,
) -> dict[str, Any]:
    """Matérialise une grille contractuelle et pin la config (mode volume)."""
    mode = cfg.subscription_pricing_mode or SubscriptionPricingMode.VOLUME.value
    if mode != SubscriptionPricingMode.VOLUME.value:
        return build_commercial_snapshot(cfg)
    if not cfg.own_portfolio_billing_enabled:
        return build_commercial_snapshot(cfg)

    period_start = _period_start_for_cfg(cfg)
    resolution = resolve_subscription_pricing(
        cfg,
        period_start=period_start,
        pricing_mode=mode,
        dispatch_mode=_dispatch_mode_for_company(cfg),
        own_portfolio_enabled=True,
    )
    if not resolution.is_valid or not resolution.tiers:
        raise PartnerAgreementError(
            "Mode volume : grille tarifaire invalide ou sans paliers. "
            "Corrigez la configuration avant de générer le contrat.",
            status_code=400,
        )
    grid = ensure_contract_pricing_grid(
        billing_config_id=cfg.id,
        revision_number=revision_number,
        reference=reference,
        resolution=resolution,
    )
    cfg.pricing_grid_id = grid.id
    cfg.use_global_pricing_grid = False
    db.session.flush()
    return build_commercial_snapshot(cfg)


def _compose_and_store_pack(
    agr: PlatformPartnerAgreement,
    *,
    company: Company,
    cfg: CompanyPlatformBillingConfig,
    parties_snapshot: dict[str, Any],
    commercial_snapshot: dict[str, Any],
    user_id: int | None,
) -> list[Path]:
    """Génère PDF officiel + DOCX interne ; retourne les chemins créés (cleanup)."""
    effective_from = _effective_from_date(cfg)
    try:
        canon = ensure_canonical_documents()
    except CanonicalDocumentError as exc:
        raise PartnerAgreementError(exc.message, status_code=500) from exc

    content = build_particular_agreement_content(
        reference=agr.reference,
        parties=parties_snapshot,
        commercial=commercial_snapshot,
        agreement_effective_from=effective_from.isoformat(),
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_sha256=canon["dpa"].sha256,
        general_terms_version=canon["general_terms"].version,
        dpa_version=canon["dpa"].version,
    )
    try:
        pdf_bytes = build_particular_pdf_bytes(content)
    except PartnerAgreementLayoutError as exc:
        raise PartnerAgreementError(exc.message, status_code=500) from exc
    docx_bytes = build_particular_docx_bytes(content)

    pdf_name = f"particular-r{agr.revision_number}.pdf"
    docx_name = f"particular-r{agr.revision_number}.docx"
    pdf_key = _storage_key(company.id, agr.id, pdf_name)
    docx_key = _storage_key(company.id, agr.id, docx_name)
    pdf_path = get_uploads_base() / pdf_key
    docx_path = get_uploads_base() / docx_key
    created: list[Path] = []
    try:
        write_upload_bytes_atomic(pdf_path, pdf_bytes)
        created.append(pdf_path)
        write_upload_bytes_atomic(docx_path, docx_bytes)
        created.append(docx_path)
    except Exception:
        for path in created:
            _best_effort_unlink(path)
        raise

    generation_snapshot = {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "template_version": PACK_SCHEMA_VERSION,  # compat gate / FE
        "main_contract_version": PARTICULAR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "legal_text_version": PARTICULAR_VERSION,
        "legal_text_sha256": legal_text_sha256(),
        "commercial_snapshot_schema_version": COMMERCIAL_SNAPSHOT_SCHEMA_VERSION,
        "parties_snapshot_sha256": canonical_json_sha256(parties_snapshot),
        "commercial_snapshot_sha256": canonical_json_sha256(commercial_snapshot),
        "internal_docx": {
            "storage_key": docx_key,
            "sha256": sha256_bytes(docx_bytes),
            "content_type": DOCX_MIME,
            "size_bytes": len(docx_bytes),
        },
        "general_terms_version": canon["general_terms"].version,
        "general_terms_sha256": canon["general_terms"].sha256,
        "dpa_version": canon["dpa"].version,
        "dpa_sha256": canon["dpa"].sha256,
        "retention_policy_version": RETENTION_POLICY_VERSION,
        "subprocessors_version": SUBPROCESSORS_VERSION,
        "penalty_calculation_version": PENALTY_CALCULATION_VERSION,
    }

    agr.parties_snapshot = parties_snapshot
    agr.commercial_snapshot = commercial_snapshot
    agr.generation_snapshot = generation_snapshot
    agr.agreement_effective_from = effective_from
    agr.generated_storage_key = pdf_key
    agr.generated_sha256 = sha256_bytes(pdf_bytes)
    agr.generated_size_bytes = len(pdf_bytes)
    agr.generated_content_type = PDF_MIME
    agr.generated_at = datetime.now(UTC)
    agr.generated_by_user_id = user_id
    agr.status = PartnerAgreementStatus.DRAFT.value
    return created


def generate_agreement(
    billing_config_id: int,
    *,
    user_id: int | None,
    signatory_authority_verification: dict[str, Any] | None = None,
    commit: bool = True,
) -> PlatformPartnerAgreement:
    cfg = db.session.get(CompanyPlatformBillingConfig, billing_config_id)
    if not cfg:
        raise PartnerAgreementError("Contrat commercial introuvable", status_code=404)

    company = db.session.get(Company, cfg.company_id)
    if not company:
        raise PartnerAgreementError("Entreprise introuvable", status_code=404)

    profile = CompanyBillingProfile.query.filter_by(company_id=company.id).first()
    partner = resolve_partner_contract_identity(company, profile)
    if not partner.is_complete(require_uid_ide=True):
        raise PartnerAgreementError(
            "Identité contractuelle partenaire incomplète : "
            + ", ".join(partner.missing_fields(require_uid_ide=True))
            + ". Renseignez IDE, forme juridique et signataire dans le modal.",
            status_code=400,
        )

    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    operator = resolve_operator_contract_identity(creditor)
    if operator is None or not operator.is_complete(require_uid_ide=False):
        missing = (
            operator.missing_fields(require_uid_ide=False)
            if operator
            else ["crediteur_absent"]
        )
        raise PartnerAgreementError(
            "Identité juridique LIRIE (créancier) incomplète : "
            + ", ".join(missing)
            + ". Complétez Paramètres admin → Facturation plateforme LIRIE "
            "(forme juridique et signataire au minimum).",
            status_code=400,
        )

    rc = validate_signatory_authority_verification(signatory_authority_verification)
    rc["checked_by_user_id"] = user_id

    active = _lock_active_agreement(billing_config_id)
    created_paths: list[Path] = []
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

        commercial_snapshot = _pin_contract_grid_if_volume(
            cfg,
            reference=agr.reference,
            revision_number=agr.revision_number,
        )
        parties_snapshot = {
            "operator": operator.to_snapshot_dict(),
            "partner": partner.to_snapshot_dict(),
            "signatory_authority_verification": rc,
        }
        created_paths = _compose_and_store_pack(
            agr,
            company=company,
            cfg=cfg,
            parties_snapshot=parties_snapshot,
            commercial_snapshot=commercial_snapshot,
            user_id=user_id,
        )
        if commit:
            db.session.commit()
            created_paths = []
            db.session.refresh(agr)
        else:
            db.session.flush()
    except Exception:
        if commit:
            db.session.rollback()
        for path in created_paths:
            _best_effort_unlink(path)
        raise

    if commit:
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
                    "pack_schema_version": PACK_SCHEMA_VERSION,
                    "main_contract_version": PARTICULAR_VERSION,
                    "generated_sha256": agr.generated_sha256,
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


def _assert_generation_integrity(agr: PlatformPartnerAgreement) -> None:
    gen = agr.generation_snapshot or {}
    pack = gen.get("pack_schema_version") or gen.get("template_version")
    if pack != PACK_SCHEMA_VERSION:
        raise PartnerAgreementError(
            "Ce brouillon utilise un ancien modèle contractuel. "
            "Migrez-le vers le pack partenaire avant l'envoi.",
            status_code=409,
        )

    parties = agr.parties_snapshot
    commercial = agr.commercial_snapshot
    if not parties or not commercial:
        raise PartnerAgreementError(
            "Snapshots manquants : régénérez le contrat.",
            status_code=409,
        )
    if canonical_json_sha256(parties) != gen.get("parties_snapshot_sha256"):
        raise PartnerAgreementError(
            "Intégrité parties_snapshot compromise. Régénérez le contrat.",
            status_code=409,
        )
    if canonical_json_sha256(commercial) != gen.get("commercial_snapshot_sha256"):
        raise PartnerAgreementError(
            "Intégrité commercial_snapshot compromise. Régénérez le contrat.",
            status_code=409,
        )

    if not agr.generated_storage_key or not agr.generated_sha256:
        raise PartnerAgreementError(
            "PDF particulier manquant. Régénérez le contrat.",
            status_code=409,
        )
    if (agr.generated_content_type or "") != PDF_MIME:
        raise PartnerAgreementError(
            "L'original juridique doit être le PDF particulier. "
            "Migrez ou régénérez le contrat.",
            status_code=409,
        )
    pdf_path = get_uploads_base() / agr.generated_storage_key
    if not pdf_path.is_file():
        raise PartnerAgreementError(
            "Fichier PDF particulier introuvable sur le stockage.",
            status_code=409,
        )
    if sha256_bytes(pdf_path.read_bytes()) != agr.generated_sha256:
        raise PartnerAgreementError(
            "Intégrité du PDF particulier compromise. Régénérez le contrat.",
            status_code=409,
        )

    internal = gen.get("internal_docx") or {}
    docx_key = internal.get("storage_key")
    if docx_key:
        docx_path = get_uploads_base() / str(docx_key)
        if not docx_path.is_file():
            raise PartnerAgreementError(
                "DOCX interne introuvable. Régénérez le contrat.",
                status_code=409,
            )
        if sha256_bytes(docx_path.read_bytes()) != internal.get("sha256"):
            raise PartnerAgreementError(
                "Intégrité du DOCX interne compromise. Régénérez le contrat.",
                status_code=409,
            )

    try:
        canon = ensure_canonical_documents(
            general_terms_version=str(
                gen.get("general_terms_version") or GENERAL_TERMS_VERSION
            ),
            dpa_version=str(gen.get("dpa_version") or DPA_VERSION),
        )
    except CanonicalDocumentError as exc:
        raise PartnerAgreementError(exc.message, status_code=500) from exc
    if canon["general_terms"].sha256 != gen.get("general_terms_sha256"):
        raise PartnerAgreementError(
            "SHA des Conditions générales incohérent. Régénérez le contrat.",
            status_code=409,
        )
    if canon["dpa"].sha256 != gen.get("dpa_sha256"):
        raise PartnerAgreementError(
            "SHA de l'Accord de traitement incohérent. Régénérez le contrat.",
            status_code=409,
        )

    cfg = db.session.get(CompanyPlatformBillingConfig, agr.billing_config_id)
    if not cfg:
        raise PartnerAgreementError("Configuration introuvable", status_code=404)
    live = build_commercial_snapshot(cfg)
    if canonical_json_sha256(live) != gen.get("commercial_snapshot_sha256"):
        raise PartnerAgreementError(
            "La configuration commerciale a été modifiée depuis la génération "
            "du contrat. Régénérez le document avant de l'envoyer.",
            status_code=409,
        )
    snap_grid = (commercial.get("subscription_pricing") or {}).get("resolved_grid_id")
    mode = commercial.get("subscription_pricing_mode")
    if (
        mode == SubscriptionPricingMode.VOLUME.value
        and commercial.get("own_portfolio_billing_enabled")
    ):
        if cfg.use_global_pricing_grid:
            raise PartnerAgreementError(
                "Incohérence : use_global_pricing_grid doit être "
                "false après génération.",
                status_code=409,
            )
        if cfg.pricing_grid_id != snap_grid:
            raise PartnerAgreementError(
                "Incohérence de grille contractuelle. Régénérez le contrat.",
                status_code=409,
            )


def _normalize_delivery_declaration(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not payload:
        return {
            "confirmed": True,
            "channel": "unspecified",
            "recipient": None,
        }
    if not isinstance(payload, dict):
        raise PartnerAgreementError(
            "Déclaration de remise invalide.",
            status_code=400,
        )
    channel = str(payload.get("channel") or "unspecified").strip()[:64]
    recipient = (payload.get("recipient") or None)
    if recipient is not None:
        recipient = str(recipient).strip()[:255] or None
    return {
        "confirmed": bool(payload.get("confirmed", True)),
        "channel": channel or "unspecified",
        "recipient": recipient,
    }


def mark_agreement_sent(
    agreement_id: int,
    *,
    user_id: int | None,
    delivery_declaration: dict[str, Any] | None = None,
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
    _assert_generation_integrity(agr)
    declaration = _normalize_delivery_declaration(delivery_declaration)

    gen = dict(agr.generation_snapshot or {})
    particular_path = get_uploads_base() / agr.generated_storage_key
    particular_pdf = particular_path.read_bytes()
    try:
        canon = ensure_canonical_documents(
            general_terms_version=str(
                gen.get("general_terms_version") or GENERAL_TERMS_VERSION
            ),
            dpa_version=str(gen.get("dpa_version") or DPA_VERSION),
        )
    except CanonicalDocumentError as exc:
        raise PartnerAgreementError(exc.message, status_code=500) from exc

    sent_at = datetime.now(UTC)
    agr.sent_at = sent_at
    agr.sent_by_user_id = user_id

    manifest_name = f"bordereau-r{agr.revision_number}.pdf"
    zip_name = f"dossier-remise-r{agr.revision_number}.zip"
    manifest_key = _storage_key(agr.company_id, agr.id, manifest_name)
    zip_key = _storage_key(agr.company_id, agr.id, zip_name)
    manifest_path = get_uploads_base() / manifest_key
    zip_path = get_uploads_base() / zip_key
    created: list[Path] = []
    try:
        manifest_pdf = build_delivery_manifest_pdf_bytes(
            reference=agr.reference,
            partner_name=_partner_display_name(agr.parties_snapshot),
            finalized_at_fr=_fmt_dt_fr(sent_at),
            particular_version=str(
                gen.get("main_contract_version") or PARTICULAR_VERSION
            ),
            particular_sha256=str(agr.generated_sha256),
            general_terms_version=canon["general_terms"].version,
            general_terms_sha256=canon["general_terms"].sha256,
            dpa_version=canon["dpa"].version,
            dpa_sha256=canon["dpa"].sha256,
            retention_policy_version=str(
                gen.get("retention_policy_version") or RETENTION_POLICY_VERSION
            ),
            subprocessors_version=str(
                gen.get("subprocessors_version") or SUBPROCESSORS_VERSION
            ),
            delivery_declaration=declaration,
        )
        zip_bytes = build_delivery_zip_bytes(
            reference=agr.reference,
            manifest_pdf=manifest_pdf,
            particular_pdf=particular_pdf,
            general_terms_pdf=canon["general_terms"].pdf_bytes,
            dpa_pdf=canon["dpa"].pdf_bytes,
            general_terms_version=canon["general_terms"].version,
            dpa_version=canon["dpa"].version,
        )
        write_upload_bytes_atomic(manifest_path, manifest_pdf)
        created.append(manifest_path)
        write_upload_bytes_atomic(zip_path, zip_bytes)
        created.append(zip_path)

        if sha256_bytes(manifest_path.read_bytes()) != sha256_bytes(manifest_pdf):
            raise PartnerAgreementError(
                "Écriture du bordereau incohérente.",
                status_code=500,
            )
        if sha256_bytes(zip_path.read_bytes()) != sha256_bytes(zip_bytes):
            raise PartnerAgreementError(
                "Écriture du ZIP incohérente.",
                status_code=500,
            )

        gen["delivery_manifest_key"] = manifest_key
        gen["delivery_manifest_sha256"] = sha256_bytes(manifest_pdf)
        gen["delivery_zip_key"] = zip_key
        gen["delivery_zip_sha256"] = sha256_bytes(zip_bytes)
        gen["delivery_declaration"] = declaration
        gen["delivery_zip_filename"] = delivery_zip_filename(agr.reference)
        agr.generation_snapshot = gen
        agr.status = PartnerAgreementStatus.SENT.value
        db.session.commit()
        created = []
        db.session.refresh(agr)
    except Exception:
        db.session.rollback()
        for path in created:
            _best_effort_unlink(path)
        raise

    AuditLogger.log_action(
        action_type="partner_agreement_marked_sent",
        action_category="platform_billing",
        user_id=user_id,
        user_type="admin",
        company_id=agr.company_id,
        action_details={
            "agreement_id": agr.id,
            "reference": agr.reference,
            "pack_schema_version": PACK_SCHEMA_VERSION,
            "delivery_zip_sha256": (agr.generation_snapshot or {}).get(
                "delivery_zip_sha256"
            ),
            "delivery_manifest_sha256": (agr.generation_snapshot or {}).get(
                "delivery_manifest_sha256"
            ),
            "delivery_declaration": declaration,
        },
        resource_type="platform_partner_agreement",
        resource_id=str(agr.id),
    )
    return agr


def migrate_draft_agreement_to_v120(
    agreement_id: int,
    *,
    user_id: int | None,
    signatory_authority_verification: dict[str, Any] | None,
) -> PlatformPartnerAgreement:
    """Migration atomique : void ancien draft + nouvelle révision pack v1."""
    old = _lock_agreement(agreement_id)
    if old.status != PartnerAgreementStatus.DRAFT.value:
        raise PartnerAgreementError(
            "Seuls les brouillons peuvent être migrés vers le pack partenaire.",
            status_code=409,
        )
    gen = old.generation_snapshot or {}
    from_version = (
        gen.get("pack_schema_version") or gen.get("template_version") or "unknown"
    )
    if (
        from_version == PACK_SCHEMA_VERSION
        and (old.generated_content_type or "") == PDF_MIME
    ):
        raise PartnerAgreementError(
            "Ce brouillon est déjà au pack partenaire actuel.",
            status_code=409,
        )

    billing_config_id = old.billing_config_id
    company_id = old.company_id
    old_id = old.id
    try:
        old.status = PartnerAgreementStatus.VOID.value
        old.void_reason = MIGRATE_VOID_REASON
        old.voided_by_user_id = user_id
        db.session.flush()

        new_agr = generate_agreement(
            billing_config_id,
            user_id=user_id,
            signatory_authority_verification=signatory_authority_verification,
            commit=False,
        )
        db.session.commit()
        db.session.refresh(new_agr)
    except Exception:
        db.session.rollback()
        raise

    try:
        AuditLogger.log_action(
            action_type="partner_agreement_migrated",
            action_category="platform_billing",
            user_id=user_id,
            user_type="admin",
            company_id=company_id,
            action_details={
                "migrated_from_agreement_id": old_id,
                "migrated_to_agreement_id": new_agr.id,
                "from_template_version": from_version,
                "to_pack_schema_version": PACK_SCHEMA_VERSION,
            },
            resource_type="platform_partner_agreement",
            resource_id=str(new_agr.id),
        )
    except Exception as audit_exc:
        logger.warning(
            "Audit partner_agreement_migrated échoué: %s",
            audit_exc,
        )
    return new_agr


def read_particular_pdf_bytes(agr: PlatformPartnerAgreement) -> bytes:
    if not agr.generated_storage_key:
        raise PartnerAgreementError("PDF particulier introuvable", status_code=404)
    path = get_uploads_base() / agr.generated_storage_key
    if not path.is_file():
        raise PartnerAgreementError("Fichier PDF introuvable", status_code=404)
    return path.read_bytes()


def build_preview_pdf_bytes(agr: PlatformPartnerAgreement) -> bytes:
    if agr.status != PartnerAgreementStatus.DRAFT.value:
        raise PartnerAgreementError(
            "La prévisualisation filigranée n'est disponible qu'en brouillon.",
            status_code=409,
        )
    return apply_draft_watermark(read_particular_pdf_bytes(agr))


def read_internal_docx_key(agr: PlatformPartnerAgreement) -> str:
    gen = agr.generation_snapshot or {}
    internal = gen.get("internal_docx") or {}
    key = internal.get("storage_key")
    if key:
        return str(key)
    # Legacy : ancien generated_* DOCX
    if agr.generated_storage_key and str(agr.generated_storage_key).endswith(".docx"):
        return agr.generated_storage_key
    raise PartnerAgreementError("DOCX interne introuvable", status_code=404)


def read_delivery_zip_key(agr: PlatformPartnerAgreement) -> str:
    gen = agr.generation_snapshot or {}
    key = gen.get("delivery_zip_key")
    if not key:
        raise PartnerAgreementError(
            "Dossier de remise indisponible (accord non finalisé).",
            status_code=404,
        )
    if agr.status not in (
        PartnerAgreementStatus.SENT.value,
        PartnerAgreementStatus.SIGNED.value,
    ):
        raise PartnerAgreementError(
            "Le dossier complet n'est disponible qu'après marquage envoyé.",
            status_code=409,
        )
    return str(key)


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
        raise PartnerAgreementError(
            "Statut incompatible avec l'annulation",
            status_code=409,
        )
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


def _validate_signed_pdf(
    agr: PlatformPartnerAgreement,
    content: bytes,
    *,
    additional_pages_confirmed: bool = False,
) -> dict[str, Any]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(content))
        total_pages = len(reader.pages)
        extracted = []
        for page in reader.pages[:3]:
            try:
                extracted.append(page.extract_text() or "")
            except Exception:
                extracted.append("")
        text = "\n".join(extracted)
    except Exception as exc:
        raise PartnerAgreementError(
            f"PDF signé illisible : {exc}",
            status_code=400,
        ) from exc

    reference_found = agr.reference in text if text.strip() else False
    scannish = not bool(text.strip())

    if total_pages < 3:
        raise PartnerAgreementError(
            "Le contrat particulier signé doit comporter au moins les "
            "trois pages contractuelles (document incomplet).",
            status_code=400,
        )
    if total_pages == 3:
        kind = "none"
    else:
        if not additional_pages_confirmed:
            raise PartnerAgreementError(
                "Le PDF comporte plus de trois pages. Confirmez qu'il s'agit "
                "de pages de certificat / journal de signature électronique "
                "(additional_pages_confirmed=true).",
                status_code=400,
            )
        kind = "signature_certificate"

    if text.strip() and not reference_found:
        raise PartnerAgreementError(
            "La référence contractuelle est introuvable dans les trois "
            "premières pages du PDF signé.",
            status_code=400,
        )

    return {
        "contract_page_count": 3,
        "total_page_count": total_pages,
        "reference_found": reference_found,
        "additional_pages_kind": kind,
        "manually_confirmed": bool(additional_pages_confirmed) and total_pages > 3,
        "text_extractable": not scannish,
        "unsigned_particular_sha256": agr.generated_sha256,
    }


def upload_signed_pdf(
    agreement_id: int,
    *,
    content: bytes,
    original_filename: str | None,
    agreement_signed_on: date,
    user_id: int | None,
    additional_pages_confirmed: bool = False,
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
            "Seul un accord envoyé peut recevoir le PDF signé",
            status_code=409,
        )

    validation = _validate_signed_pdf(
        agr,
        content,
        additional_pages_confirmed=additional_pages_confirmed,
    )

    digest = hashlib.sha256(content).hexdigest()
    filename = f"particular-r{agr.revision_number}-signed.pdf"
    rel_key = _storage_key(agr.company_id, agr.id, filename)
    abs_path = get_uploads_base() / rel_key
    created_path: Path | None = abs_path
    try:
        write_upload_bytes_atomic(abs_path, content)
        snapshot = dict(agr.generation_snapshot or {})
        snapshot["signed_document_validation"] = validation
        agr.generation_snapshot = snapshot
        agr.signed_storage_key = rel_key
        agr.signed_sha256 = digest
        agr.signed_size_bytes = len(content)
        agr.signed_content_type = PDF_MIME
        agr.signed_original_filename = (original_filename or filename)[:255]
        agr.signed_file_uploaded_at = datetime.now(UTC)
        agr.signed_uploaded_by_user_id = user_id
        agr.agreement_signed_on = agreement_signed_on
        agr.status = PartnerAgreementStatus.SIGNED.value
        db.session.commit()
        created_path = None
        db.session.refresh(agr)
    except Exception:
        db.session.rollback()
        _best_effort_unlink(created_path)
        raise

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
            "signed_sha256": digest,
            "unsigned_particular_sha256": agr.generated_sha256,
            "signed_document_validation": validation,
        },
        resource_type="platform_partner_agreement",
        resource_id=str(agr.id),
    )
    return agr
