"""Tests accords partenaires pack (PDF particulier, CG/DPA, lifecycle)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.enums import LegalForm, PartnerAgreementStatus
from services.platform_billing.partner_agreement_versions import (
    DPA_VERSION,
    GENERAL_TERMS_VERSION,
    PACK_SCHEMA_VERSION,
    PARTICULAR_VERSION,
)
from services.platform_billing.partner_identity import (
    detect_identity_divergence,
    resolve_partner_contract_identity,
)


def _company(**kwargs):
    base = {
        "id": 1,
        "name": "Emmenez-moi Sàrl",
        "uid_ide": "CHE-273.048.653",
        "domicile_address_line1": "Route de Chevrens 145",
        "domicile_address_line2": None,
        "domicile_zip": "1247",
        "domicile_city": "Anières",
        "domicile_country": "CH",
        "legal_form": LegalForm.SARL.value,
        "signatory_name": "Khalid ALAOUI",
        "signatory_title": "Gérant",
        "billing_email": None,
        "contact_email": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _profile(**kwargs):
    base = {
        "id": 42,
        "legal_name": "Emmenez-moi Sàrl",
        "uid_ide": "CHE-273.048.653",
        "street_name": "Route de Chevrens",
        "building_number": "145",
        "postal_code": "1247",
        "city": "Anières",
        "country_code": "CH",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_resolve_partner_uses_profile_block_not_hybrid():
    company = _company(
        domicile_address_line1="Autre rue",
        domicile_zip="1200",
        domicile_city="Genève",
        name="Autre nom",
    )
    profile = _profile()
    identity = resolve_partner_contract_identity(company, profile)
    assert identity.identity_source == "company_billing_profile"
    assert identity.legal_name == "Emmenez-moi Sàrl"
    assert identity.street_name == "Route de Chevrens"
    assert identity.postal_code == "1247"
    assert identity.city == "Anières"
    assert identity.signatory_name == "Khalid ALAOUI"
    assert identity.is_complete(require_uid_ide=True)


def test_resolve_partner_falls_back_to_company_block():
    company = _company()
    identity = resolve_partner_contract_identity(company, None)
    assert identity.identity_source == "company"
    assert identity.street_name == "Route de Chevrens 145"
    assert identity.is_complete(require_uid_ide=True)


def test_resolve_incomplete_without_signatory():
    company = _company(signatory_name=None)
    identity = resolve_partner_contract_identity(company, None)
    assert not identity.is_complete(require_uid_ide=True)
    assert "signataire" in identity.missing_fields(require_uid_ide=True)


def test_operator_uid_ide_optional():
    from services.platform_billing.partner_identity import (
        resolve_operator_contract_identity,
    )

    creditor = SimpleNamespace(
        id=1,
        legal_name="Drin Jasiqi",
        uid_ide=None,
        street_name="Avenue Ernest-Pictet",
        building_number="9",
        postal_code="1203",
        city="Genève",
        country_code="CH",
        legal_form=LegalForm.SOLE_PROPRIETORSHIP.value,
        signatory_name="Drin Jasiqi",
        signatory_title="Exploitant",
    )
    op = resolve_operator_contract_identity(creditor)
    assert op is not None
    assert op.is_complete(require_uid_ide=False)
    assert "uid_ide" not in op.missing_fields(require_uid_ide=False)


def test_operator_brand_name_replaced_by_signatory():
    """Si legal_name = enseigne LIRIE, utiliser le signataire (personne physique)."""
    from services.platform_billing.partner_identity import (
        resolve_operator_contract_identity,
    )

    creditor = SimpleNamespace(
        id=1,
        legal_name="LIRIE",
        uid_ide=None,
        street_name="Avenue Ernest- Pictet",
        building_number="9",
        postal_code="1203",
        city="Genève",
        country_code="CH",
        legal_form=LegalForm.SOLE_PROPRIETORSHIP.value,
        signatory_name="Drin Jasiqi",
        signatory_title="Exploitant",
    )
    op = resolve_operator_contract_identity(creditor)
    assert op is not None
    assert op.legal_name == "Drin Jasiqi"
    assert op.contractual_email == "info@lirie.ch"


def test_divergence_warnings():
    company = _company(name="Autre", domicile_zip="9999")
    profile = _profile()
    warnings = detect_identity_divergence(company, profile)
    assert "raison_sociale" in warnings
    assert "npa" in warnings


def _sample_parties_commercial():
    parties = {
        "operator": {
            "legal_name": "LIRIE",
            "legal_form": LegalForm.SOLE_PROPRIETORSHIP.value,
            "legal_form_label": "Indépendant",
            "street_name": "Avenue Ernest- Pictet",
            "building_number": "9",
            "postal_code": "1203",
            "city": "Genève",
            "country_code": "CH",
            "uid_ide": None,
            "signatory_name": "Drin Jasiqi",
            "signatory_title": "Exploitant",
            "contractual_email": "info@lirie.ch",
        },
        "partner": {
            "legal_name": "Emmenez-moi",
            "legal_form": LegalForm.SARL.value,
            "legal_form_label": "Sàrl",
            "street_name": "Route de Chevrens",
            "building_number": "145",
            "postal_code": "1247",
            "city": "Anières",
            "country_code": "CH",
            "uid_ide": "CHE-273.048.653",
            "signatory_name": "Khalid ALAOUI",
            "signatory_title": "Associée Gérant, avec signature individuelle",
            "contractual_email": "contact@emmenez-moi.ch",
        },
        "signatory_authority_verification": {"signature_mode": "individual"},
    }
    commercial = {
        "subscription_pricing_mode": "free",
        "free_license_max_months": 12,
        "lirie_commission_enabled": True,
        "own_portfolio_billing_enabled": True,
        "commission_rate": "0.100000",
        "payment_terms_days": 15,
        "statement_dispute_days": 5,
        "support_hourly_rate_default": "45",
        "penalty": {
            "multiplier": 2,
            "minimum": 1000,
            "currency": "CHF",
            "calculation_version": "lirie-penalty-v1",
        },
    }
    return parties, commercial


def test_particular_pack_pdf_three_pages_and_sha_in_content():
    from io import BytesIO

    from pypdf import PdfReader

    from services.platform_billing.partner_agreement_canonical import (
        ensure_canonical_documents,
    )
    from services.platform_billing.partner_agreement_particular_content import (
        build_particular_agreement_content,
    )
    from services.platform_billing.partner_agreement_particular_docx import (
        build_particular_docx_bytes,
    )
    from services.platform_billing.partner_agreement_particular_pdf import (
        build_particular_pdf_bytes,
        count_pdf_pages,
    )

    parties, commercial = _sample_parties_commercial()
    canon = ensure_canonical_documents()
    content = build_particular_agreement_content(
        reference="LIRIE/PART/2026-08/001",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_sha256=canon["dpa"].sha256,
    )
    pdf = build_particular_pdf_bytes(content)
    docx = build_particular_docx_bytes(content)
    assert count_pdf_pages(pdf) == 3
    assert pdf.startswith(b"%PDF")
    assert docx[:2] == b"PK"
    text = "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(pdf)).pages)
    # Normalise les coupures de ligne PDF pour les assertions de phrases.
    flat = " ".join(text.split())
    assert "LIRIE/PART/2026-08/001" in flat
    assert PARTICULAR_VERSION in flat
    assert PACK_SCHEMA_VERSION == "lirie-partner-pack-v1"
    assert GENERAL_TERMS_VERSION in flat
    assert DPA_VERSION in flat
    assert canon["general_terms"].sha256 not in flat
    assert canon["dpa"].sha256 not in flat
    assert "acceptation définitive" in flat
    assert "1er août 2026" in flat
    assert "Associée Gérant" in flat
    assert "10 %" in flat or "10%" in flat
    assert "s'applique également aux traitements" in flat
    assert "LIRIE ne répond que des dommages directs" in flat
    assert "chaque Partie ne répond" not in flat
    assert "En cas de contradiction" in flat
    assert "demande expresse documentée" in flat
    assert "même valeur juridique" in flat
    assert "pouvoir nécessaire" in flat
    assert "Formation et traitement des courses" in flat
    assert "bordereau de remise" in flat
    assert "non attribué" not in flat
    assert "BROUILLON" not in flat.upper()
    # Les empreintes hexadécimales restent hors du contrat (bordereau / snapshot).
    assert canon["general_terms"].sha256 not in flat
    assert canon["dpa"].sha256 not in flat
    assert "main_contract_pdf_sha256" not in flat
    for blob in (
        content.reference,
        content.particular_version,
        content.acceptance_clause[:40],
    ):
        assert " ".join(blob.split()) in flat


def test_canonical_sha_stable_across_partners():
    from services.platform_billing.partner_agreement_canonical import (
        ensure_canonical_documents,
    )

    a = ensure_canonical_documents()
    b = ensure_canonical_documents()
    assert a["general_terms"].sha256 == b["general_terms"].sha256
    assert a["dpa"].sha256 == b["dpa"].sha256


def test_delivery_zip_deterministic_and_manifest_without_zip_sha():
    from services.platform_billing.partner_agreement_canonical import (
        ensure_canonical_documents,
    )
    from services.platform_billing.partner_agreement_manifest_pdf import (
        build_delivery_manifest_pdf_bytes,
    )
    from services.platform_billing.partner_agreement_package import (
        build_delivery_zip_bytes,
        sha256_bytes,
    )
    from services.platform_billing.partner_agreement_particular_content import (
        build_particular_agreement_content,
    )
    from services.platform_billing.partner_agreement_particular_pdf import (
        build_particular_pdf_bytes,
    )

    parties, commercial = _sample_parties_commercial()
    canon = ensure_canonical_documents()
    content = build_particular_agreement_content(
        reference="LIRIE/PART/2026-08/003",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_sha256=canon["dpa"].sha256,
    )
    particular = build_particular_pdf_bytes(content)
    manifest = build_delivery_manifest_pdf_bytes(
        reference="LIRIE/PART/2026-08/003",
        partner_name="Emmenez-moi Sàrl",
        finalized_at_fr="04.08.2026 22:00 CEST",
        particular_version=PARTICULAR_VERSION,
        particular_sha256=sha256_bytes(particular),
        general_terms_version=canon["general_terms"].version,
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_version=canon["dpa"].version,
        dpa_sha256=canon["dpa"].sha256,
        retention_policy_version="lirie-retention-v1",
        subprocessors_version="lirie-subprocessors-v2",
    )
    from io import BytesIO

    from pypdf import PdfReader

    manifest_text = "\n".join(
        (p.extract_text() or "") for p in PdfReader(BytesIO(manifest)).pages
    )
    assert "SHA-256" in manifest_text
    assert "ne contient ni son propre SHA-256 ni le SHA-256 du fichier ZIP" in (
        manifest_text
    )
    zip1 = build_delivery_zip_bytes(
        reference="LIRIE/PART/2026-08/003",
        manifest_pdf=manifest,
        particular_pdf=particular,
        general_terms_pdf=canon["general_terms"].pdf_bytes,
        dpa_pdf=canon["dpa"].pdf_bytes,
        general_terms_version=canon["general_terms"].version,
        dpa_version=canon["dpa"].version,
    )
    zip2 = build_delivery_zip_bytes(
        reference="LIRIE/PART/2026-08/003",
        manifest_pdf=manifest,
        particular_pdf=particular,
        general_terms_pdf=canon["general_terms"].pdf_bytes,
        dpa_pdf=canon["dpa"].pdf_bytes,
        general_terms_version=canon["general_terms"].version,
        dpa_version=canon["dpa"].version,
    )
    assert sha256_bytes(zip1) == sha256_bytes(zip2)
    assert sha256_bytes(zip1) not in manifest_text


def test_preview_watermark_is_derivative():
    from services.platform_billing.partner_agreement_canonical import (
        ensure_canonical_documents,
    )
    from services.platform_billing.partner_agreement_particular_content import (
        build_particular_agreement_content,
    )
    from services.platform_billing.partner_agreement_particular_pdf import (
        build_particular_pdf_bytes,
    )
    from services.platform_billing.partner_agreement_preview import (
        WATERMARK_TEXT,
        apply_draft_watermark,
    )

    parties, commercial = _sample_parties_commercial()
    canon = ensure_canonical_documents()
    content = build_particular_agreement_content(
        reference="LIRIE/PART/2026-08/001",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_sha256=canon["dpa"].sha256,
    )
    official = build_particular_pdf_bytes(content)
    preview = apply_draft_watermark(official)
    assert preview != official
    assert official.startswith(b"%PDF")
    assert preview.startswith(b"%PDF")
    assert WATERMARK_TEXT


def test_mark_sent_requires_draft():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        mark_agreement_sent,
    )

    agr = MagicMock()
    agr.status = PartnerAgreementStatus.SENT.value
    with patch(
        "services.platform_billing.partner_agreement._lock_agreement",
        return_value=agr,
    ):
        with pytest.raises(PartnerAgreementError) as exc:
            mark_agreement_sent(1, user_id=1)
        assert exc.value.status_code == 409


def test_mark_sent_rejects_tampered_snapshot():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        _assert_generation_integrity,
        canonical_json_sha256,
    )

    commercial = {"billing_config_id": 1, "commission_rate": "0.100000"}
    parties = {"partner": {"legal_name": "X"}}
    agr = MagicMock()
    agr.commercial_snapshot = {**commercial, "tampered": True}
    agr.parties_snapshot = parties
    agr.generated_storage_key = "x.pdf"
    agr.generated_sha256 = "abc"
    agr.generated_content_type = "application/pdf"
    agr.generation_snapshot = {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "template_version": PACK_SCHEMA_VERSION,
        "parties_snapshot_sha256": canonical_json_sha256(parties),
        "commercial_snapshot_sha256": canonical_json_sha256(commercial),
    }
    with pytest.raises(PartnerAgreementError) as exc:
        _assert_generation_integrity(agr)
    assert exc.value.status_code == 409
    assert "commercial_snapshot" in exc.value.message


def test_validate_rc_collective_requires_co_signatory():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        validate_signatory_authority_verification,
    )

    with pytest.raises(PartnerAgreementError):
        validate_signatory_authority_verification(
            {
                "attested": True,
                "signature_mode": "collective",
                "signatory_name": "A",
                "signatory_function": "gérant",
            }
        )
    ok = validate_signatory_authority_verification(
        {
            "attested": True,
            "signature_mode": "individual",
            "signatory_name": "A",
            "signatory_function": "gérant",
            "company_uid": "CHE-1",
        }
    )
    assert ok["attested"] is True
    assert ok["signature_mode"] == "individual"


def test_mark_sent_rejects_old_template_version():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        _assert_generation_integrity,
        canonical_json_sha256,
    )

    commercial = {"billing_config_id": 1}
    parties = {"partner": {"legal_name": "X"}}
    agr = MagicMock()
    agr.commercial_snapshot = commercial
    agr.parties_snapshot = parties
    agr.generated_storage_key = "x.pdf"
    agr.generated_sha256 = "abc"
    agr.generated_content_type = "application/pdf"
    agr.generation_snapshot = {
        "template_version": "lirie-partner-v1.10",
        "parties_snapshot_sha256": canonical_json_sha256(parties),
        "commercial_snapshot_sha256": canonical_json_sha256(commercial),
    }
    with pytest.raises(PartnerAgreementError) as exc:
        _assert_generation_integrity(agr)
    assert exc.value.status_code == 409
    assert "pack" in exc.value.message.lower() or "ancien" in exc.value.message


def test_migrate_rejects_non_draft_and_already_pack():
    from services.platform_billing.partner_agreement import (
        PDF_MIME,
        PartnerAgreementError,
        migrate_draft_agreement_to_v120,
    )

    sent = MagicMock()
    sent.status = PartnerAgreementStatus.SENT.value
    with patch(
        "services.platform_billing.partner_agreement._lock_agreement",
        return_value=sent,
    ):
        with pytest.raises(PartnerAgreementError) as exc:
            migrate_draft_agreement_to_v120(
                1,
                user_id=1,
                signatory_authority_verification={"attested": True},
            )
        assert exc.value.status_code == 409

    draft = MagicMock()
    draft.status = PartnerAgreementStatus.DRAFT.value
    draft.generated_content_type = PDF_MIME
    draft.generation_snapshot = {"pack_schema_version": PACK_SCHEMA_VERSION}
    with patch(
        "services.platform_billing.partner_agreement._lock_agreement",
        return_value=draft,
    ):
        with pytest.raises(PartnerAgreementError) as exc:
            migrate_draft_agreement_to_v120(
                2,
                user_id=1,
                signatory_authority_verification={
                    "attested": True,
                    "signatory_name": "A",
                    "signatory_function": "gérant",
                    "signature_mode": "individual",
                },
            )
        assert "déjà au pack" in exc.value.message


def test_ensure_contract_pricing_grid_key_and_inactive():
    from decimal import Decimal

    from services.platform_billing.subscription_pricing_resolver import (
        PricingTierSnapshot,
        SubscriptionPricingResolution,
        ensure_contract_pricing_grid,
    )

    resolution = SubscriptionPricingResolution(
        source_kind="global_grid",
        pricing_mode="volume",
        requested_grid_id=None,
        resolved_grid_id=9,
        grid_key="default",
        grid_label="Default",
        currency="CHF",
        valid_from=None,
        valid_until=None,
        legacy_dispatch_mode=None,
        tiers=(
            PricingTierSnapshot(0, 200, Decimal("79.00"), "P1"),
            PricingTierSnapshot(201, 500, Decimal("149.00"), "P2"),
            PricingTierSnapshot(501, None, Decimal("249.00"), "P3"),
        ),
        validation_errors=(),
    )

    fake_grid = SimpleNamespace(
        id=None,
        grid_key=None,
        label=None,
        currency=None,
        is_active=True,
    )

    def _grid_ctor(**kwargs):
        fake_grid.grid_key = kwargs.get("grid_key")
        fake_grid.label = kwargs.get("label")
        fake_grid.currency = kwargs.get("currency")
        fake_grid.is_active = kwargs.get("is_active", True)
        fake_grid.id = 42
        return fake_grid

    query = MagicMock()
    query.filter_by.return_value.order_by.return_value.first.return_value = None
    tier_query = MagicMock()

    with (
        patch(
            "services.platform_billing.subscription_pricing_resolver."
            "PlatformSubscriptionPricingGrid",
            side_effect=_grid_ctor,
        ) as grid_cls,
        patch(
            "services.platform_billing.subscription_pricing_resolver."
            "PlatformSubscriptionPricingTier"
        ) as tier_cls,
        patch("services.platform_billing.subscription_pricing_resolver.db") as mock_db,
    ):
        grid_cls.query = query
        tier_cls.query = tier_query
        tier_cls.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        grid = ensure_contract_pricing_grid(
            billing_config_id=7,
            revision_number=3,
            reference="LIRIE/PART/2026-08/099",
            resolution=resolution,
        )
    assert grid.grid_key == "contract-cfg-7-r3"
    assert grid.is_active is False
    assert mock_db.session.add.call_count == 4  # grille + 3 paliers


def test_validate_tiers_detects_gap_and_overlap():
    from decimal import Decimal

    from services.platform_billing.subscription_pricing_resolver import (
        PricingTierSnapshot,
        _validate_tiers,
    )

    gap = _validate_tiers(
        [
            PricingTierSnapshot(0, 100, Decimal("10"), None),
            PricingTierSnapshot(200, 300, Decimal("20"), None),
        ]
    )
    assert any(e.startswith("trou_") for e in gap)

    overlap = _validate_tiers(
        [
            PricingTierSnapshot(0, 100, Decimal("10"), None),
            PricingTierSnapshot(50, 200, Decimal("20"), None),
        ]
    )
    assert any(e.startswith("chevauchement_") for e in overlap)


def test_upload_signed_rejects_non_pdf():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        upload_signed_pdf,
    )

    agr = MagicMock()
    agr.status = PartnerAgreementStatus.SENT.value
    with patch(
        "services.platform_billing.partner_agreement._lock_agreement",
        return_value=agr,
    ):
        with pytest.raises(PartnerAgreementError) as exc:
            upload_signed_pdf(
                1,
                content=b"PK\x03\x04not-a-pdf",
                original_filename="x.docx",
                agreement_signed_on=date(2026, 8, 12),
                user_id=1,
            )
        assert "PDF" in exc.value.message


def test_void_signed_forbidden():
    from services.platform_billing.partner_agreement import (
        PartnerAgreementError,
        void_agreement,
    )

    agr = MagicMock()
    agr.status = PartnerAgreementStatus.SIGNED.value
    with patch(
        "services.platform_billing.partner_agreement._lock_agreement",
        return_value=agr,
    ):
        with pytest.raises(PartnerAgreementError) as exc:
            void_agreement(1, reason="test", user_id=1)
        assert exc.value.status_code == 409
