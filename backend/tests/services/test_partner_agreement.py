"""Tests accords partenaires Word (identité, DOCX, lifecycle)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.enums import LegalForm, PartnerAgreementStatus
from services.platform_billing.partner_agreement_docx import (
    TEMPLATE_VERSION,
    build_partner_agreement_docx_bytes,
)
from services.platform_billing.partner_identity import (
    detect_identity_divergence,
    resolve_partner_contract_identity,
)


def _company(**kwargs):
    base = dict(
        id=1,
        name="Emmenez-moi Sàrl",
        uid_ide="CHE-273.048.653",
        domicile_address_line1="Route de Chevrens 145",
        domicile_address_line2=None,
        domicile_zip="1247",
        domicile_city="Anières",
        domicile_country="CH",
        legal_form=LegalForm.SARL.value,
        signatory_name="Khalid ALAOUI",
        signatory_title="Gérant",
        billing_email=None,
        contact_email=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _profile(**kwargs):
    base = dict(
        id=42,
        legal_name="Emmenez-moi Sàrl",
        uid_ide="CHE-273.048.653",
        street_name="Route de Chevrens",
        building_number="145",
        postal_code="1247",
        city="Anières",
        country_code="CH",
    )
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


def test_docx_contains_reference_and_commission():
    import io
    import zipfile

    parties = {
        "operator": {
            # Enseigne stockée à tort — le DOCX doit utiliser le signataire
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
            # Sans suffixe Sàrl — le générateur le complète
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
            "signatory_title": "associé-gérant, avec signature individuelle",
            "contractual_email": "contact@emmenez-moi.ch",
        },
    }
    commercial = {
        "subscription_pricing_mode": "free",
        "free_license_max_months": 60,
        "lirie_commission_enabled": True,
        "own_portfolio_billing_enabled": True,
        "commission_rate": "0.100000",
        "commission_cancellation_policy": "exclude",
        "payment_terms_days": 30,
        "statement_dispute_days": 10,
    }
    raw = build_partner_agreement_docx_bytes(
        reference="LIRIE/PART/2026-08/001",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
    )
    assert raw[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    assert "LIRIE/PART/2026-08/001" in xml
    assert TEMPLATE_VERSION in xml
    assert "10 %" in xml or "10%" in xml
    assert "{{" not in xml
    assert "OWN_PORTFOLIO" in xml
    assert "LIRIE_MARKETPLACE" in xml
    assert "Drin Jasiqi, exerçant en qualité d'indépendant sous l'enseigne LIRIE" in xml
    assert "Statut : indépendant" in xml
    assert "Représenté(e) par : Drin Jasiqi" not in xml
    assert "Ernest-Pictet" in xml
    assert "Ernest- Pictet" not in xml
    assert "Emmenez-moi Sàrl" in xml
    assert "Ci-après désigné : « le Partenaire »" in xml
    assert "non attribué" in xml
    assert "notification de la mise à disposition du relevé dans LIRIE" in xml
    assert "date d'émission de la facture" in xml
    assert "cinq (5) jours ouvrables suivant son exécution" in xml
    assert "n'altère pas son origine commerciale" in xml
    assert "renouvelle tacitement pour une durée indéterminée" in xml
    assert "Aucun abonnement ne sera appliqué automatiquement" in xml
    assert "données anonymisées ou agrégées" in xml
    assert "se substitue à l'Exploitant pour les obligations futures" in xml
    assert "info@lirie.ch" in xml
    assert "contact@emmenez-moi.ch" in xml
    assert "CHF 10'000" in xml
    assert "mise en demeure" in xml
    assert "Lieu :" in xml
    assert "courrier recommandé" in xml
    assert "objection motivée" in xml
    assert "ARTICLE 6 BIS" in xml
    assert "DÉFAUT DE PAIEMENT" in xml
    assert "factures échues et impayées" in xml
    assert "frais officiels de poursuite" in xml
    assert "Aucun frais forfaitaire de recouvrement" in xml
    assert "échéancier écrit" in xml
    # Snapshot commercial du test n'inclut pas encore les champs dunning → défauts
    commercial_with_dunning = {
        **commercial,
        "automated_dunning_enabled": True,
        "reminder_delay_days_after_due": 0,
        "reminder_grace_days": 10,
        "full_suspend_days_after_due": 30,
        "full_suspend_overdue_invoice_count": 2,
        "termination_notice_days": 10,
        "partial_block_marketplace_offers": True,
        "partial_block_marketplace_acceptance": True,
        "partial_block_billable_support": True,
        "partial_block_billable_configuration": True,
    }
    raw2 = build_partner_agreement_docx_bytes(
        reference="LIRIE/PART/2026-08/001",
        parties=parties,
        commercial=commercial_with_dunning,
        agreement_effective_from="2026-08-01",
    )
    with zipfile.ZipFile(io.BytesIO(raw2)) as zf:
        xml2 = zf.read("word/document.xml").decode("utf-8")
    assert "lirie-partner-v1.4" in xml2
    assert "Courses Marketplace LIRIE" in xml2


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
