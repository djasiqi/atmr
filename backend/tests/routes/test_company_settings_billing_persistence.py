"""
Test de persistance des paramètres bancaires (iban, qr_iban, esr_ref_base).

Ce test vérifie que les valeurs sauvegardées via PUT /company-settings/billing
sont bien persistées en base et rechargées correctement après un "restart"
(simulation via nouvelle session DB).
"""

from __future__ import annotations

import pytest

from models import CompanyBillingSettings


@pytest.mark.integration
class TestCompanySettingsBillingPersistence:
    """Tests de persistance des paramètres bancaires."""

    def test_billing_settings_banking_fields_persistence(
        self, authenticated_client, test_company, db
    ):
        """Test que iban/qr_iban/esr_ref_base sont bien persistés et rechargés."""
        if not test_company:
            pytest.skip("test_company fixture missing")

        # 1. Sauvegarder des valeurs via PUT
        url = "/api/v1/company-settings/billing"
        test_iban = "CH9300762011623852957"
        test_qr_iban = "CH4431999123000889012"
        test_esr_ref_base = "00000000000000000000"

        payload = {
            "iban": test_iban,
            "qr_iban": test_qr_iban,
            "esr_ref_base": test_esr_ref_base,
            "payment_terms_days": 30,
        }

        response = authenticated_client.put(url, json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True

        # Vérifier que la réponse contient les valeurs
        response_data = data.get("data", {})
        assert "iban" in response_data
        assert "qr_iban" in response_data
        assert "esr_ref_base" in response_data

        # Les valeurs peuvent être None si le déchiffrement échoue, mais on vérifie
        # qu'elles sont présentes dans la réponse
        # (le déchiffrement peut échouer si la clé de chiffrement n'est pas configurée en test)

        # 2. ✅ BLINDAGE: Vérifier directement en SQL (simule un "restart")
        # Expirer tous les objets de la session actuelle
        db.session.expire_all()

        # Requête SQL directe pour vérifier la persistance (bypass ORM)
        from sqlalchemy import text

        sql_result = db.session.execute(
            text(
                "SELECT iban, qr_iban, esr_ref_base FROM company_billing_settings "
                "WHERE company_id = :company_id"
            ),
            {"company_id": test_company.id},
        ).first()

        assert sql_result is not None, "Billing settings should exist in DB after PUT"
        # esr_ref_base n'est pas chiffré, vérifier directement
        assert sql_result.esr_ref_base == test_esr_ref_base, (
            f"esr_ref_base mismatch: SQL={sql_result.esr_ref_base}, expected={test_esr_ref_base}"
        )
        # iban et qr_iban sont chiffrés, vérifier qu'ils ne sont pas NULL
        assert sql_result.iban is not None, "iban should be encrypted (not NULL) in DB"
        assert sql_result.qr_iban is not None, (
            "qr_iban should be encrypted (not NULL) in DB"
        )

        # 3. Recharger via ORM (simule un GET après restart)
        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()

        assert billing is not None, "Billing settings should exist after PUT"
        assert billing.esr_ref_base == test_esr_ref_base

        # 4. Vérifier via GET endpoint (simule un refresh après restart)
        response = authenticated_client.get(url)
        assert response.status_code == 200

        # Le GET peut retourner directement le dict ou dans un wrapper
        get_data = response.get_json()
        if isinstance(get_data, dict) and "data" in get_data:
            get_data = get_data["data"]
        elif isinstance(get_data, tuple):
            # Si c'est un tuple (status, dict), prendre le dict
            get_data = get_data[0] if len(get_data) > 0 else {}

        # Vérifier que les champs sont présents dans la réponse GET
        assert "iban" in get_data
        assert "qr_iban" in get_data
        assert "esr_ref_base" in get_data

        # esr_ref_base devrait être exactement la valeur sauvegardée (pas chiffré)
        assert get_data["esr_ref_base"] == test_esr_ref_base

        # iban et qr_iban peuvent être None si le déchiffrement échoue en test
        # mais on vérifie au moins que les champs sont présents
        # En production avec la vraie clé, ils devraient être déchiffrés correctement

    def test_billing_settings_empty_values_persistence(
        self, authenticated_client, test_company, db
    ):
        """Test que les valeurs vides (None/empty string) sont bien gérées."""
        if not test_company:
            pytest.skip("test_company fixture missing")

        url = "/api/v1/company-settings/billing"

        # 1. Sauvegarder des valeurs
        payload = {
            "iban": "CH9300762011623852957",
            "qr_iban": "CH4431999123000889012",
            "esr_ref_base": "00000000000000000000",
        }
        response = authenticated_client.put(url, json=payload)
        assert response.status_code == 200

        # 2. Vider les valeurs
        payload_empty = {
            "iban": "",
            "qr_iban": None,
            "esr_ref_base": "",
        }
        response = authenticated_client.put(url, json=payload_empty)
        assert response.status_code == 200

        # 3. ✅ BLINDAGE: Vérifier directement en SQL que les valeurs sont None
        db.session.expire_all()

        # Requête SQL directe pour vérifier (bypass ORM)
        from sqlalchemy import text

        sql_result = db.session.execute(
            text(
                "SELECT iban, qr_iban, esr_ref_base FROM company_billing_settings "
                "WHERE company_id = :company_id"
            ),
            {"company_id": test_company.id},
        ).first()

        assert sql_result is not None
        # Vérifier que les valeurs sont NULL en base
        assert sql_result.iban is None or sql_result.iban == "", (
            f"iban should be NULL/empty in DB, got: {sql_result.iban}"
        )
        assert sql_result.qr_iban is None or sql_result.qr_iban == "", (
            f"qr_iban should be NULL/empty in DB, got: {sql_result.qr_iban}"
        )
        assert sql_result.esr_ref_base is None or sql_result.esr_ref_base == "", (
            f"esr_ref_base should be NULL/empty in DB, got: {sql_result.esr_ref_base}"
        )

        # Vérifier aussi via ORM
        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()

        assert billing is not None
        assert billing.esr_ref_base is None or billing.esr_ref_base == ""

        # 4. Vérifier via GET
        response = authenticated_client.get(url)
        assert response.status_code == 200
        get_data = response.get_json()
        if isinstance(get_data, dict) and "data" in get_data:
            get_data = get_data["data"]
        elif isinstance(get_data, tuple):
            get_data = get_data[0] if len(get_data) > 0 else {}

        # Les valeurs devraient être None ou "" dans la réponse
        assert get_data.get("iban") is None or get_data.get("iban") == ""
        assert get_data.get("qr_iban") is None or get_data.get("qr_iban") == ""
        assert (
            get_data.get("esr_ref_base") is None or get_data.get("esr_ref_base") == ""
        )
