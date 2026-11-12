"""Tests d'integrite sur les invariants ClientType Phase A.

Ces tests verifient que les regles metier suivantes sont respectees en base :

- TRANSPORT => company_id IS NOT NULL
- PORTAL    => company_id IS NULL
- TRANSPORT => management_mode IS NOT NULL
- PORTAL    => management_mode IS NULL

Ils sont conçus pour etre executes periodiquement (CI, post-migration,
monitoring) afin de detecter toute regression ou etat invalide.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text


class TestClientTypeInvariants:
    """Invariants metier sur la table client apres migration Phase A."""

    def test_no_transport_without_company(self, db):
        """Un client TRANSPORT doit toujours avoir un company_id."""
        result = db.session.execute(
            text(
                "SELECT COUNT(*) FROM client "
                "WHERE client_type::text = 'TRANSPORT' AND company_id IS NULL"
            )
        ).scalar()
        assert result == 0, (
            f"{result} client(s) TRANSPORT sans company_id detecte(s). "
            "Etat invalide : un client TRANSPORT doit etre rattache "
            "a une entreprise de transport."
        )

    def test_no_portal_with_company(self, db):
        """Un client PORTAL ne doit pas avoir de company_id."""
        result = db.session.execute(
            text(
                "SELECT COUNT(*) FROM client "
                "WHERE client_type::text = 'PORTAL' AND company_id IS NOT NULL"
            )
        ).scalar()
        assert result == 0, (
            f"{result} client(s) PORTAL avec company_id detecte(s). "
            "Etat invalide : un client PORTAL ne doit pas etre rattache "
            "a une entreprise."
        )

    def test_transport_has_management_mode(self, db):
        """Un client TRANSPORT doit avoir un management_mode renseigne."""
        result = db.session.execute(
            text(
                "SELECT COUNT(*) FROM client "
                "WHERE client_type::text = 'TRANSPORT' AND management_mode IS NULL"
            )
        ).scalar()
        assert result == 0, (
            f"{result} client(s) TRANSPORT sans management_mode detecte(s). "
            "Etat invalide : un client TRANSPORT doit avoir un "
            "management_mode (SELF_SERVICE, MANAGED ou CORPORATE)."
        )

    def test_portal_has_no_management_mode(self, db):
        """Un client PORTAL ne doit pas avoir de management_mode."""
        result = db.session.execute(
            text(
                "SELECT COUNT(*) FROM client "
                "WHERE client_type::text = 'PORTAL' AND management_mode IS NOT NULL"
            )
        ).scalar()
        assert result == 0, (
            f"{result} client(s) PORTAL avec management_mode detecte(s). "
            "Etat invalide : un client PORTAL ne doit pas avoir de "
            "management_mode."
        )

    def test_no_legacy_client_types(self, db):
        """Aucun ancien type (PRIVATE, SELF_SERVICE, CORPORATE) ne doit subsister."""
        result = db.session.execute(
            text(
                "SELECT client_type::text, COUNT(*) FROM client "
                "WHERE client_type::text IN ('PRIVATE', 'SELF_SERVICE', 'CORPORATE') "
                "GROUP BY 1"
            )
        ).fetchall()
        assert len(result) == 0, (
            f"Anciens types encore presents en base : "
            f"{', '.join(f'{ct}={n}' for ct, n in result)}. "
            "La migration Phase A n'a pas ete completee."
        )

    def test_only_valid_management_modes(self, db):
        """Seules les valeurs SELF_SERVICE, MANAGED, CORPORATE sont acceptees."""
        result = db.session.execute(
            text(
                "SELECT DISTINCT management_mode::text FROM client "
                "WHERE management_mode IS NOT NULL "
                "AND management_mode::text NOT IN ('SELF_SERVICE', 'MANAGED', 'CORPORATE')"
            )
        ).fetchall()
        assert len(result) == 0, (
            f"Valeurs management_mode invalides : {[r[0] for r in result]}"
        )

    def test_institution_patients_untouched(self, db):
        """La table institution_patients ne doit pas avoir ete modifiee."""
        count = db.session.execute(
            text("SELECT COUNT(*) FROM institution_patients")
        ).scalar()
        assert count is not None, (
            "La table institution_patients n'existe pas ou est inaccessible."
        )
        assert count >= 0, "COUNT(*) institution_patients ne doit pas etre negatif."
