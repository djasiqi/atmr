"""Tests unitaires pour persisted_fixture().

Ce module teste le helper persisted_fixture() qui garantit la persistance
des objets dans la base de données pour les fixtures de tests.
"""

from __future__ import annotations

from models import Company, Driver
from tests.conftest import persisted_fixture
from tests.factories import CompanyFactory, DriverFactory


class TestPersistedFixture:
    """Tests pour persisted_fixture()."""

    def test_persisted_fixture_creates_and_commits_company(self, db):
        """Test que persisted_fixture crée et commit une Company."""
        # Créer une company via persisted_fixture
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Vérifier que la company a un ID (a été flushée)
        assert company.id is not None, (
            "Company doit avoir un ID après persisted_fixture"
        )

        # Vérifier que la company est visible dans la DB
        reloaded = db.session.query(Company).get(company.id)
        assert reloaded is not None, (
            "Company doit être visible dans la DB après persisted_fixture"
        )
        assert reloaded.id == company.id, "Company rechargée doit avoir le même ID"

    def test_persisted_fixture_with_reload_false(self, db):
        """Test persisted_fixture avec reload=False."""
        company = persisted_fixture(db, CompanyFactory(), Company, reload=False)

        # Vérifier que la company a un ID
        assert company.id is not None, "Company doit avoir un ID"

        # Vérifier que la company est dans la DB
        reloaded = db.session.query(Company).get(company.id)
        assert reloaded is not None, "Company doit être dans la DB"

    def test_persisted_fixture_with_reload_true(self, db):
        """Test persisted_fixture avec reload=True (par défaut)."""
        company = persisted_fixture(db, CompanyFactory(), Company, reload=True)

        # Vérifier que la company a un ID
        assert company.id is not None, "Company doit avoir un ID"

        # Vérifier que la company est dans la DB
        reloaded = db.session.query(Company).get(company.id)
        assert reloaded is not None, "Company doit être dans la DB"
        assert reloaded.id == company.id, "Company rechargée doit avoir le même ID"

    def test_persisted_fixture_with_assert_exists_true(self, db):
        """Test persisted_fixture avec assert_exists=True (par défaut)."""
        company = persisted_fixture(db, CompanyFactory(), Company, assert_exists=True)

        # Vérifier que la company existe
        assert company.id is not None, "Company doit avoir un ID"
        reloaded = db.session.query(Company).get(company.id)
        assert reloaded is not None, "Company doit exister dans la DB"

    def test_persisted_fixture_with_complex_model(self, db):
        """Test persisted_fixture avec un modèle complexe (Driver avec relations)."""
        # Créer d'abord une company
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Créer un driver avec la company
        driver = persisted_fixture(
            db, DriverFactory(company=company, is_active=True), Driver
        )

        # Vérifier que le driver a un ID
        assert driver.id is not None, "Driver doit avoir un ID"

        # Vérifier que le driver est dans la DB
        reloaded = db.session.query(Driver).get(driver.id)
        assert reloaded is not None, "Driver doit être dans la DB"
        assert reloaded.company_id == company.id, "Driver doit être lié à la company"

    def test_persisted_fixture_uses_db_session_correctly(self, db):
        """Test que persisted_fixture utilise db.session correctement (Flask-SQLAlchemy)."""
        # Vérifier que db est bien une instance Flask-SQLAlchemy
        assert hasattr(db, "session"), (
            "db doit être une instance Flask-SQLAlchemy avec .session"
        )

        # Créer une company
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Vérifier que la company est commitée (visible via db.session)
        reloaded = db.session.query(Company).get(company.id)
        assert reloaded is not None, "Company doit être visible via db.session.query()"

    def test_persisted_fixture_multiple_objects(self, db):
        """Test que persisted_fixture fonctionne avec plusieurs objets successifs."""
        # Créer plusieurs companies
        company1 = persisted_fixture(db, CompanyFactory(), Company)
        company2 = persisted_fixture(db, CompanyFactory(), Company)
        company3 = persisted_fixture(db, CompanyFactory(), Company)

        # Vérifier que toutes ont des IDs différents
        assert company1.id != company2.id, "Companies doivent avoir des IDs différents"
        assert company2.id != company3.id, "Companies doivent avoir des IDs différents"
        assert company1.id != company3.id, "Companies doivent avoir des IDs différents"

        # Vérifier que toutes sont dans la DB
        assert db.session.query(Company).get(company1.id) is not None
        assert db.session.query(Company).get(company2.id) is not None
        assert db.session.query(Company).get(company3.id) is not None
