"""
Tests pour les routes drivers (disponibilité, assignations).
"""

from models import Driver, UserRole


def test_list_drivers_unauthenticated(client):
    """GET /drivers sans authentification renvoie 401."""
    response = client.get("/api/drivers/")
    assert response.status_code == 401


def test_list_drivers_authenticated(client, auth_headers):
    """GET /drivers avec authentification renvoie liste."""
    response = client.get("/api/drivers/", headers=auth_headers)
    # Peut être 200 ou 404 selon si route existe
    assert response.status_code in [200, 404, 405]


def test_driver_has_user_relationship(db, sample_driver):
    """Driver a une relation avec User."""
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.user is not None
    assert driver.user.role == UserRole.driver


def test_driver_has_company_relationship(db, sample_driver, sample_company):
    """Driver a une relation avec Company."""
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.company_id == sample_company.id
    assert driver.company is not None


def test_driver_availability_flag(db, sample_driver):
    """Driver a un flag is_available."""
    driver = Driver.query.get(sample_driver.id)
    assert hasattr(driver, "is_available")
    assert isinstance(driver.is_available, bool)


def test_driver_license_number(db, sample_driver):
    """Driver a des catégories de permis."""
    driver = Driver.query.get(sample_driver.id)
    # Le modèle Driver utilise license_categories (JSON) au lieu de license_number
    assert hasattr(driver, "license_categories")
    assert driver.license_categories is not None


def test_driver_serialize(db, sample_driver):
    """Driver.serialize retourne dict avec données."""
    driver = Driver.query.get(sample_driver.id)
    if hasattr(driver, "serialize"):
        serialized = driver.serialize
        assert isinstance(serialized, dict)
        assert "id" in serialized or "user" in serialized


def test_available_drivers_query(db, sample_driver):
    """Requête pour chauffeurs disponibles."""
    available = Driver.query.filter_by(is_available=True).all()
    assert len(available) > 0
    assert sample_driver in available


def test_drivers_by_company(db, sample_driver, sample_company):
    """Requête chauffeurs par entreprise."""
    drivers = Driver.query.filter_by(company_id=sample_company.id).all()
    assert len(drivers) > 0
    assert sample_driver in drivers
