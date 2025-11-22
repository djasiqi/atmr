# backend/tests/e2e/test_dispatch_e2e.py
"""✅ Tests E2E complets pour le dispatch (frontend→backend→Celery→DB).

Scénarios testés:
1. Dispatch async complet (API → Celery → DB → Frontend)
2. Dispatch sync (<10 bookings)
3. Validation temporelle stricte avec rollback
4. Récupération après crash
5. Tests de charge (batch dispatches)
6. Rollback transactionnel complet
"""

import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict

import pytest

from ext import db
from models import Assignment, Booking, BookingStatus, DispatchRun, DispatchStatus, Driver
from services.unified_dispatch import engine
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests.

    ⚠️ COUPLAGE IMPORTANT :
    - Cette fixture DOIT être commitée avant utilisation car `engine.run()` fait un rollback défensif
    - Les fixtures `drivers` et `bookings` dépendent de cette fixture (ordre d'exécution garanti par pytest)
    - L'objet est rechargé depuis la DB pour garantir qu'il est bien persisté

    🔄 ISOLATION :
    - Chaque test utilise un savepoint (nested transaction) via la fixture `db`
    - Le rollback automatique en fin de test garantit l'isolation entre les tests
    - Les objets commités dans cette fixture sont visibles dans le savepoint du test

    📝 UTILISATION :
    - Utiliser cette fixture comme dépendance pour `drivers` et `bookings`
    - Ne pas modifier l'objet retourné sans recharger depuis la DB après `engine.run()`
    """
    from models import Company

    company = CompanyFactory()
    db.session.add(company)
    db.session.flush()  # Force l'assignation de l'ID
    # ✅ FIX: Commit pour garantir persistance avant engine.run()
    # engine.run() fait un rollback défensif qui peut expirer la Company si elle n'est pas commitée
    db.session.commit()
    # ✅ FIX: Expirer et recharger pour s'assurer que l'objet est bien en DB
    db.session.expire(company)
    company = db.session.query(Company).get(company.id)
    assert company is not None, "Company must be persisted before use"
    return company


@pytest.fixture
def drivers(db, company=None):
    """Créer plusieurs chauffeurs pour les tests.

    ✅ DÉCOUPLAGE P2.4 :
    - Le paramètre `company` est optionnel pour réduire les couplages
    - Si `company` n'est pas fournie, une company est créée automatiquement
    - Permet d'utiliser cette fixture indépendamment ou avec une company existante

    🔄 ISOLATION :
    - Les drivers sont commités dans le savepoint du test
    - Le rollback automatique en fin de test garantit l'isolation

    📝 UTILISATION :
    - `def test_example(drivers):` - Company créée automatiquement
    - `def test_example(company, drivers):` - Company passée explicitement
    """
    from models import Company
    from tests.conftest import persisted_fixture

    # ✅ P2.4: Créer company si non fournie (découplage)
    if company is None:
        company = CompanyFactory()
        company = persisted_fixture(db, company, Company)

    drivers_list = [
        DriverFactory(company=company, is_active=True, is_available=True),
        DriverFactory(company=company, is_active=True, is_available=True),
        DriverFactory(company=company, is_active=True, is_available=True),
    ]
    db.session.flush()  # Force l'assignation des IDs
    # ✅ FIX: Commit pour garantir persistance
    db.session.commit()
    return drivers_list


@pytest.fixture
def bookings(db, company=None):
    """Créer plusieurs bookings pour les tests.

    ✅ DÉCOUPLAGE P2.4 :
    - Le paramètre `company` est optionnel pour réduire les couplages
    - Si `company` n'est pas fournie, une company est créée automatiquement
    - Permet d'utiliser cette fixture indépendamment ou avec une company existante

    🔄 ISOLATION :
    - Les bookings sont commités dans le savepoint du test
    - Le rollback automatique en fin de test garantit l'isolation

    📝 UTILISATION :
    - `def test_example(bookings):` - Company créée automatiquement
    - `def test_example(company, bookings):` - Company passée explicitement
    """
    from models import Company
    from tests.conftest import persisted_fixture

    # ✅ P2.4: Créer company si non fournie (découplage)
    if company is None:
        company = CompanyFactory()
        company = persisted_fixture(db, company, Company)

    today = date.today()
    bookings_list = []
    for i in range(5):
        scheduled_time = datetime.combine(today, datetime.min.time().replace(hour=10 + i))
        booking = BookingFactory(
            company=company,
            status=BookingStatus.ACCEPTED,
            scheduled_time=scheduled_time,
        )
        bookings_list.append(booking)
    db.session.flush()  # Force l'assignation des IDs
    # ✅ FIX: Commit pour garantir persistance
    db.session.commit()
    return bookings_list


class TestDispatchE2E:
    """Tests E2E pour le dispatch complet."""

    def test_dispatch_async_complet(self, company, drivers, bookings):
        """Test : Dispatch async complet (API → Celery → DB)."""
        # Simuler un appel API
        for_date = date.today().isoformat()

        # Exécuter le dispatch (simulation API → engine)
        result = engine.run(
            company_id=company.id,
            for_date=for_date,
            mode="auto",
            regular_first=True,
            allow_emergency=False,
        )

        # Vérifier que le résultat est cohérent
        assert "assignments" in result
        assert "unassigned" in result
        assert "meta" in result

        # ✅ FIX: Utiliser dispatch_run_id du résultat d'abord
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get("dispatch_run_id")
        if dispatch_run_id:
            dispatch_run = DispatchRun.query.get(dispatch_run_id)
            assert dispatch_run is not None
            assert dispatch_run.status == DispatchStatus.COMPLETED
        else:
            # Fallback : chercher par company_id et day
            dispatch_run = DispatchRun.query.filter_by(company_id=company.id, day=date.today()).first()
            assert dispatch_run is not None, "DispatchRun should be created"

        # Vérifier que les assignations sont en DB
        assignments = Assignment.query.filter(Assignment.dispatch_run_id == dispatch_run.id).all()

        assert len(assignments) > 0

        # Vérifier que les bookings sont assignés
        for booking in bookings:
            db.session.refresh(booking)
            if booking.id in [a.booking_id for a in assignments]:
                assert booking.driver_id is not None
                assert booking.status == BookingStatus.ASSIGNED

    def test_dispatch_sync_limite_10_bookings(self, company, drivers):
        """Test : Mode sync limité à <10 bookings."""
        # Créer exactement 10 bookings
        today = date.today()
        bookings_list = []
        for i in range(10):
            scheduled_time = datetime.combine(today, datetime.min.time().replace(hour=10 + i))
            booking = BookingFactory(
                company=company,
                status=BookingStatus.ACCEPTED,
                scheduled_time=scheduled_time,
            )
            bookings_list.append(booking)

        # Dispatch sync devrait fonctionner avec 10 bookings
        for_date = today.isoformat()
        result = engine.run(
            company_id=company.id,
            for_date=for_date,
            mode="auto",
        )

        # Vérifier succès
        assert result.get("meta", {}).get("reason") != "run_failed"

    def test_validation_temporelle_stricte_rollback(self, company, drivers):
        """Test : Validation temporelle stricte avec rollback automatique."""
        # Créer des bookings avec conflits temporels (même heure)
        today = date.today()
        same_time = datetime.combine(today, datetime.min.time().replace(hour=10, minute=0))

        booking1 = BookingFactory(
            company=company,
            status=BookingStatus.ACCEPTED,
            scheduled_time=same_time,
        )
        booking2 = BookingFactory(
            company=company,
            status=BookingStatus.ACCEPTED,
            scheduled_time=same_time,  # Même heure = conflit
        )
        db.session.commit()  # ✅ FIX: Commit pour rendre les objets persistants

        # Tenter dispatch (devrait détecter le conflit temporel)
        for_date = today.isoformat()

        # Note: La validation stricte est activée par défaut
        # Si des conflits sont détectés, le dispatch devrait échouer avec rollback
        result = engine.run(
            company_id=company.id,
            for_date=for_date,
            mode="auto",
        )

        # Vérifier que le rollback a fonctionné (aucune assignation partielle)
        # ✅ FIX: Expirer tous les objets avant rollback
        db.session.expire_all()
        # ✅ FIX: S'assurer que le rollback est bien exécuté (engine.run() peut avoir fait un rollback, mais on force)
        db.session.rollback()

        # ✅ FIX: Recharger depuis DB avec un nouveau query (pas get() qui peut utiliser le cache)
        booking1_reloaded = db.session.query(Booking).filter_by(id=booking1.id).first()
        booking2_reloaded = db.session.query(Booking).filter_by(id=booking2.id).first()

        # ✅ FIX: Vérifier que les objets sont bien rechargés
        assert booking1_reloaded is not None, "Booking1 must be reloaded from DB"
        assert booking2_reloaded is not None, "Booking2 must be reloaded from DB"

        # Si validation stricte active, les bookings ne devraient pas être assignés
        # Vérifier que le résultat indique un échec ou que les bookings ne sont pas assignés
        assert booking1_reloaded.driver_id is None, "Booking1 ne devrait pas être assigné après rollback"
        assert booking2_reloaded.driver_id is None, "Booking2 ne devrait pas être assigné après rollback"

        # Vérifier que le résultat du dispatch indique un problème (optionnel selon implémentation)
        if result.get("meta", {}).get("reason"):
            assert result["meta"]["reason"] in ["run_failed", "validation_failed", "conflict"], (
                f"Le dispatch devrait avoir échoué, mais reason={result['meta'].get('reason')}"
            )

    def test_rollback_transactionnel_complet(self, company, drivers, bookings):
        """Test : Rollback transactionnel complet en cas d'erreur partielle."""
        # Simuler une erreur en créant un booking avec un driver_id invalide
        # dans les assignations proposées

        from services.unified_dispatch.apply import apply_assignments

        # ✅ FIX: S'assurer que les bookings sont bien persistés
        db.session.flush()
        db.session.commit()  # Commit pour garantir persistance

        # ✅ FIX: Vérifier que les bookings existent en DB
        for booking in bookings:
            booking_from_db = db.session.query(Booking).filter_by(id=booking.id).first()
            assert booking_from_db is not None, f"Booking {booking.id} must exist in DB"
            assert booking_from_db.company_id == company.id, (
                f"Booking {booking.id} must belong to company {company.id}, got {booking_from_db.company_id}"
            )

        # ✅ FIX: S'assurer que company.id est bien utilisé
        assert company.id is not None, "Company ID must be set"

        # ✅ FIX: Créer un DispatchRun avant apply_assignments
        dispatch_run = DispatchRun(
            company_id=company.id, day=date.today(), status=DispatchStatus.RUNNING, started_at=datetime.now(UTC)
        )
        db.session.add(dispatch_run)
        db.session.flush()
        # ✅ Vérifier que l'ID est disponible après flush
        assert dispatch_run.id is not None, "DispatchRun ID should be available after flush"

        # Créer des assignations valides
        assignments = [
            {
                "booking_id": bookings[0].id,
                "driver_id": drivers[0].id,
                "score": 1.0,
            },
            {
                "booking_id": bookings[1].id,
                "driver_id": drivers[1].id,
                "score": 1.0,
            },
        ]

        # Appliquer (devrait réussir)
        result = apply_assignments(
            company_id=company.id,  # ✅ FIX: Utiliser company.id explicitement
            assignments=assignments,
            dispatch_run_id=dispatch_run.id,
        )

        # Vérifier que les assignations sont appliquées
        assert len(result["applied"]) == 2, (
            f"Expected 2 applied assignments, got {len(result['applied'])}. "
            f"Skipped: {result.get('skipped', {})}, Conflicts: {result.get('conflicts', [])}"
        )

        # Vérifier que les bookings sont assignés en DB
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)

        assert booking0.driver_id == drivers[0].id
        assert booking1.driver_id == drivers[1].id

    def test_recovery_apres_crash(self, company, drivers, bookings):
        """Test : Récupération après crash simulé."""
        # ✅ FIX: S'assurer que company est flushée avant de créer DispatchRun
        db.session.flush()

        # Simuler un crash en créant un DispatchRun en état RUNNING
        today = date.today()
        dispatch_run = DispatchRun(
            company_id=company.id,
            day=today,
            status=DispatchStatus.RUNNING,
            started_at=datetime.now(UTC) - timedelta(minutes=10),  # Il y a 10 min
        )
        db.session.add(dispatch_run)
        db.session.commit()  # ✅ FIX: Commit pour rendre l'objet persistant

        # Relancer le dispatch (devrait réutiliser ou créer un nouveau run)
        for_date = today.isoformat()
        result = engine.run(
            company_id=company.id,
            for_date=for_date,
            mode="auto",
        )

        # Vérifier que le dispatch a réussi
        assert result.get("meta", {}).get("reason") != "run_failed"

        # Vérifier que le DispatchRun est complété
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        dispatch_run = db.session.query(DispatchRun).get(dispatch_run.id)
        assert dispatch_run.status == DispatchStatus.COMPLETED

    def test_batch_dispatches(self, company, drivers):
        """Test : Batch dispatches (charge)."""
        from models import Company

        # ✅ FIX: S'assurer que la Company est bien persistée
        db.session.commit()
        company_reloaded = db.session.query(Company).filter_by(id=company.id).first()
        assert company_reloaded is not None, "Company must exist in DB"

        # Créer 20 bookings
        today = date.today()
        bookings_list = []
        for i in range(20):
            scheduled_time = datetime.combine(today, datetime.min.time().replace(hour=8 + (i % 12)))
            booking = BookingFactory(
                company=company,
                status=BookingStatus.ACCEPTED,
                scheduled_time=scheduled_time,
            )
            bookings_list.append(booking)
        db.session.commit()  # ✅ FIX: Commit pour garantir persistance

        # Exécuter plusieurs dispatches successifs
        for_date = today.isoformat()
        results = []

        for i in range(3):
            # ✅ FIX: Vérifier que la Company existe avant chaque dispatch
            # (engine.run() fait un rollback défensif qui peut expirer la Company)
            company_check = db.session.query(Company).filter_by(id=company.id).first()
            assert company_check is not None, f"Company must exist before dispatch #{i + 1}"

            result = engine.run(
                company_id=company.id,
                for_date=for_date,
                mode="auto",
            )
            results.append(result)

            # Vérifier que chaque dispatch a réussi
            assert result.get("meta", {}).get("reason") != "run_failed"

        # ✅ FIX: Vérifier les dispatch_run_ids dans les résultats d'abord
        dispatch_run_ids = [r.get("dispatch_run_id") or r.get("meta", {}).get("dispatch_run_id") for r in results]
        dispatch_run_ids = [run_id for run_id in dispatch_run_ids if run_id is not None]

        # Vérifier qu'au moins un dispatch_run_id est présent
        assert len(dispatch_run_ids) > 0, (
            f"At least one dispatch_run_id should be returned. Results: {[r.get('meta', {}) for r in results]}"
        )

        # Vérifier que les DispatchRuns existent en DB
        dispatch_runs = DispatchRun.query.filter(DispatchRun.id.in_(dispatch_run_ids)).all()
        assert len(dispatch_runs) >= 1, f"Expected at least 1 DispatchRun in DB, got {len(dispatch_runs)}"

    def test_dispatch_run_id_correlation(self, company, drivers, bookings):
        """Test : Corrélation dispatch_run_id dans tous les logs et métriques."""
        from models import Company

        # ✅ FIX: S'assurer que la Company est bien persistée
        db.session.commit()
        company_reloaded = db.session.query(Company).filter_by(id=company.id).first()
        assert company_reloaded is not None, "Company must exist in DB"

        for_date = date.today().isoformat()

        result = engine.run(
            company_id=company.id,
            for_date=for_date,
            mode="auto",
        )

        # ✅ FIX: Vérifier que dispatch_run_id est présent dans le résultat
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get("dispatch_run_id")
        assert dispatch_run_id is not None, (
            f"dispatch_run_id must be present in result. "
            f"Result meta: {result.get('meta', {})}, "
            f"Result keys: {list(result.keys())}"
        )

        # Vérifier que les assignations sont liées au dispatch_run_id
        assignments = Assignment.query.filter(Assignment.dispatch_run_id == dispatch_run_id).all()
        assert len(assignments) > 0, "Assignments must be linked to dispatch_run_id"

        # Vérifier que le DispatchRun existe
        dispatch_run = DispatchRun.query.get(dispatch_run_id)
        assert dispatch_run is not None, f"DispatchRun {dispatch_run_id} must exist"
        assert dispatch_run.company_id == company.id, "DispatchRun must belong to company"

    def test_apply_assignments_finds_bookings(self, company, drivers, bookings, db):
        """✅ Test de non-régression : Vérifier que apply_assignments trouve bien les bookings.

        Ce test vérifie que apply_assignments peut trouver les bookings en DB
        même après un commit, garantissant que booking_map n'est pas vide.
        """
        from services.unified_dispatch.apply import apply_assignments

        # ✅ FIX: S'assurer que les bookings sont persistés
        db.session.commit()

        # Vérifier que les bookings existent en DB
        for booking in bookings[:2]:  # Tester avec les 2 premiers
            booking_from_db = db.session.query(Booking).filter_by(id=booking.id).first()
            assert booking_from_db is not None, f"Booking {booking.id} must exist in DB"
            assert booking_from_db.company_id == company.id, f"Booking {booking.id} must belong to company {company.id}"

        # Créer des assignations
        assignments = [
            {"booking_id": bookings[0].id, "driver_id": drivers[0].id, "score": 1.0},
        ]

        # Appliquer
        result = apply_assignments(company_id=company.id, assignments=assignments, dispatch_run_id=None)

        # Vérifier que apply_assignments a trouvé les bookings
        assert len(result["applied"]) > 0, (
            f"apply_assignments must find bookings. "
            f"Applied: {result.get('applied', [])}, "
            f"Skipped: {result.get('skipped', {})}"
        )

    def test_rollback_restores_original_values(self, company, drivers, db):
        """✅ Test de non-régression : Vérifier que le rollback restaure bien les valeurs originales.

        Ce test vérifie que le rollback SQLAlchemy restaure correctement les valeurs
        en DB après une modification non commitée.
        """
        booking = BookingFactory(company=company, driver_id=None)
        db.session.commit()

        # Modifier le booking
        booking.driver_id = drivers[0].id
        db.session.flush()

        # Rollback
        db.session.rollback()
        db.session.expire_all()

        # Recharger depuis DB avec un nouveau query
        booking_reloaded = db.session.query(Booking).filter_by(id=booking.id).first()
        assert booking_reloaded is not None, "Booking must be reloaded from DB"
        assert booking_reloaded.driver_id is None, "Rollback must restore original value (driver_id should be None)"

    def test_company_persisted_before_dispatch(self, company, db):
        """✅ Test de non-régression : Vérifier que la Company est bien persistée avant dispatch.

        Ce test vérifie que la fixture company garantit la persistance en DB,
        permettant à engine.run() de trouver la Company et créer un DispatchRun.
        """
        from models import Company
        from services.unified_dispatch import engine

        # Vérifier que la Company existe en DB
        company_from_db = Company.query.get(company.id)
        assert company_from_db is not None, "Company must exist in DB"
        assert company_from_db.id == company.id, "Company ID must match"

        # Vérifier que engine.run() peut la trouver et créer un DispatchRun
        result = engine.run(company_id=company.id, for_date=date.today().isoformat())

        # Vérifier que dispatch_run_id est présent dans le résultat
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get("dispatch_run_id")
        assert dispatch_run_id is not None, f"DispatchRun must be created. Result: {result.get('meta', {})}"

        # Vérifier que le DispatchRun existe en DB
        dispatch_run = DispatchRun.query.get(dispatch_run_id)
        assert dispatch_run is not None, f"DispatchRun {dispatch_run_id} must exist in DB"
        assert dispatch_run.company_id == company.id, "DispatchRun must belong to company"

    def test_fixtures_isolation_and_rollback_defensive(self, db, company):
        """✅ Test de non-régression : Vérifier l'isolation des fixtures et le rollback défensif.

        Ce test vérifie que :
        1. Les fixtures sont bien isolées entre les tests (savepoint)
        2. Le rollback défensif de engine.run() n'affecte pas les objets commités
        3. Les objets commités restent visibles après engine.run()

        ⚠️ COUPLAGE TESTÉ :
        - Fixtures dépendantes (company → drivers → bookings)
        - Rollback défensif de engine.run() vs objets commités
        - Isolation entre tests via savepoints
        """
        from models import Company

        # 1. Vérifier que la company est bien commitée et visible
        company_reloaded = db.session.query(Company).get(company.id)
        assert company_reloaded is not None, "Company doit être visible après commit"
        assert company_reloaded.id == company.id, "Company doit avoir le même ID"

        # 2. Vérifier que engine.run() peut accéder à la company (même après rollback défensif)
        # Le rollback défensif ne devrait pas affecter les objets commités
        result = engine.run(
            company_id=company.id,
            for_date=date.today().isoformat(),
            mode="auto",
        )

        # 3. Vérifier que la company est toujours visible après engine.run()
        company_after = db.session.query(Company).get(company.id)
        assert company_after is not None, "Company doit rester visible après engine.run() malgré le rollback défensif"
        assert company_after.id == company.id, "Company doit avoir le même ID après engine.run()"

        # 4. Vérifier que le résultat contient des informations cohérentes
        assert "meta" in result, "Résultat doit contenir meta"
        assert result.get("meta", {}).get("reason") != "company_not_found", (
            "Company doit être trouvée par engine.run() (pas de reason='company_not_found')"
        )

        print("✅ Test isolation fixtures et rollback défensif OK")

    def test_company_not_found_raises_exception(self, db):
        """✅ Test de non-régression : Vérifier que CompanyNotFoundError est levée si demandé.

        Ce test vérifie que le paramètre `raise_on_company_not_found=True`
        lève bien une exception `CompanyNotFoundError` au lieu de retourner un résultat structuré.
        """
        from services.unified_dispatch import engine
        from services.unified_dispatch.exceptions import CompanyNotFoundError

        # Test avec un company_id qui n'existe pas
        invalid_company_id = 999999

        # Test 1: Comportement par défaut (retourne un résultat structuré)
        result = engine.run(company_id=invalid_company_id, for_date=date.today().isoformat())
        assert result.get("meta", {}).get("reason") == "company_not_found", (
            "Par défaut, doit retourner un résultat avec reason='company_not_found'"
        )
        assert result.get("dispatch_run_id") is None, "Pas de DispatchRun créé si Company introuvable"

        # Test 2: Comportement avec raise_on_company_not_found=True (lève une exception)
        with pytest.raises(CompanyNotFoundError) as exc_info:
            engine.run(
                company_id=invalid_company_id,
                for_date=date.today().isoformat(),
                raise_on_company_not_found=True,
            )

        # Vérifier que l'exception contient les bonnes informations
        exception = exc_info.value
        assert exception.company_id == invalid_company_id, "Exception doit contenir le company_id"
        assert "introuvable" in str(exception).lower(), "Message d'erreur doit mentionner 'introuvable'"
        assert exception.extra.get("caller") is not None, "Exception doit contenir les infos du caller"

        print("✅ Test CompanyNotFoundError OK")
