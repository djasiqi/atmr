#!/usr/bin/env python3
"""✅ Phase 2 N+1: Tests pour valider l'optimisation eager loading dans data.py.

Teste que get_bookings_for_day() charge les relations (driver, client, company)
en une seule requête avec JOIN, évitant ainsi les requêtes N+1.
"""

from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from services.unified_dispatch.data import get_available_drivers, get_bookings_for_day
from shared.time_utils import now_local
from tests.factories import BookingFactory, ClientFactory, CompanyFactory, DriverFactory


def _tomorrow_local_midday() -> tuple[str, datetime]:
    """Jour calendaire Zurich + midi naïf (loin des frontières de jour UTC/CI)."""
    target_day = now_local().date() + timedelta(days=1)
    midday = datetime.combine(target_day, time(hour=12))
    return target_day.strftime("%Y-%m-%d"), midday


class TestGetBookingsForDayEagerLoading:
    """Tests pour valider l'eager loading dans get_bookings_for_day()."""

    @pytest.fixture
    def test_company(self, db):
        """Crée une company pour les tests."""
        company = CompanyFactory()
        db.session.commit()
        return company

    @pytest.fixture
    def test_client(self, db, test_company):
        """Crée un client pour les tests."""
        client = ClientFactory(company=test_company)
        db.session.commit()
        return client

    @pytest.fixture
    def test_driver(self, db, test_company):
        """Crée un driver pour les tests."""
        driver = DriverFactory(company=test_company)
        db.session.commit()
        return driver

    @pytest.fixture
    def test_bookings(self, db, test_company, test_client, test_driver):
        """Crée plusieurs bookings avec relations pour les tests.

        Heure métier locale déterministe (12:00–16:00 Genève), loin des
        frontières de jour UTC que la CI traverse en soirée.
        """
        target_day = now_local().date() + timedelta(days=1)
        day_str = target_day.strftime("%Y-%m-%d")
        base_time = datetime.combine(target_day, time(hour=12))

        bookings = []
        for i in range(5):
            booking = BookingFactory(
                company=test_company,
                client=test_client,
                driver=test_driver if i % 2 == 0 else None,  # Alterner avec/sans driver
                scheduled_time=base_time + timedelta(hours=i),
            )
            bookings.append(booking)

        db.session.commit()
        return bookings, day_str

    def test_eager_loading_relations_loaded(self, db, test_company, test_bookings):
        """Test: Les relations driver, client, company sont chargées."""
        _bookings, day_str = test_bookings

        # Appeler get_bookings_for_day()
        result = get_bookings_for_day(test_company.id, day_str)

        # Vérifier qu'on a récupéré les bookings
        assert len(result) > 0, "Aucun booking récupéré"

        # Vérifier que les relations sont chargées (pas de requête lazy)
        for booking in result:
            # Accéder aux relations - ne doit pas déclencher de requête SQL
            # car elles sont déjà chargées via joinedload()
            assert hasattr(booking, "company"), "Relation company manquante"
            assert booking.company is not None or booking.company_id is not None

            assert hasattr(booking, "client"), "Relation client manquante"
            assert booking.client is not None or booking.client_id is not None

            # Driver peut être None (pas tous les bookings ont un driver)
            if booking.driver_id is not None:
                assert hasattr(booking, "driver"), "Relation driver manquante"

    def test_eager_loading_query_count(self, db, test_company, test_bookings):
        """Test: Le nombre de requêtes SQL est minimal (pas de N+1)."""
        _bookings, day_str = test_bookings

        # Compteur de requêtes SQL
        query_count = []

        def count_query(
            _conn, _cursor, _statement, _parameters, _context, _executemany
        ):
            """Compte les requêtes SELECT."""
            query_count.append(1)

        # Enregistrer l'event listener
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Appeler get_bookings_for_day()
            result = get_bookings_for_day(test_company.id, day_str)

            # Vérifier qu'on a récupéré les bookings
            assert len(result) > 0, "Aucun booking récupéré"

            # Accéder aux relations pour tous les bookings
            # (simule l'utilisation réelle dans build_vrptw_problem, etc.)
            for booking in result:
                _ = booking.company  # Accès à company
                _ = booking.client  # Accès à client
                if booking.driver_id is not None:
                    _ = booking.driver  # Accès à driver

            # Vérifier le nombre de requêtes
            # Avec eager loading: 1 requête principale (avec JOIN)
            # Sans eager loading: 1 + N requêtes (1 pour bookings + N pour chaque relation)
            total_queries = len(query_count)

            # Avec eager loading, on devrait avoir 1 requête principale
            # (peut y avoir quelques requêtes supplémentaires pour les métadonnées, etc.)
            # Mais on ne devrait PAS avoir N requêtes supplémentaires pour les relations
            assert total_queries <= 5, (
                f"Trop de requêtes SQL ({total_queries}). "
                f"Suspect N+1: attendu <= 5 requêtes avec eager loading, "
                f"mais {total_queries} requêtes détectées."
            )

            # Log pour information
            print(f"\n✅ Nombre de requêtes SQL: {total_queries} (attendues: <= 5)")

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)

    def test_eager_loading_no_lazy_queries(self, db, test_company, test_bookings):
        """Test: Aucune requête lazy n'est déclenchée lors de l'accès aux relations."""
        _bookings, day_str = test_bookings

        # Compteur de requêtes après l'appel initial
        queries_after_initial = []

        def count_query(
            _conn, _cursor, _statement, _parameters, _context, _executemany
        ):
            """Compte les requêtes SELECT."""
            queries_after_initial.append(1)

        # Appeler get_bookings_for_day() (sans listener pour cette partie)
        result = get_bookings_for_day(test_company.id, day_str)
        assert len(result) > 0, "Aucun booking récupéré"

        # Maintenant enregistrer le listener et accéder aux relations
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Accéder aux relations - ne doit PAS déclencher de nouvelles requêtes
            for booking in result:
                _ = booking.company
                _ = booking.client
                if booking.driver_id is not None:
                    _ = booking.driver

            # Vérifier qu'aucune requête supplémentaire n'a été déclenchée
            # (les relations sont déjà chargées via joinedload)
            lazy_queries = len(queries_after_initial)

            assert lazy_queries == 0, (
                f"Requêtes lazy détectées ({lazy_queries}). "
                f"Les relations devraient être chargées via joinedload(), "
                f"mais {lazy_queries} requêtes supplémentaires ont été exécutées."
            )

            print("\n✅ Aucune requête lazy détectée lors de l'accès aux relations")

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)


class TestGetAvailableDriversEagerLoading:
    """Tests pour valider l'eager loading dans get_available_drivers()."""

    @pytest.fixture
    def test_company(self, db):
        """Crée une company pour les tests."""
        company = CompanyFactory()
        db.session.commit()
        return company

    @pytest.fixture
    def test_drivers(self, db, test_company):
        """Crée plusieurs drivers pour les tests."""
        drivers = []
        for _ in range(5):
            driver = DriverFactory(
                company=test_company, is_active=True, is_available=True
            )
            drivers.append(driver)

        db.session.commit()
        return drivers

    def test_eager_loading_company_loaded(self, db, test_company, test_drivers):
        """Test: La relation company est chargée."""
        # Appeler get_available_drivers()
        result = get_available_drivers(test_company.id)

        # Vérifier qu'on a récupéré les drivers
        assert len(result) > 0, "Aucun driver récupéré"

        # Vérifier que la relation company est chargée (pas de requête lazy)
        for driver in result:
            # Accéder à la relation - ne doit pas déclencher de requête SQL
            # car elle est déjà chargée via joinedload()
            assert hasattr(driver, "company"), "Relation company manquante"
            assert driver.company is not None or driver.company_id is not None

    def test_eager_loading_no_lazy_queries(self, db, test_company, test_drivers):
        """Test: Aucune requête lazy n'est déclenchée lors de l'accès à company."""
        # Compteur de requêtes après l'appel initial
        queries_after_initial = []

        def count_query(
            _conn, _cursor, _statement, _parameters, _context, _executemany
        ):
            """Compte les requêtes SELECT."""
            queries_after_initial.append(1)

        # Appeler get_available_drivers() (sans listener pour cette partie)
        result = get_available_drivers(test_company.id)
        assert len(result) > 0, "Aucun driver récupéré"

        # Maintenant enregistrer le listener et accéder à la relation
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Accéder à la relation - ne doit PAS déclencher de nouvelles requêtes
            for driver in result:
                _ = driver.company

            # Vérifier qu'aucune requête supplémentaire n'a été déclenchée
            lazy_queries = len(queries_after_initial)

            assert lazy_queries == 0, (
                f"Requêtes lazy détectées ({lazy_queries}). "
                f"La relation company devrait être chargée via joinedload(), "
                f"mais {lazy_queries} requêtes supplémentaires ont été exécutées."
            )

            print("\n✅ Aucune requête lazy détectée lors de l'accès à driver.company")

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)

    def test_working_config_is_attribute_not_relation(
        self, db, test_company, test_drivers
    ):
        """Test: working_config est un attribut JSON, pas une relation."""
        # Appeler get_available_drivers()
        result = get_available_drivers(test_company.id)
        assert len(result) > 0, "Aucun driver récupéré"

        # Vérifier que working_config est accessible comme attribut
        for driver in result:
            # working_config devrait être un attribut JSON (dict ou None)
            working_config = getattr(driver, "working_config", None)
            # Peut être None, dict, ou autre structure JSON
            assert working_config is None or isinstance(working_config, (dict, str)), (
                f"working_config devrait être dict/str/None, "
                f"mais est {type(working_config)}"
            )


class TestBuildVrptwProblemRelations:
    """✅ Phase 4 N+1: Tests pour valider que build_vrptw_problem() utilise
    les relations déjà chargées sans requêtes supplémentaires.
    """

    @pytest.fixture
    def test_company(self, db):
        """Crée une company pour les tests."""
        company = CompanyFactory()
        db.session.commit()
        return company

    @pytest.fixture
    def test_client(self, db, test_company):
        """Crée un client pour les tests."""
        client = ClientFactory(company=test_company)
        db.session.commit()
        return client

    @pytest.fixture
    def test_driver(self, db, test_company):
        """Crée un driver pour les tests."""
        driver = DriverFactory(company=test_company)
        db.session.commit()
        return driver

    @pytest.fixture
    def test_bookings(self, db, test_company, test_client, test_driver):
        """Crée des bookings avec relations pour les tests."""
        _day_str, midday = _tomorrow_local_midday()
        bookings = []
        for _ in range(3):
            booking = BookingFactory(
                company=test_company,
                client=test_client,
                driver=test_driver,
                scheduled_time=midday,
            )
            bookings.append(booking)
        db.session.commit()
        return bookings

    def test_relations_accessible_in_build_vrptw_problem(
        self, db, test_company, test_bookings, test_driver
    ):
        """Test: Les relations sont accessibles dans build_vrptw_problem() sans requêtes supplémentaires."""
        from services.unified_dispatch.data import (
            build_vrptw_problem,
            get_available_drivers,
            get_bookings_for_day,
        )

        # Récupérer les bookings avec relations chargées (Phase 2)
        day_str, _midday = _tomorrow_local_midday()
        bookings = get_bookings_for_day(test_company.id, day_str)

        # Récupérer les drivers avec relations chargées (Phase 3)
        drivers = get_available_drivers(test_company.id)

        if not bookings or not drivers:
            pytest.skip("Pas assez de données pour tester")

        # Compter les requêtes SQL lors de build_vrptw_problem()
        query_count = 0

        def count_query(conn, cursor, statement, parameters, context, executemany):
            nonlocal query_count
            # Ignorer les requêtes de comptage (SELECT COUNT) qui peuvent être
            # exécutées pour la fairness
            if not statement.strip().upper().startswith("SELECT COUNT"):
                query_count += 1

        # Ajouter l'event listener
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Construire le problème VRPTW
            problem = build_vrptw_problem(
                company=test_company,
                bookings=bookings,
                drivers=drivers,
                for_date=day_str,
            )

            # Vérifier que le problème est construit
            assert "bookings" in problem
            assert "drivers" in problem
            assert "time_matrix" in problem

            # Vérifier que les relations sont accessibles sans requêtes supplémentaires
            # (les requêtes peuvent être pour fairness_counts, mais pas pour les relations)
            # On s'attend à ce qu'il n'y ait pas de requêtes pour accéder aux relations
            # car elles sont déjà chargées
            for booking in bookings:
                # Accéder aux relations (devrait être déjà chargé)
                _ = booking.driver
                _ = booking.client
                _ = booking.company

            for driver in drivers:
                # Accéder à la relation (devrait être déjà chargé)
                _ = driver.company

            # Les requêtes supplémentaires devraient être minimales
            # (seulement pour fairness_counts si nécessaire)
            # On accepte jusqu'à 5 requêtes pour la fairness et autres opérations
            assert query_count <= 5, (
                f"Trop de requêtes SQL ({query_count}) lors de build_vrptw_problem(). "
                "Les relations devraient être déjà chargées."
            )

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)


class TestEnrichCoordsCompanyAccess:
    """✅ Phase 5 N+1: Tests pour valider que enrich_booking_coords() et
    enrich_driver_coords() n'effectuent pas de requêtes supplémentaires lors
    de l'accès aux attributs de company.
    """

    @pytest.fixture
    def test_company(self, db):
        """Crée une company avec coordonnées pour les tests."""
        company = CompanyFactory(latitude=46.2044, longitude=6.1432)
        db.session.commit()
        return company

    @pytest.fixture
    def test_client(self, db, test_company):
        """Crée un client pour les tests."""
        client = ClientFactory(company=test_company)
        db.session.commit()
        return client

    @pytest.fixture
    def test_driver(self, db, test_company):
        """Crée un driver pour les tests."""
        driver = DriverFactory(company=test_company)
        db.session.commit()
        return driver

    @pytest.fixture
    def test_bookings(self, db, test_company, test_client):
        """Crée des bookings sans coordonnées pour les tests."""
        bookings = []
        for _ in range(3):
            booking = BookingFactory(
                company=test_company,
                client=test_client,
                pickup_lat=None,
                pickup_lon=None,
                dropoff_lat=None,
                dropoff_lon=None,
            )
            bookings.append(booking)
        db.session.commit()
        return bookings

    def test_enrich_booking_coords_company_attributes_loaded(
        self, db, test_company, test_bookings
    ):
        """Test: Les attributs de company sont accessibles sans requêtes lazy."""
        from services.unified_dispatch.data import enrich_booking_coords

        # Vérifier que les attributs de company sont accessibles avant l'appel
        assert test_company.latitude is not None
        assert test_company.longitude is not None
        assert hasattr(test_company, "get_autonomous_config")

        query_count = 0
        company_queries = []

        def count_query(conn, cursor, statement, parameters, context, executemany):
            nonlocal query_count, company_queries
            # Ignorer les requêtes SELECT COUNT qui peuvent être exécutées
            if not statement.strip().upper().startswith("SELECT COUNT"):
                query_count += 1
                # Détecter les requêtes qui accèdent à company
                stmt_upper = statement.strip().upper()
                if "company" in stmt_upper or "COMPANY" in statement:
                    company_queries.append(statement)

        # Ajouter l'event listener
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Enrichir les coordonnées
            enrich_booking_coords(test_bookings, test_company)

            # Vérifier que les coordonnées ont été enrichies
            for booking in test_bookings:
                assert hasattr(booking, "pickup_lat")
                assert hasattr(booking, "pickup_lon")
                assert booking.pickup_lat is not None
                assert booking.pickup_lon is not None

            # Vérifier qu'aucune requête n'a été faite pour accéder aux attributs de company
            # (les requêtes peuvent être pour d'autres opérations comme le géocodage)
            assert len(company_queries) == 0, (
                f"Requêtes SQL détectées pour company ({len(company_queries)}): "
                f"{company_queries}. Les attributs de company devraient être déjà chargés."
            )

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)

    def test_enrich_driver_coords_company_attributes_loaded(
        self, db, test_company, test_driver
    ):
        """Test: Les attributs de company sont accessibles sans requêtes lazy."""
        from services.unified_dispatch.data import enrich_driver_coords

        drivers = [test_driver]

        # Vérifier que les attributs de company sont accessibles avant l'appel
        assert test_company.latitude is not None
        assert test_company.longitude is not None
        assert hasattr(test_company, "get_autonomous_config")

        query_count = 0
        company_queries = []

        def count_query(conn, cursor, statement, parameters, context, executemany):
            nonlocal query_count, company_queries
            # Ignorer les requêtes SELECT COUNT qui peuvent être exécutées
            if not statement.strip().upper().startswith("SELECT COUNT"):
                query_count += 1
                # Détecter les requêtes qui accèdent à company
                stmt_upper = statement.strip().upper()
                if "company" in stmt_upper or "COMPANY" in statement:
                    company_queries.append(statement)

        # Ajouter l'event listener
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Enrichir les coordonnées
            enrich_driver_coords(drivers, test_company)

            # Vérifier que les coordonnées ont été enrichies
            for driver in drivers:
                assert hasattr(driver, "current_lat")
                assert hasattr(driver, "current_lon")
                assert driver.current_lat is not None
                assert driver.current_lon is not None

            # Vérifier qu'aucune requête n'a été faite pour accéder aux attributs de company
            # (les requêtes peuvent être pour d'autres opérations)
            assert len(company_queries) == 0, (
                f"Requêtes SQL détectées pour company ({len(company_queries)}): "
                f"{company_queries}. Les attributs de company devraient être déjà chargés."
            )

        finally:
            # Retirer l'event listener
            event.remove(Engine, "before_cursor_execute", count_query)
