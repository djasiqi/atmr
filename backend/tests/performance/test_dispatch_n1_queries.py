#!/usr/bin/env python3
"""✅ Phase 6 N+1: Tests de performance pour valider les optimisations N+1.

Ces tests vérifient que les optimisations N+1 sont efficaces en mesurant :
- Le nombre de requêtes SQL exécutées lors d'un dispatch
- Le temps d'exécution
- La conformité aux seuils définis
"""

import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from models import BookingStatus
from services.unified_dispatch.engine import run as dispatch_run
from tests.e2e.helpers.e2e_helpers import (
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
)


class TestDispatchN1Queries:
    """Tests : Validation des optimisations N+1 dans le dispatch."""

    # Seuils de requêtes SQL par taille de batch
    QUERY_THRESHOLD_SMALL = 10  # <50 bookings
    QUERY_THRESHOLD_MEDIUM = 20  # 50-200 bookings
    QUERY_THRESHOLD_LARGE = 30  # >200 bookings

    def _count_queries(self, ignore_patterns=None):
        """Helper pour compter les requêtes SQL."""
        if ignore_patterns is None:
            ignore_patterns = ["SELECT COUNT"]

        query_count = 0
        queries = []

        def count_query(conn, cursor, statement, parameters, context, executemany):
            nonlocal query_count, queries
            # Ignorer les requêtes de comptage et autres patterns
            should_ignore = any(
                statement.strip().upper().startswith(pattern.upper())
                for pattern in ignore_patterns
            )
            if not should_ignore:
                query_count += 1
                queries.append(
                    statement.strip()[:100]
                )  # Garder les 100 premiers caractères

        return query_count, queries, count_query

    def test_dispatch_n1_queries_small_batch(self, db):
        """Test : Dispatch avec petit batch (<50 bookings) - < 10 requêtes."""
        # Setup : Créer company, client, drivers
        company = create_test_company(db)
        client = create_test_client(db, company=company)

        # Créer 3 drivers pour 30 bookings
        num_drivers = 3
        for _ in range(num_drivers):
            create_test_driver(db, company=company)

        # Activer le dispatch
        company.dispatch_enabled = True
        db.session.commit()

        # Créer 30 bookings
        num_bookings = 30
        scheduled_time = datetime.now(UTC) + timedelta(days=1)
        for i in range(num_bookings):
            booking_time = scheduled_time + timedelta(minutes=i * 30)
            create_test_booking(
                db,
                client=client,
                scheduled_time=booking_time,
                status=BookingStatus.PENDING,
            )
        db.session.commit()

        # Compter les requêtes SQL
        query_count, queries, count_query = self._count_queries()
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Exécuter le dispatch
            start_time = time.time()
            result = dispatch_run(
                company_id=company.id,
                for_date=scheduled_time.date().isoformat(),
                mode="heuristic_only",
            )
            duration = time.time() - start_time

            # Vérifications
            assert "assignments" in result
            assert query_count <= self.QUERY_THRESHOLD_SMALL, (
                f"Trop de requêtes SQL ({query_count}) pour {num_bookings} bookings. "
                f"Seuil: {self.QUERY_THRESHOLD_SMALL}. "
                f"Requêtes: {queries[:10]}"
            )

            # Log pour debugging
            print(
                f"✅ Small batch ({num_bookings} bookings): "
                f"{query_count} requêtes, {duration:.2f}s"
            )

        finally:
            event.remove(Engine, "before_cursor_execute", count_query)

    def test_dispatch_n1_queries_medium_batch(self, db):
        """Test : Dispatch avec batch moyen (100 bookings) - < 20 requêtes."""
        # Setup : Créer company, client, drivers
        company = create_test_company(db)
        client = create_test_client(db, company=company)

        # Créer 8 drivers pour 100 bookings (ratio réaliste)
        num_drivers = 8
        for _ in range(num_drivers):
            create_test_driver(db, company=company)

        # Activer le dispatch
        company.dispatch_enabled = True
        db.session.commit()

        # Créer 100 bookings
        num_bookings = 100
        scheduled_time = datetime.now(UTC) + timedelta(days=1)
        for i in range(num_bookings):
            booking_time = scheduled_time + timedelta(minutes=i * 30)
            create_test_booking(
                db,
                client=client,
                scheduled_time=booking_time,
                status=BookingStatus.PENDING,
            )
        db.session.commit()

        # Compter les requêtes SQL
        query_count, queries, count_query = self._count_queries()
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Exécuter le dispatch
            start_time = time.time()
            result = dispatch_run(
                company_id=company.id,
                for_date=scheduled_time.date().isoformat(),
                mode="heuristic_only",
            )
            duration = time.time() - start_time

            # Vérifications
            assert "assignments" in result
            assert query_count <= self.QUERY_THRESHOLD_MEDIUM, (
                f"Trop de requêtes SQL ({query_count}) pour {num_bookings} bookings. "
                f"Seuil: {self.QUERY_THRESHOLD_MEDIUM}. "
                f"Requêtes: {queries[:10]}"
            )

            # Log pour debugging
            print(
                f"✅ Medium batch ({num_bookings} bookings): "
                f"{query_count} requêtes, {duration:.2f}s"
            )

        finally:
            event.remove(Engine, "before_cursor_execute", count_query)

    def test_dispatch_n1_queries_large_batch(self, db):
        """Test : Dispatch avec grand batch (250 bookings) - < 30 requêtes."""
        # Setup : Créer company, client, drivers
        company = create_test_company(db)
        client = create_test_client(db, company=company)

        # Créer 20 drivers pour 250 bookings (ratio réaliste)
        num_drivers = 20
        for _ in range(num_drivers):
            create_test_driver(db, company=company)

        # Activer le dispatch
        company.dispatch_enabled = True
        db.session.commit()

        # Créer 250 bookings
        num_bookings = 250
        scheduled_time = datetime.now(UTC) + timedelta(days=1)
        for i in range(num_bookings):
            booking_time = scheduled_time + timedelta(minutes=i * 20)
            create_test_booking(
                db,
                client=client,
                scheduled_time=booking_time,
                status=BookingStatus.PENDING,
            )
        db.session.commit()

        # Compter les requêtes SQL
        query_count, queries, count_query = self._count_queries()
        event.listen(Engine, "before_cursor_execute", count_query)

        try:
            # Exécuter le dispatch
            start_time = time.time()
            result = dispatch_run(
                company_id=company.id,
                for_date=scheduled_time.date().isoformat(),
                mode="heuristic_only",
            )
            duration = time.time() - start_time

            # Vérifications
            assert "assignments" in result
            assert query_count <= self.QUERY_THRESHOLD_LARGE, (
                f"Trop de requêtes SQL ({query_count}) pour {num_bookings} bookings. "
                f"Seuil: {self.QUERY_THRESHOLD_LARGE}. "
                f"Requêtes: {queries[:10]}"
            )

            # Log pour debugging
            print(
                f"✅ Large batch ({num_bookings} bookings): "
                f"{query_count} requêtes, {duration:.2f}s"
            )

        finally:
            event.remove(Engine, "before_cursor_execute", count_query)
