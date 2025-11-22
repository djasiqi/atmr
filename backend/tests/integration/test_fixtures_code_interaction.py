# backend/tests/integration/test_fixtures_code_interaction.py
"""✅ Tests d'intégration : Interaction entre fixtures (savepoints) et code métier (transactions).

Ces tests vérifient que l'isolation entre les fixtures de test (savepoints) et le code métier
(transactions) fonctionne correctement.

Scénarios testés :
1. Les objets commités dans les fixtures sont visibles dans le code métier
2. Le rollback défensif de engine.run() n'affecte pas les objets commités dans les fixtures
3. Les transactions du code métier n'affectent pas l'isolation des fixtures
4. Les savepoints imbriqués fonctionnent correctement avec le code métier
"""

from datetime import date

from models import Company, DispatchRun
from services.unified_dispatch import engine
from tests.conftest import ensure_committed, nested_savepoint, persisted_fixture
from tests.factories import CompanyFactory


class TestFixturesCodeInteraction:
    """Tests d'intégration pour l'interaction entre fixtures et code métier."""

    def test_fixture_committed_visible_in_code_metier(self, db):
        """✅ Test : Les objets commités dans les fixtures sont visibles dans le code métier.

        Ce test vérifie que les objets créés et commités via persisted_fixture()
        sont bien visibles dans le code métier (engine.run()).
        """
        # Créer une company via persisted_fixture (commit automatique)
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Vérifier que la company est visible dans le code métier
        result = engine.run(
            company_id=company.id,
            for_date=date.today().isoformat(),
            mode="auto",
        )

        # Vérifier que engine.run() a pu trouver la company et créer un DispatchRun
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get(
            "dispatch_run_id"
        )
        assert dispatch_run_id is not None, (
            f"DispatchRun doit être créé. Company doit être visible dans le code métier. "
            f"Résultat: {result.get('meta', {})}"
        )

        # Vérifier que le DispatchRun existe en DB
        dispatch_run = DispatchRun.query.get(dispatch_run_id)
        assert dispatch_run is not None, (
            f"DispatchRun {dispatch_run_id} doit exister en DB"
        )
        assert dispatch_run.company_id == company.id, (
            "DispatchRun doit appartenir à la company"
        )

        print("✅ Test : Fixture commitée visible dans code métier OK")

    def test_rollback_defensif_does_not_affect_committed_fixtures(self, db):
        """✅ Test : Le rollback défensif de engine.run() n'affecte pas les objets commités.

        Ce test vérifie que le rollback défensif effectué par engine.run() au début
        n'affecte pas les objets qui ont été commités dans les fixtures.
        """
        # Créer une company via persisted_fixture (commit automatique)
        company = persisted_fixture(db, CompanyFactory(), Company)
        company_id = company.id

        # Appeler engine.run() qui fait un rollback défensif
        result = engine.run(
            company_id=company_id,
            for_date=date.today().isoformat(),
            mode="auto",
        )

        # Vérifier que la company est toujours visible après engine.run()
        company_reloaded = db.session.query(Company).get(company_id)
        assert company_reloaded is not None, (
            "Company doit rester visible après engine.run() malgré le rollback défensif"
        )
        assert company_reloaded.id == company_id, "Company doit avoir le même ID"

        # Vérifier que engine.run() a pu créer un DispatchRun
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get(
            "dispatch_run_id"
        )
        assert dispatch_run_id is not None, "DispatchRun doit être créé"

        print("✅ Test : Rollback défensif n'affecte pas fixtures commitées OK")

    def test_code_metier_transaction_does_not_affect_fixture_isolation(self, db):
        """✅ Test : Les transactions du code métier n'affectent pas l'isolation des fixtures.

        Ce test vérifie que les transactions créées par le code métier (via engine.run())
        n'affectent pas l'isolation garantie par les savepoints des fixtures.
        """
        # Créer une company via persisted_fixture (dans le savepoint du test)
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Appeler engine.run() qui crée sa propre transaction
        result1 = engine.run(
            company_id=company.id,
            for_date=date.today().isoformat(),
            mode="auto",
        )

        # Appeler engine.run() une deuxième fois
        result2 = engine.run(
            company_id=company.id,
            for_date=date.today().isoformat(),
            mode="auto",
        )

        # Vérifier que les deux appels ont créé des DispatchRun distincts
        dispatch_run_id1 = result1.get("dispatch_run_id") or result1.get(
            "meta", {}
        ).get("dispatch_run_id")
        dispatch_run_id2 = result2.get("dispatch_run_id") or result2.get(
            "meta", {}
        ).get("dispatch_run_id")

        assert dispatch_run_id1 is not None, "Premier DispatchRun doit être créé"
        assert dispatch_run_id2 is not None, "Deuxième DispatchRun doit être créé"
        assert dispatch_run_id1 != dispatch_run_id2, (
            "Les deux DispatchRun doivent être distincts"
        )

        # Vérifier que la company est toujours visible (isolation préservée)
        company_reloaded = db.session.query(Company).get(company.id)
        assert company_reloaded is not None, (
            "Company doit rester visible (isolation préservée)"
        )

        print(
            "✅ Test : Transactions code métier n'affectent pas isolation fixtures OK"
        )

    def test_nested_savepoint_with_code_metier(self, db):
        """✅ Test : Les savepoints imbriqués fonctionnent correctement avec le code métier.

        Ce test vérifie que les savepoints imbriqués (via nested_savepoint()) fonctionnent
        correctement avec le code métier (engine.run()).
        """
        # Créer une company dans le savepoint principal
        company = persisted_fixture(db, CompanyFactory(), Company)

        # Créer un savepoint imbriqué
        with nested_savepoint(db):
            # Appeler engine.run() dans le savepoint imbriqué
            result = engine.run(
                company_id=company.id,
                for_date=date.today().isoformat(),
                mode="auto",
            )

            # Vérifier que engine.run() a pu créer un DispatchRun
            dispatch_run_id = result.get("dispatch_run_id") or result.get(
                "meta", {}
            ).get("dispatch_run_id")
            assert dispatch_run_id is not None, (
                "DispatchRun doit être créé dans le savepoint imbriqué"
            )

            # Vérifier que le DispatchRun existe
            dispatch_run = DispatchRun.query.get(dispatch_run_id)
            assert dispatch_run is not None, "DispatchRun doit exister"

        # Après la sortie du savepoint imbriqué, le DispatchRun devrait toujours exister
        # car il a été créé dans une transaction normale (pas dans le savepoint)
        # Note: engine.run() crée sa propre transaction, donc elle n'est pas affectée par le rollback du savepoint
        dispatch_run_after = DispatchRun.query.get(dispatch_run_id)
        # Le DispatchRun devrait toujours exister car engine.run() commit dans sa propre transaction
        assert dispatch_run_after is not None, (
            "DispatchRun doit toujours exister car engine.run() commit dans sa propre transaction"
        )

        # La company doit toujours exister (créée dans le savepoint principal)
        company_reloaded = db.session.query(Company).get(company.id)
        assert company_reloaded is not None, (
            "Company doit toujours exister (savepoint principal)"
        )

        print("✅ Test : Savepoints imbriqués avec code métier OK")

    def test_ensure_committed_with_code_metier(self, db):
        """✅ Test : ensure_committed() garantit la persistance avant code métier.

        Ce test vérifie que ensure_committed() garantit bien que tous les objets
        sont commités avant d'appeler le code métier.
        """
        # Créer une company sans utiliser persisted_fixture
        company = CompanyFactory()
        db.session.add(company)
        db.session.flush()

        # Utiliser ensure_committed() pour garantir le commit
        with ensure_committed(db):
            # Vérifier que la company est commitée
            company_reloaded = db.session.query(Company).get(company.id)
            assert company_reloaded is not None, (
                "Company doit être visible après ensure_committed()"
            )

            # Appeler engine.run() qui fait un rollback défensif
            result = engine.run(
                company_id=company.id,
                for_date=date.today().isoformat(),
                mode="auto",
            )

            # Vérifier que engine.run() a pu créer un DispatchRun
            dispatch_run_id = result.get("dispatch_run_id") or result.get(
                "meta", {}
            ).get("dispatch_run_id")
            assert dispatch_run_id is not None, (
                "DispatchRun doit être créé. ensure_committed() doit garantir la persistance"
            )

        print("✅ Test : ensure_committed() garantit persistance avant code métier OK")
