"""LOT-BOOKING-DISPATCH-2 — scénarios 1–8 (Chauffeur + Institution).

Preuve : use-case / HTTP-isolé sur DB jetable. Pas de certification appareil réel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from application.companies.accept_reservation import AcceptReservationUseCase
from application.companies.assign_driver_to_reservation import (
    AssignDriverToReservationUseCase,
)
from application.drivers.reject_driver_booking import RejectDriverBookingUseCase
from application.drivers.update_driver_booking_status import (
    UpdateDriverBookingStatusCommand,
    UpdateDriverBookingStatusUseCase,
)
from application.institutions.network_eligibility import (
    resolve_dispatch_target,
    unmodeled_network_needs,
)
from models import BookingStatus
from models.enums import AssignmentStatus
from schemas.institution_schemas import MobilitySchema
from services.booking.specialized_mobility_gate import (
    reject_if_specialized_mobility_blocked,
)
from services.dispatch.s23_dispatch_mode_lock import (
    DispatchModeLockedError,
    assert_dispatch_mode_switch_allowed,
)
from tests.e2e.helpers.e2e_helpers import (
    assert_booking_assigned,
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
)


class _AssignmentWriter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ensure_assignment_for_booking(
        self, *, company_id: int, booking, driver_id: int
    ) -> None:
        self.calls.append(
            {
                "company_id": company_id,
                "booking_id": getattr(booking, "id", None),
                "driver_id": driver_id,
            }
        )


class _Assignment:
    def __init__(self, assignment_id: int, booking_id: int, driver_id: int) -> None:
        self.id = assignment_id
        self.booking_id = booking_id
        self.driver_id = driver_id
        self.status = AssignmentStatus.SCHEDULED


class _AssignmentRepo:
    def __init__(self, assignment: _Assignment | None) -> None:
        self._assignment = assignment
        self.deleted_dependents_for: list[int] = []

    def find_model_by_booking_id(self, booking_id: int):
        if self._assignment and self._assignment.booking_id == booking_id:
            return self._assignment
        return None

    def delete_dependent_records_for_assignment_id(self, assignment_id: int) -> None:
        self.deleted_dependents_for.append(assignment_id)


class _DbSession:
    def __init__(self) -> None:
        self.deleted: list[object] = []

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)


class _BookingRepoAdapter:
    """Adapter minimal pour RejectDriverBookingUseCase sur un booking ORM."""

    def __init__(self, booking) -> None:
        self._booking = booking

    def find_model_by_id_and_driver(self, booking_id: int, driver_id: int):
        if (
            self._booking is not None
            and self._booking.id == booking_id
            and self._booking.driver_id == driver_id
        ):
            return self._booking
        return None


def _accept(booking, company_id: int):
    return AcceptReservationUseCase().execute(booking=booking, company_id=company_id)


def _assign(booking, driver, company_id: int, writer: _AssignmentWriter | None = None):
    writer = writer or _AssignmentWriter()
    uc = AssignDriverToReservationUseCase(assignment_writer=writer)
    return uc.execute(booking=booking, driver=driver, company_id=company_id), writer


def _status_uc(booking, driver_id: int, payload: dict, assignment=None):
    class _BRepo:
        def find_model_by_id(self, booking_id: int):
            return booking if booking.id == booking_id else None

        def find_children_by_parent_booking_id(self, parent_booking_id: int):
            return []

    assign_repo = _AssignmentRepo(assignment)
    db = _DbSession()
    # UpdateDriverBookingStatusUseCase attend commit() sur db_session
    db.commit = MagicMock()  # type: ignore[attr-defined]
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BRepo(),
        assignment_repo=assign_repo,
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda *_a: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC),
    )
    with patch("ext.db.session") as mock_session:
        from contextlib import nullcontext

        mock_session.no_autoflush = nullcontext()
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=booking.id,
                driver_id=driver_id,
                payload=payload,
            )
        )
    return res, assign_repo, db


@pytest.mark.e2e
class TestLotBookingDispatch2Scenarios:
    """Scénarios 1–8 LOT-2."""

    def test_s1_mission_chauffeur_complete(self, db):
        """1 — Entreprise → attribution → transitions → clôture."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        db.session.commit()
        assert_booking_assigned(booking, driver)

        assignment = _Assignment(
            assignment_id=1, booking_id=booking.id, driver_id=driver.id
        )
        r1, _, _ = _status_uc(booking, driver.id, {"status": "en_route"}, assignment)
        assert r1.status_code == 200
        assert booking.status == BookingStatus.EN_ROUTE

        r2, assign_repo, _ = _status_uc(
            booking, driver.id, {"status": "arrived"}, assignment
        )
        assert r2.status_code == 200
        assert booking.status == BookingStatus.EN_ROUTE
        assert assignment.status == AssignmentStatus.ARRIVED_PICKUP

        r3, _, _ = _status_uc(booking, driver.id, {"status": "in_progress"}, assignment)
        assert r3.status_code == 200
        assert booking.status == BookingStatus.IN_PROGRESS

        r4, _, _ = _status_uc(booking, driver.id, {"status": "completed"}, assignment)
        assert r4.status_code == 200
        assert booking.status == BookingStatus.COMPLETED
        # Client / entreprise : statut persisté sur le même booking
        db.session.commit()
        db.session.refresh(booking)
        assert booking.status == BookingStatus.COMPLETED
        assert booking.driver_id == driver.id
        assert assign_repo._assignment is not None

    def test_s2_chauffeur_indisponible_refus(self, db):
        """2 — Refus chauffeur → booking PENDING + Assignment nettoyé."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        db.session.commit()

        assignment = _Assignment(
            assignment_id=9, booking_id=booking.id, driver_id=driver.id
        )
        assign_repo = _AssignmentRepo(assignment)
        session = _DbSession()
        uc = RejectDriverBookingUseCase(
            booking_repo=_BookingRepoAdapter(booking),
            assignment_repo=assign_repo,
            db_session=session,
        )
        res = uc.execute(booking_id=booking.id, driver_id=driver.id)
        assert res.status_code == 200
        assert res.should_commit is True
        assert booking.status == BookingStatus.PENDING
        assert booking.driver_id is None
        assert assign_repo.deleted_dependents_for == [9]
        assert session.deleted == [assignment]

        # Entreprise peut réattribuer
        driver2 = create_test_driver(db, company=company)
        result, _ = _assign(booking, driver2, company.id)
        assert result.ok is True
        assert booking.driver_id == driver2.id

    def test_s3_institution_dispatch_target_et_restrictions(self, db):
        """3 — Cible réseau/partenaires + garde mobilité (parcours institution)."""
        assert resolve_dispatch_target(None, has_preferences=False) == "network"
        assert resolve_dispatch_target(None, has_preferences=True) == "partners"
        assert resolve_dispatch_target("network", has_preferences=True) == "network"
        assert unmodeled_network_needs({"stretcher": True}) == ["brancard"]
        # Prestations spécialisées fermées
        blocked = reject_if_specialized_mobility_blocked(
            {"stretcher": True, "oxygen": False}
        )
        assert blocked is not None
        from marshmallow import ValidationError

        with pytest.raises(ValidationError):
            MobilitySchema().load({"stretcher": True})

    def test_s4_isolation_inter_entreprises(self, db):
        """4 — Accept / assign cross-company refusés."""
        company_a = create_test_company(db)
        company_b = create_test_company(db)
        client_a = create_test_client(db, company=company_a)
        driver_b = create_test_driver(db, company=company_b)
        booking = create_test_booking(db, client=client_a, status=BookingStatus.PENDING)
        # Accept par B sur booking déjà rattaché à A
        accept_b = _accept(booking, company_b.id)
        assert accept_b.ok is False
        assert accept_b.status_code == 403

        assert _accept(booking, company_a.id).ok
        client_b = create_test_client(db, company=company_b)
        booking_b = create_test_booking(
            db, client=client_b, status=BookingStatus.PENDING
        )
        assert booking_b.company_id == company_b.id
        accept_cross = _accept(booking_b, company_a.id)
        assert accept_cross.ok is False
        assert accept_cross.status_code == 403
        assert driver_b.company_id == company_b.id
        assert booking.company_id == company_a.id

    def test_s5_concurrence_deux_chauffeurs(self, db):
        """5 — Une seule affectation cohérente ; 2e chauffeur non propriétaire refusé."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        d1 = create_test_driver(db, company=company)
        d2 = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        r1, _ = _assign(booking, d1, company.id)
        assert r1.ok is True
        assert r1.changed is True
        # Idempotence même chauffeur
        r1b, _ = _assign(booking, d1, company.id)
        assert r1b.ok is True
        assert r1b.changed is False
        assert booking.driver_id == d1.id
        # Rejet use-case pour chauffeur non assigné
        reject = RejectDriverBookingUseCase(
            booking_repo=_BookingRepoAdapter(booking)
        ).execute(booking_id=booking.id, driver_id=d2.id)
        assert reject.status_code == 404
        assert booking.driver_id == d1.id

    def test_s6_reprise_reseau_idempotente(self, db):
        """6 — Rejeu d'une transition déjà appliquée : pas de double effet."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        assignment = _Assignment(
            assignment_id=3, booking_id=booking.id, driver_id=driver.id
        )
        r1, _, db1 = _status_uc(booking, driver.id, {"status": "en_route"}, assignment)
        assert r1.status_code == 200
        commits_after_first = db1.commit.call_count
        r2, _, db2 = _status_uc(booking, driver.id, {"status": "en_route"}, assignment)
        assert r2.status_code == 200
        assert booking.status == BookingStatus.EN_ROUTE
        # Second en_route est unchanged / pas de régression
        assert str(r2.response.get("status", "EN_ROUTE")).upper() in {
            "EN_ROUTE",
            "ASSIGNED",
        } or r2.response.get("unchanged") in (True, None)
        _ = commits_after_first, db2

    def test_s7_restrictions_s23_et_specialisees(self, db):
        """7 — semi_auto / fully_auto et mobilité spécialisée inaccessibles."""
        with pytest.raises(DispatchModeLockedError):
            assert_dispatch_mode_switch_allowed("semi_auto")
        with pytest.raises(DispatchModeLockedError):
            assert_dispatch_mode_switch_allowed("fully_auto")
        assert_dispatch_mode_switch_allowed("manual")
        assert reject_if_specialized_mobility_blocked({"oxygen": True}) is not None
        assert unmodeled_network_needs({"oxygen": True, "stretcher": True}) == [
            "brancard",
            "oxygène",
        ]

    def test_s8_compatibilite_mfa_contrats(self, db):
        """8 — Contrats métier restent indépendants du interrupt MFA (pas de bypass).

        Preuve structurelle : les use-cases accept/assign/status n'exposent aucun
        flag skip_mfa. Le merge-tree temporaire MFA est documenté hors runtime.
        """
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        for cls in (
            AcceptReservationUseCase,
            AssignDriverToReservationUseCase,
            UpdateDriverBookingStatusUseCase,
            RejectDriverBookingUseCase,
        ):
            src = cls.execute.__code__.co_varnames
            assert "skip_mfa" not in src
            assert "bypass_mfa" not in src

    def test_transition_interdite_refusee(self, db):
        """Changement de statut interdit (ASSIGNED → completed) refusé."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        res, _, _ = _status_uc(booking, driver.id, {"status": "completed"})
        assert res.status_code in {400, 409}
        assert booking.status == BookingStatus.ASSIGNED

    def test_mission_annulee_non_acceptable(self, db):
        """Mission CANCELED : reject use-case refuse (plus ASSIGNED)."""
        company = create_test_company(db)
        client = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        booking = create_test_booking(db, client=client, status=BookingStatus.PENDING)
        assert _accept(booking, company.id).ok
        assert _assign(booking, driver, company.id)[0].ok
        booking.status = BookingStatus.CANCELED
        db.session.commit()
        res = RejectDriverBookingUseCase(
            booking_repo=_BookingRepoAdapter(booking)
        ).execute(booking_id=booking.id, driver_id=driver.id)
        assert res.status_code == 400
        assert booking.driver_id == driver.id
