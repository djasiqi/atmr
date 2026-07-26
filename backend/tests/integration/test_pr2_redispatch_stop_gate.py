"""Tests stop-gate PR2 — révalidation transporteur & redispatch.

Couvre le workflow de révalidation des modifications critiques institution :
- Cas 1 : une modification critique sur une course acceptée crée une demande
  de validation (BookingChangeRequest) au lieu d'appliquer le patch.
- Cas 2 : le transporteur accepte → le patch est appliqué.
- Cas 3 : le transporteur refuse → la course est libérée pour redispatch.
- Cas 4 : une nouvelle modification supersede la demande PENDING précédente.
- Cas 5 : modification concurrente (verrou optimiste) → 409 (squelette).

Ces tests nécessitent PostgreSQL (JSONB, FK ON DELETE). Ils sont ignorés si le
schéma de test n'est pas disponible.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from models import (
    Booking,
    BookingChangeRequest,
    Institution,
    TransportRequest,
)
from models.booking_change_request import BookingChangeRequestStatus
from models.enums import BookingStatus, InstitutionRole, RequestStatus

# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"EMS Test {uuid.uuid4().hex[:6]}"
    inst.institution_type = "ems"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def committed_booking(db, test_company, test_client):
    """Course acceptée par un transporteur (engagé, pas encore démarrée)."""
    if not test_company or not test_client:
        pytest.skip("test_company and test_client required")

    booking = Booking()
    booking.user_id = test_client.user_id
    booking.company_id = test_company.id
    booking.client_id = test_client.id
    booking.customer_name = "Patient Stop-Gate"
    booking.pickup_location = "Rue A 1, 1200 Genève"
    booking.dropoff_location = "Hôpital B, 1205 Genève"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=3)
    booking.status = BookingStatus.ACCEPTED
    booking.amount = 50.0
    booking.edit_version = 1
    db.session.add(booking)
    db.session.flush()
    return booking


@pytest.fixture
def transport_request(db, institution, committed_booking):
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.scheduled_time = committed_booking.scheduled_time
    tr.pickup_location = committed_booking.pickup_location
    tr.dropoff_location = committed_booking.dropoff_location
    tr.status = RequestStatus.CONVERTED.value
    tr.booking_id = committed_booking.id
    tr.accepted_by_company_id = committed_booking.company_id
    db.session.add(tr)
    db.session.flush()
    return tr


def _make_ctx(booking, transport_request, institution_id):
    from services.institutions.booking_change_service import InstitutionBookingContext

    return InstitutionBookingContext(
        booking=booking,
        transport_request=transport_request,
        institution_id=institution_id,
    )


# ---------------------------------------------------------------------------
# Cas 1 — Création d'une demande de validation
# ---------------------------------------------------------------------------


class TestCas1RevalidationCreatesChangeRequest:
    def test_critical_change_creates_pending_change_request(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body, code = update_institution_booking(
            ctx,
            payload={
                "version": 1,
                "dropoff_location": "Nouvelle Clinique, 1206 Genève",
                "reason": "Changement de destination demandé par le service",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )

        assert code == 202, body
        assert body["status"] == "pending_revalidation"

        db.session.refresh(committed_booking)
        assert committed_booking.active_change_request_id is not None
        # Le patch n'est PAS appliqué immédiatement
        assert committed_booking.dropoff_location == "Hôpital B, 1205 Genève"

        cr = BookingChangeRequest.query.get(committed_booking.active_change_request_id)
        assert cr is not None
        assert cr.status == BookingChangeRequestStatus.PENDING
        assert "dropoff_location" in (cr.changed_fields or {})

    def test_minor_change_applies_directly(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body, code = update_institution_booking(
            ctx,
            payload={"version": 1, "notes_medical": "Note non critique"},
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )

        assert code == 200, body
        db.session.refresh(committed_booking)
        assert committed_booking.active_change_request_id is None
        assert committed_booking.notes_medical == "Note non critique"


# ---------------------------------------------------------------------------
# Cas 2 — Acceptation transporteur
# ---------------------------------------------------------------------------


class TestCas2CompanyAccepts:
    def test_accept_applies_patch(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from application.institutions.respond_to_change_request import (
            RespondToChangeRequestInput,
            RespondToChangeRequestUseCase,
        )
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body, code = update_institution_booking(
            ctx,
            payload={
                "version": 1,
                "dropoff_location": "Clinique X, 1206 Genève",
                "reason": "Nouvelle destination validée",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )
        assert code == 202
        change_id = body["change_request"]["id"]
        company_id = transport_request.accepted_by_company_id

        result = RespondToChangeRequestUseCase().execute(
            RespondToChangeRequestInput(
                booking_id=committed_booking.id,
                change_request_id=change_id,
                company_id=company_id,
                user_id=None,
                action="accept",
                version=1,
            )
        )

        assert result.success, result.error
        assert result.status == BookingChangeRequestStatus.ACCEPTED
        db.session.refresh(committed_booking)
        assert committed_booking.dropoff_location == "Clinique X, 1206 Genève"
        assert committed_booking.active_change_request_id is None


# ---------------------------------------------------------------------------
# Cas 3 — Refus transporteur → libération / redispatch
# ---------------------------------------------------------------------------


class TestCas3CompanyRefuses:
    def test_refuse_releases_booking(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from application.institutions.respond_to_change_request import (
            RespondToChangeRequestInput,
            RespondToChangeRequestUseCase,
        )
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body, code = update_institution_booking(
            ctx,
            payload={
                "version": 1,
                "scheduled_time": (datetime.now(UTC) + timedelta(hours=6)).isoformat(),
                "reason": "Décalage horaire important",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )
        assert code == 202
        change_id = body["change_request"]["id"]
        company_id = transport_request.accepted_by_company_id

        result = RespondToChangeRequestUseCase().execute(
            RespondToChangeRequestInput(
                booking_id=committed_booking.id,
                change_request_id=change_id,
                company_id=company_id,
                user_id=None,
                action="refuse",
                version=1,
                reason="Indisponible au nouvel horaire",
            )
        )

        assert result.success, result.error
        assert result.status == BookingChangeRequestStatus.REFUSED
        db.session.refresh(committed_booking)
        # Course libérée → repassée en PENDING, transporteur détaché
        assert committed_booking.status == BookingStatus.PENDING
        assert committed_booking.company_id is None
        assert committed_booking.active_change_request_id is None


# ---------------------------------------------------------------------------
# Cas 4 — Supersession d'une demande PENDING
# ---------------------------------------------------------------------------


class TestCas4Supersede:
    def test_second_change_supersedes_first(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body1, code1 = update_institution_booking(
            ctx,
            payload={
                "version": 1,
                "dropoff_location": "Destination 1, 1207 Genève",
                "reason": "Première modification",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )
        assert code1 == 202
        first_id = body1["change_request"]["id"]

        ctx2 = _make_ctx(committed_booking, transport_request, institution.id)
        body2, code2 = update_institution_booking(
            ctx2,
            payload={
                "version": 1,
                "pickup_location": "Nouveau départ, 1208 Genève",
                "reason": "Deuxième modification",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )
        assert code2 == 202
        second_id = body2["change_request"]["id"]

        assert first_id != second_id
        first = BookingChangeRequest.query.get(first_id)
        assert first.status == BookingChangeRequestStatus.SUPERSEDED
        db.session.refresh(committed_booking)
        assert committed_booking.active_change_request_id == second_id


# ---------------------------------------------------------------------------
# Cas 5 — Modification concurrente (verrou optimiste) — SQUELETTE
# ---------------------------------------------------------------------------


class TestCas5ConcurrentModification:
    """Squelette : deux réponses concurrentes sur la même demande de validation.

    La première réponse incrémente la version de la BCR et clôt la demande
    (active_change_request_id remis à None). La seconde réponse, utilisant la
    version périmée / la demande non-active, doit être rejetée en 409.
    """

    def test_stale_version_is_rejected_409(
        self, db, requires_postgresql, committed_booking, transport_request, institution
    ):
        from application.institutions.respond_to_change_request import (
            RespondToChangeRequestInput,
            RespondToChangeRequestUseCase,
        )
        from services.institutions.booking_change_service import (
            update_institution_booking,
        )

        ctx = _make_ctx(committed_booking, transport_request, institution.id)
        body, code = update_institution_booking(
            ctx,
            payload={
                "version": 1,
                "dropoff_location": "Concurrent Dest, 1209 Genève",
                "reason": "Modification concurrente",
            },
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
        )
        assert code == 202
        change_id = body["change_request"]["id"]
        company_id = transport_request.accepted_by_company_id

        uc = RespondToChangeRequestUseCase()

        # 1ère réponse (accept) — réussit, version 1 -> 2, demande clôturée
        first = uc.execute(
            RespondToChangeRequestInput(
                booking_id=committed_booking.id,
                change_request_id=change_id,
                company_id=company_id,
                user_id=None,
                action="accept",
                version=1,
            )
        )
        assert first.success

        # 2ème réponse concurrente avec version périmée → 409
        second = uc.execute(
            RespondToChangeRequestInput(
                booking_id=committed_booking.id,
                change_request_id=change_id,
                company_id=company_id,
                user_id=None,
                action="refuse",
                version=1,
            )
        )
        assert not second.success
        assert second.status_code == 409

    @pytest.mark.skip(reason="Squelette : exécution réellement parallèle (threads/DB)")
    def test_truly_parallel_responses(self):
        """À compléter : deux transactions simultanées avec SELECT FOR UPDATE.

        Objectif : vérifier que le verrou pessimiste sur Booking sérialise les
        réponses et qu'une seule gagne (l'autre obtient 409). Nécessite deux
        connexions DB concurrentes (threads + sessions distinctes).
        """
        pytest.skip("À implémenter avec connexions concurrentes réelles")


# ---------------------------------------------------------------------------
# Annulation cascade d'un parcours multi-destinations
# ---------------------------------------------------------------------------


class TestCancelMultiStopCascade:
    """Annuler le booking principal annule aussi les legs liés du parcours."""

    def _make_leg(
        self,
        db,
        test_company,
        test_client,
        *,
        route_group_id,
        seq,
        pickup,
        dropoff,
        parent_id=None,
        scheduled=None,
        confirmed=True,
    ):
        leg = Booking()
        leg.user_id = test_client.user_id
        leg.company_id = test_company.id
        leg.client_id = test_client.id
        leg.customer_name = "Patient Multi-Stop"
        leg.pickup_location = pickup
        leg.dropoff_location = dropoff
        leg.scheduled_time = scheduled
        leg.time_confirmed = confirmed
        leg.status = BookingStatus.ACCEPTED
        leg.amount = 0.5
        leg.edit_version = 1
        leg.route_group_id = route_group_id
        leg.route_sequence_number = seq
        if parent_id is not None:
            leg.parent_booking_id = parent_id
        db.session.add(leg)
        db.session.flush()
        return leg

    def test_cancel_principal_cascades_to_all_legs(
        self, db, requires_postgresql, test_company, test_client, institution
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        from services.institutions.booking_change_service import (
            cancel_institution_booking,
        )

        route_group_id = str(uuid.uuid4())
        base_time = datetime.now(UTC) + timedelta(hours=3)

        principal = self._make_leg(
            db,
            test_company,
            test_client,
            route_group_id=route_group_id,
            seq=1,
            pickup="Clinique",
            dropoff="HUG",
            scheduled=base_time,
        )
        leg2 = self._make_leg(
            db,
            test_company,
            test_client,
            route_group_id=route_group_id,
            seq=2,
            pickup="HUG",
            dropoff="Dr Lakki",
            scheduled=base_time.replace(hour=0, minute=0),
            confirmed=False,
        )
        leg3 = self._make_leg(
            db,
            test_company,
            test_client,
            route_group_id=route_group_id,
            seq=3,
            pickup="Dr Lakki",
            dropoff="Clinique",
            scheduled=base_time.replace(hour=0, minute=0),
            confirmed=False,
        )

        tr = TransportRequest()
        tr.public_id = str(uuid.uuid4())
        tr.institution_id = institution.id
        tr.scheduled_time = base_time
        tr.pickup_location = principal.pickup_location
        tr.dropoff_location = principal.dropoff_location
        tr.status = RequestStatus.CONVERTED.value
        tr.booking_id = principal.id
        tr.accepted_by_company_id = principal.company_id
        tr.route_group_id = route_group_id
        db.session.add(tr)
        db.session.flush()

        ctx = _make_ctx(principal, tr, institution.id)
        body, code = cancel_institution_booking(
            ctx,
            reason="Client a demandé l'annulation",
            reason_code="CLIENT_REQUEST",
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin Test",
            client_version=1,
        )

        assert code == 200, body
        assert set(body["cancelled_linked_booking_ids"]) == {leg2.id, leg3.id}

        for leg in (principal, leg2, leg3):
            db.session.refresh(leg)
            assert leg.status == BookingStatus.CANCELED
        # Annulation anticipée (3h avant) : pas de facturation sur le principal
        assert principal.is_cancellation_billable is False
        # Les legs liés ne sont pas refacturés (facturation portée par le principal)
        assert leg2.is_cancellation_billable is False
        assert leg3.is_cancellation_billable is False
