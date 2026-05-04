# tests/test_etape5_cancellation_audit.py
# ruff: noqa: I001
"""Tests pour ÉTAPE 5: Annulation, frais EN_ROUTE, audit immuable.

Ce module teste:
- Annulation TransportRequest convertie -> 409
- Annulation Booking EN_ROUTE -> frais dus (billable)
- Permissions facturation institution
- Audit logs immuables (trigger DB)
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_access_token

from application.bookings.cancellation_rules import (
    compute_cancellation_fields_with_status,
    get_cancellation_billing_info,
    is_booking_cancellable,
    is_status_billable_cancellation,
)
from models import (
    Booking,
    BookingStatus,
    Company,
    Institution,
    InstitutionPatient,
    RequestStatus,
    TransportRequest,
    User,
    UserRole,
)


class TestCancellationRulesEnRoute:
    """Tests pour les règles d'annulation basées sur le statut."""

    def test_pending_is_cancellable(self):
        """Test: PENDING est annulable."""
        assert is_booking_cancellable("PENDING") is True

    def test_accepted_is_cancellable(self):
        """Test: ACCEPTED est annulable."""
        assert is_booking_cancellable("ACCEPTED") is True

    def test_en_route_is_cancellable(self):
        """Test: EN_ROUTE est annulable (mais avec frais)."""
        assert is_booking_cancellable("EN_ROUTE") is True

    def test_in_progress_is_cancellable(self):
        """Test: IN_PROGRESS est annulable (mais avec frais)."""
        assert is_booking_cancellable("IN_PROGRESS") is True

    def test_completed_not_cancellable(self):
        """Test: COMPLETED n'est pas annulable."""
        assert is_booking_cancellable("COMPLETED") is False

    def test_canceled_not_cancellable(self):
        """Test: CANCELED n'est pas annulable (déjà annulé)."""
        assert is_booking_cancellable("CANCELED") is False

    def test_en_route_is_always_billable(self):
        """Test: EN_ROUTE est toujours facturable."""
        assert is_status_billable_cancellation("EN_ROUTE") is True

    def test_in_progress_is_always_billable(self):
        """Test: IN_PROGRESS est toujours facturable."""
        assert is_status_billable_cancellation("IN_PROGRESS") is True

    def test_pending_not_always_billable(self):
        """Test: PENDING n'est pas automatiquement facturable."""
        assert is_status_billable_cancellation("PENDING") is False

    def test_billing_info_en_route(self):
        """Test: Infos facturation pour EN_ROUTE."""
        info = get_cancellation_billing_info("EN_ROUTE", "OTHER")

        assert info["is_billable"] is True
        assert info["billing_reason"] == "status_en_route"
        assert "déplacement" in info["billing_description"].lower()

    def test_billing_info_in_progress(self):
        """Test: Infos facturation pour IN_PROGRESS avec surcharge."""
        info = get_cancellation_billing_info("IN_PROGRESS", "OTHER")

        assert info["is_billable"] is True
        assert info["billing_reason"] == "status_in_progress"
        assert info["surcharge_percent"] == 100

    def test_billing_info_pending_with_billable_reason(self):
        """Test: PENDING avec motif billable."""
        info = get_cancellation_billing_info("PENDING", "NO_SHOW")

        assert info["is_billable"] is True
        assert info["billing_reason"] == "reason_code"

    def test_billing_info_pending_with_non_billable_reason(self):
        """Test: PENDING avec motif non billable."""
        info = get_cancellation_billing_info("PENDING", "COMPANY_ISSUE")

        assert info["is_billable"] is False
        assert info["billing_reason"] == "none"

    def test_compute_fields_en_route_overrides_reason(self):
        """Test: EN_ROUTE override le motif non-billable."""
        fields = compute_cancellation_fields_with_status(
            booking_status="EN_ROUTE",
            reason_code="COMPANY_ISSUE",  # Normalement non-billable
            reason_text=None,
            cancelled_by_role="institution",
        )

        # Le statut EN_ROUTE force la facturation
        assert fields["is_cancellation_billable"] is True
        assert fields["billing_info"]["billing_reason"] == "status_en_route"


class TestCancelConvertedRequest:
    """Tests pour l'annulation de demandes converties."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Cancel Test"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_user(self, db, sample_institution):
        """Crée un utilisateur institution admin."""
        user = User()
        user.email = f"admin_{uuid.uuid4().hex[:8]}@test.com"
        user.password_hash = "test"
        user.role = UserRole.INSTITUTION.value
        user.institution_id = sample_institution.id
        user.institution_role = "institution_admin"
        db.session.add(user)
        db.session.flush()
        return user

    @pytest.fixture
    def auth_headers(self, sample_user, sample_institution):
        """Headers JWT pour l'utilisateur institution."""
        token = create_access_token(
            identity=sample_user.id,
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "institution_id": sample_institution.id,
                "institution_role": "institution_admin",
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def converted_request(self, db, sample_institution):
        """Crée une demande de transport CONVERTED avec un booking."""
        # Créer un booking minimal
        # Note: Booking requires company_id, client_id, etc.
        # Pour ce test simplifié, on crée juste la TransportRequest
        transport_req = TransportRequest()
        transport_req.institution_id = sample_institution.id
        transport_req.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        transport_req.pickup_location = "123 Rue Test"
        transport_req.dropoff_location = "456 Avenue Dest"
        transport_req.scheduled_time = datetime.now(UTC) + timedelta(days=2)
        transport_req.status = RequestStatus.CONVERTED.value
        transport_req.booking_id = 99999  # ID fictif pour le test
        db.session.add(transport_req)
        db.session.flush()
        return transport_req

    def test_cancel_converted_returns_409(
        self, client, db, sample_institution, auth_headers, converted_request
    ):
        """Test: Annuler une demande CONVERTED retourne 409 avec booking_id."""
        response = client.post(
            f"/api/v1/institutions/requests/{converted_request.id}/cancel",
            headers=auth_headers,
        )

        assert response.status_code == 409
        data = response.get_json()

        assert "resulting_booking_id" in data
        assert data["resulting_booking_id"] == converted_request.booking_id
        assert (
            "convertie" in data["error"].lower() or "booking" in data["error"].lower()
        )

    def test_cancel_draft_request_succeeds(
        self, client, db, sample_institution, auth_headers
    ):
        """Test: Annuler une demande DRAFT réussit."""
        # Créer une demande DRAFT
        transport_req = TransportRequest()
        transport_req.institution_id = sample_institution.id
        transport_req.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        transport_req.pickup_location = "123 Rue Test"
        transport_req.dropoff_location = "456 Avenue Dest"
        transport_req.scheduled_time = datetime.now(UTC) + timedelta(days=2)
        transport_req.status = RequestStatus.DRAFT.value
        db.session.add(transport_req)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{transport_req.id}/cancel",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == RequestStatus.CANCELLED.value


class TestBillingPermissions:
    """Tests pour les permissions de facturation institution."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Billing Test"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def billing_user(self, db, sample_institution):
        """Crée un utilisateur institution avec rôle BILLING."""
        user = User()
        user.email = f"billing_{uuid.uuid4().hex[:8]}@test.com"
        user.password_hash = "test"
        user.role = UserRole.INSTITUTION.value
        user.institution_id = sample_institution.id
        user.institution_role = "institution_billing"
        db.session.add(user)
        db.session.flush()
        return user

    @pytest.fixture
    def reader_user(self, db, sample_institution):
        """Crée un utilisateur institution avec rôle READER (non autorisé)."""
        user = User()
        user.email = f"reader_{uuid.uuid4().hex[:8]}@test.com"
        user.password_hash = "test"
        user.role = UserRole.INSTITUTION.value
        user.institution_id = sample_institution.id
        user.institution_role = "institution_reader"
        db.session.add(user)
        db.session.flush()
        return user

    @pytest.fixture
    def billing_auth_headers(self, billing_user, sample_institution):
        """Headers JWT pour l'utilisateur billing."""
        token = create_access_token(
            identity=billing_user.id,
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "institution_id": sample_institution.id,
                "institution_role": "institution_billing",
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def reader_auth_headers(self, reader_user, sample_institution):
        """Headers JWT pour l'utilisateur reader."""
        token = create_access_token(
            identity=reader_user.id,
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "institution_id": sample_institution.id,
                "institution_role": "institution_reader",
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def sample_request(self, db, sample_institution):
        """Crée une demande de transport de test."""
        transport_req = TransportRequest()
        transport_req.institution_id = sample_institution.id
        transport_req.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        transport_req.pickup_location = "123 Rue Test"
        transport_req.dropoff_location = "456 Avenue Dest"
        transport_req.scheduled_time = datetime.now(UTC) + timedelta(days=2)
        transport_req.status = RequestStatus.SENT.value
        transport_req.billing_intent = "patient"
        db.session.add(transport_req)
        db.session.flush()
        return transport_req

    def test_billing_user_can_update_request_billing(
        self, client, db, sample_institution, billing_auth_headers, sample_request
    ):
        """Test: Utilisateur billing peut modifier la facturation d'une request."""
        response = client.put(
            f"/api/v1/institutions/billing/requests/{sample_request.id}",
            headers=billing_auth_headers,
            json={"billing_intent": "institution"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["billing_intent"] == "institution"

    def test_reader_user_cannot_update_request_billing(
        self, client, db, sample_institution, reader_auth_headers, sample_request
    ):
        """Test: Utilisateur reader ne peut pas modifier la facturation."""
        response = client.put(
            f"/api/v1/institutions/billing/requests/{sample_request.id}",
            headers=reader_auth_headers,
            json={"billing_intent": "institution"},
        )

        assert response.status_code == 403


class TestAuditLogsImmutable:
    """Tests pour l'immuabilité des audit logs.

    Note: Ces tests nécessitent que la migration soit appliquée.
    Ils vérifient que les triggers PostgreSQL bloquent les UPDATE/DELETE.
    """

    def test_audit_log_insert_works(self, db):
        """Test: INSERT dans audit_logs fonctionne."""
        from security.audit_log import AuditLogger

        # Doit fonctionner sans erreur
        AuditLogger.log_action(
            action_type="test_insert",
            action_category="test",
            result_status="success",
            action_details={"test": True},
        )

        # Commit pour persister
        db.session.commit()

    @pytest.mark.skip(reason="Nécessite migration appliquée avec triggers")
    def test_audit_log_update_blocked(self, db):
        """Test: UPDATE sur audit_logs est bloqué par trigger.

        Ce test est skip par défaut car il nécessite que la migration
        soit appliquée. À activer après `flask db upgrade`.
        """
        from sqlalchemy import text
        from sqlalchemy.exc import InternalError

        # Créer un log
        from security.audit_log import AuditLogger

        AuditLogger.log_action(
            action_type="test_update_blocked",
            action_category="test",
            result_status="success",
            action_details={"test": True},
        )
        db.session.commit()

        # Tenter un UPDATE - doit échouer
        # Note: PostgreSQL trigger lève l'exception lors de l'execute,
        # le commit n'est jamais atteint
        with pytest.raises(InternalError) as exc_info:
            db.session.execute(
                text(
                    "UPDATE audit_logs SET result_status = 'failure' WHERE action_type = 'test_update_blocked'"
                )
            )

        db.session.rollback()  # Cleanup après l'erreur
        assert (
            "immutable" in str(exc_info.value).lower()
            or "not allowed" in str(exc_info.value).lower()
        )

    @pytest.mark.skip(reason="Nécessite migration appliquée avec triggers")
    def test_audit_log_delete_blocked(self, db):
        """Test: DELETE sur audit_logs est bloqué par trigger.

        Ce test est skip par défaut car il nécessite que la migration
        soit appliquée. À activer après `flask db upgrade`.
        """
        from sqlalchemy import text
        from sqlalchemy.exc import InternalError

        # Créer un log
        from security.audit_log import AuditLogger

        AuditLogger.log_action(
            action_type="test_delete_blocked",
            action_category="test",
            result_status="success",
            action_details={"test": True},
        )
        db.session.commit()

        # Tenter un DELETE - doit échouer
        # Note: PostgreSQL trigger lève l'exception lors de l'execute,
        # le commit n'est jamais atteint
        with pytest.raises(InternalError) as exc_info:
            db.session.execute(
                text("DELETE FROM audit_logs WHERE action_type = 'test_delete_blocked'")
            )

        db.session.rollback()  # Cleanup après l'erreur
        assert (
            "immutable" in str(exc_info.value).lower()
            or "not allowed" in str(exc_info.value).lower()
        )


class TestInstitutionEvents:
    """Tests pour les événements Socket.IO institution."""

    def test_emit_request_sent(self):
        """Test: emit_request_sent fonctionne."""
        from services.events.institution_events import emit_request_sent

        with patch("services.events.institution_events.socketio") as mock_socketio:
            result = emit_request_sent(
                institution_id=1,
                request_id=100,
                public_id="abc-123",
                external_reference="REF-001",
                mode="sequential",
                offers_created=1,
            )

            assert result is True
            mock_socketio.emit.assert_called_once()
            call_args = mock_socketio.emit.call_args
            assert call_args[0][0] == "request_sent"
            assert call_args[1]["room"] == "institution_1"

    def test_emit_booking_status_updated(self):
        """Test: emit_booking_status_updated fonctionne."""
        from services.events.institution_events import emit_booking_status_updated

        with patch("services.events.institution_events.socketio") as mock_socketio:
            result = emit_booking_status_updated(
                institution_id=1,
                booking_id=200,
                request_id=100,
                public_id="abc-123",
                old_status="ACCEPTED",
                new_status="EN_ROUTE",
            )

            assert result is True
            mock_socketio.emit.assert_called_once()
            call_args = mock_socketio.emit.call_args
            assert call_args[0][0] == "booking_status_updated"

    def test_get_institution_from_booking(self, db):
        """Test: get_institution_from_booking retrouve l'institution."""
        from services.events.institution_events import get_institution_from_booking

        # Sans TransportRequest liée, retourne None
        result = get_institution_from_booking(99999)
        assert result is None
