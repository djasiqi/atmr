# tests/routes/test_request_offers.py
# ruff: noqa: I001
"""Tests pour les offres de transport institutionnelles (ÉTAPE 4).

Ce module teste:
- Création d'offres via send (séquentiel vs broadcast)
- Acceptation atomique (first accept wins)
- Rejet avec escalade
- Expiration et fallback
- Conversion en Booking
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token

from models import (
    Booking,
    Company,
    Institution,
    InstitutionPatient,
    InstitutionTransportPreference,
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    TransportRequestLeg,
    User,
    UserRole,
)
from models.institution_api_key import InstitutionApiKey, generate_api_key


class TestSendWithOffers:
    """Tests pour l'envoi de demandes avec création d'offres."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Test Offers"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_user(self, db, sample_institution):
        """Crée un utilisateur institution admin."""
        user = User()
        user.email = f"admin_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
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
            identity=str(sample_user.public_id),
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "aud": "atmr-api",
                "user_id": sample_user.id,
                "institution_id": sample_institution.id,
                "institution_role": "institution_admin",
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def sample_company(self, db):
        """Crée une entreprise de test éligible."""
        user = User()
        user.email = f"company_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
        user.role = UserRole.COMPANY.value
        db.session.add(user)
        db.session.flush()

        company = Company()
        company.name = "Transport Test"
        company.user_id = user.id
        company.is_approved = True
        company.dispatch_enabled = True
        db.session.add(company)
        db.session.flush()
        return company

    @pytest.fixture
    def sample_company_2(self, db):
        """Crée une deuxième entreprise de test."""
        user = User()
        user.email = f"company2_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
        user.role = UserRole.COMPANY.value
        db.session.add(user)
        db.session.flush()

        company = Company()
        company.name = "Transport Test 2"
        company.user_id = user.id
        company.is_approved = True
        company.dispatch_enabled = True
        db.session.add(company)
        db.session.flush()
        return company

    @pytest.fixture
    def sample_request(self, db, sample_institution):
        """Crée une demande de transport de test."""
        scheduled = datetime.now(UTC) + timedelta(days=2)
        request = TransportRequest()
        request.institution_id = sample_institution.id
        request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        request.pickup_location = "123 Rue Test"
        request.dropoff_location = "456 Avenue Dest"
        request.mission_date = scheduled.date()
        request.scheduled_time = scheduled
        request.pickup_time_confirmed = True
        request.status = RequestStatus.DRAFT.value
        db.session.add(request)
        db.session.flush()
        return request

    def test_send_without_preferences_creates_broadcast(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Test: Sans préférences, send crée des offres broadcast."""
        # Vérifier pas de préférences
        assert not InstitutionTransportPreference.has_preferences(sample_institution.id)

        # Envoyer la demande
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )

        assert response.status_code == 200, response.get_json()
        data = response.get_json()

        # Vérifier les infos d'envoi
        assert "send_info" in data
        assert data["send_info"]["mode"] == OfferMode.BROADCAST.value
        assert data["send_info"]["offers_created"] >= 1

        # Vérifier la demande
        db.session.refresh(sample_request)
        assert sample_request.status == RequestStatus.SENT.value
        assert sample_request.sent_at is not None

        # Vérifier l'offre créée
        offers = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id
        ).all()
        assert len(offers) >= 1
        assert all(o.mode == OfferMode.BROADCAST.value for o in offers)
        assert all(o.status == OfferStatus.PENDING.value for o in offers)

    def test_send_with_preferences_creates_sequential(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
        sample_company_2,
    ):
        """Test: Avec préférences, send crée une offre séquentielle (première préférence)."""
        # Définir des préférences
        InstitutionTransportPreference.set_preferences(
            institution_id=sample_institution.id,
            company_ids=[sample_company.id, sample_company_2.id],
        )
        db.session.commit()

        # Envoyer la demande
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )

        assert response.status_code == 200, response.get_json()
        data = response.get_json()

        # Vérifier mode séquentiel
        assert data["send_info"]["mode"] == OfferMode.SEQUENTIAL.value
        assert data["send_info"]["offers_created"] == 1

        # Vérifier l'offre créée (première préférence seulement)
        offers = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id
        ).all()
        assert len(offers) == 1
        assert offers[0].company_id == sample_company.id
        assert offers[0].mode == OfferMode.SEQUENTIAL.value
        assert offers[0].order == 1
        assert offers[0].expires_at is not None

    def test_send_already_sent_is_idempotent(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Test GO-LIVE: Renvoyer une demande SENT avec offres PENDING retourne 200 (idempotent)."""
        # Première envoi
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200
        first_offers = response.json.get("send_info", {}).get("offers_created", 0)

        # Deuxième envoi -> 200 (idempotent, pas 409)
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200, (
            f"Expected idempotent 200, got {response.status_code}"
        )

        # Vérifier que le nombre d'offres est cohérent
        second_offers = response.json.get("send_info", {}).get("offers_created", 0)
        assert second_offers == first_offers, (
            "Idempotent send should return same offer count"
        )

    def test_send_reactivates_time_expired_pending_offers(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Relance : offres PENDING expirées dans le temps sont réactivées avec un nouveau délai."""
        pref = InstitutionTransportPreference()
        pref.institution_id = sample_institution.id
        pref.company_id = sample_company.id
        pref.order = 1
        db.session.add(pref)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200

        offer = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id,
            company_id=sample_company.id,
        ).first()
        assert offer is not None
        old_expires_at = offer.expires_at

        offer.expires_at = datetime.now(UTC) - timedelta(hours=2)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200

        db.session.refresh(offer)
        assert offer.status == OfferStatus.PENDING.value
        assert offer.expires_at is not None
        refreshed_expires = offer.expires_at
        if refreshed_expires.tzinfo is None:
            refreshed_expires = refreshed_expires.replace(tzinfo=UTC)
        assert refreshed_expires > datetime.now(UTC)
        assert offer.expires_at != old_expires_at

    def test_dispatch_can_relaunch_when_only_time_expired_pending(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Liste institution : relance possible si seules des offres PENDING expirées existent."""
        pref = InstitutionTransportPreference()
        pref.institution_id = sample_institution.id
        pref.company_id = sample_company.id
        pref.order = 1
        db.session.add(pref)
        db.session.commit()

        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200

        offer = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id,
            company_id=sample_company.id,
        ).first()
        offer.expires_at = datetime.now(UTC) - timedelta(minutes=30)
        db.session.commit()

        list_response = client.get(
            "/api/v1/institutions/requests",
            headers=auth_headers,
        )
        assert list_response.status_code == 200
        requests = list_response.get_json().get("requests", [])
        row = next((r for r in requests if r["id"] == sample_request.id), None)
        assert row is not None
        assert row["dispatch"]["has_pending_offers"] is False
        assert row["dispatch"]["has_only_expired_pending"] is True
        assert row["dispatch"]["can_relaunch"] is True

    @patch("ext.socketio.emit")
    @patch("services.events.institution_events.persist_company_notification")
    def test_send_relaunch_notifies_company_with_relaunch_dedupe(
        self,
        mock_notify,
        _mock_socket_emit,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Relance : notification entreprise avec clé de déduplication distincte."""
        mock_notify.return_value = {"id": 1}
        pref = InstitutionTransportPreference()
        pref.institution_id = sample_institution.id
        pref.company_id = sample_company.id
        pref.order = 1
        db.session.add(pref)
        db.session.commit()

        client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )

        offer = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id,
            company_id=sample_company.id,
        ).first()
        offer.status = OfferStatus.EXPIRED.value
        db.session.commit()

        mock_notify.reset_mock()
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert mock_notify.called
        kwargs = mock_notify.call_args.kwargs
        assert kwargs["company_id"] == sample_company.id
        assert kwargs["title"] == "Demande de transport relancée"
        assert ":relaunch:" in kwargs["dedupe_key"]

    @patch("ext.socketio.emit")
    @patch("services.events.institution_events.persist_company_notification")
    def test_send_relaunch_broadcasts_to_all_eligible_companies(
        self,
        mock_notify,
        _mock_socket_emit,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
        sample_company_2,
    ):
        """Relance : réactive l'offre expirée et contacte toutes les entreprises éligibles."""
        mock_notify.return_value = {"id": 1}
        InstitutionTransportPreference.set_preferences(
            institution_id=sample_institution.id,
            company_ids=[sample_company.id, sample_company_2.id],
        )
        db.session.commit()

        client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )

        offer1 = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id,
            company_id=sample_company.id,
        ).first()
        assert offer1 is not None
        old_expires = datetime.now(UTC) - timedelta(minutes=30)
        offer1.status = OfferStatus.EXPIRED.value
        offer1.expires_at = old_expires
        db.session.commit()

        mock_notify.reset_mock()
        before_relaunch = datetime.now(UTC)
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["send_info"]["mode"] == OfferMode.BROADCAST.value
        assert data["send_info"]["offers_created"] >= 2

        db.session.refresh(offer1)
        assert offer1.status == OfferStatus.PENDING.value
        assert offer1.expires_at is not None
        assert offer1.expires_at > before_relaunch

        offer2 = RequestOffer.query.filter_by(
            transport_request_id=sample_request.id,
            company_id=sample_company_2.id,
        ).first()
        assert offer2 is not None
        assert offer2.status == OfferStatus.PENDING.value
        assert offer2.expires_at is not None
        assert offer2.expires_at > before_relaunch

        notified_company_ids = {
            call.kwargs["company_id"] for call in mock_notify.call_args_list
        }
        assert sample_company.id in notified_company_ids
        assert sample_company_2.id in notified_company_ids

    def test_send_converted_request_fails_409(
        self,
        client,
        db,
        sample_institution,
        auth_headers,
        sample_request,
        sample_company,
    ):
        """Test GO-LIVE: Envoyer une demande CONVERTED retourne 409."""
        # Simuler une demande déjà convertie
        sample_request.status = RequestStatus.CONVERTED.value
        sample_request.booking_id = 12345  # Booking fictif
        db.session.commit()

        # Tentative d'envoi -> 409
        response = client.post(
            f"/api/v1/institutions/requests/{sample_request.id}/send",
            headers=auth_headers,
        )
        assert response.status_code == 409, (
            f"Expected 409 for CONVERTED, got {response.status_code}"
        )
        assert "convertie" in response.json.get("error", "").lower()


class TestAcceptOffer:
    """Tests pour l'acceptation d'offres."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Accept Test"
        institution.public_id = str(uuid.uuid4())
        institution.billing_address = "123 Rue Test, 1200 Genève"
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_company(self, db):
        """Crée une entreprise de test."""
        user = User()
        user.email = f"company_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
        user.role = UserRole.COMPANY.value
        db.session.add(user)
        db.session.flush()

        company = Company()
        company.name = "Transport Accept Test"
        company.user_id = user.id
        company.is_approved = True
        company.dispatch_enabled = True
        db.session.add(company)
        db.session.flush()
        return company, user

    @pytest.fixture
    def sample_company_2(self, db):
        """Crée une deuxième entreprise de test."""
        user = User()
        user.email = f"company2_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
        user.role = UserRole.COMPANY.value
        db.session.add(user)
        db.session.flush()

        company = Company()
        company.name = "Transport Accept Test 2"
        company.user_id = user.id
        company.is_approved = True
        company.dispatch_enabled = True
        db.session.add(company)
        db.session.flush()
        return company, user

    @pytest.fixture
    def company_auth_headers(self, sample_company):
        """Headers JWT pour l'entreprise."""
        company, user = sample_company
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": UserRole.COMPANY.value,
                "aud": "atmr-api",
                "user_id": user.id,
                "company_id": company.id,
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def company_2_auth_headers(self, sample_company_2):
        """Headers JWT pour la deuxième entreprise."""
        company, user = sample_company_2
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": UserRole.COMPANY.value,
                "aud": "atmr-api",
                "user_id": user.id,
                "company_id": company.id,
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def sample_request_with_offer(self, db, sample_institution, sample_company):
        """Crée une demande avec une offre PENDING."""
        company, _user = sample_company

        request = TransportRequest()
        request.institution_id = sample_institution.id
        request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        request.pickup_location = "123 Rue Test"
        request.dropoff_location = "456 Avenue Dest"
        scheduled = datetime.now(UTC) + timedelta(days=2)
        request.mission_date = scheduled.date()
        request.scheduled_time = scheduled
        request.pickup_time_confirmed = True
        request.status = RequestStatus.SENT.value
        request.sent_at = datetime.now(UTC)
        db.session.add(request)
        db.session.flush()

        offer = RequestOffer(
            transport_request_id=request.id,
            company_id=company.id,
            mode=OfferMode.BROADCAST.value,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=2),
        )
        db.session.add(offer)
        db.session.flush()

        return request, offer

    def test_accept_offer_creates_booking(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
        sample_company,
    ):
        """Test: Accepter une offre crée un booking."""
        request, offer = sample_request_with_offer
        company, _user = sample_company

        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["success"] is True
        assert data["booking_id"] is not None

        # Vérifier l'offre
        db.session.refresh(offer)
        assert offer.status == OfferStatus.ACCEPTED.value
        assert offer.responded_at is not None

        # Vérifier la demande
        db.session.refresh(request)
        assert request.status == RequestStatus.CONVERTED.value
        assert request.accepted_by_company_id == company.id
        assert request.booking_id == data["booking_id"]

        # Vérifier le booking
        booking = Booking.query.get(data["booking_id"])
        assert booking is not None
        assert booking.company_id == company.id
        assert booking.pickup_location == request.pickup_location

    def test_accept_multi_stop_offer_first_leg_timed_others_to_define(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
    ):
        """Multi-destinations : leg 0 (A-B) a une heure, les suivants restent à définir."""
        request, offer = sample_request_with_offer
        request.multi_stop = True
        request.return_to_institution = True
        request.route_group_id = str(uuid.uuid4())
        request.dropoff_location = "Retour institution"
        # Départ confirmé : leg 0 (A-B) hérite de l'heure mission, legs suivants à définir.
        request.pickup_time_confirmed = True

        legs = [
            TransportRequestLeg(
                transport_request_id=request.id,
                sequence_index=0,
                route_sequence_number=1,
                pickup_location="Clinique",
                dropoff_location="HUG",
                scheduled_time=None,
            ),
            TransportRequestLeg(
                transport_request_id=request.id,
                sequence_index=1,
                route_sequence_number=2,
                pickup_location="HUG",
                dropoff_location="Urgences ophtalmologie HUG",
                scheduled_time=None,
            ),
            TransportRequestLeg(
                transport_request_id=request.id,
                sequence_index=2,
                route_sequence_number=3,
                pickup_location="Urgences ophtalmologie HUG",
                dropoff_location="Clinique",
                scheduled_time=None,
            ),
        ]
        db.session.add_all(legs)
        db.session.commit()

        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
        )

        assert response.status_code == 200, response.get_json()
        created_bookings = Booking.query.filter_by(
            route_group_id=request.route_group_id
        ).all()
        assert len(created_bookings) == 3

        by_seq = {b.route_sequence_number: b for b in created_bookings}
        # Leg 0 (A-B) : heure obligatoire, confirmée, course "aller" classique.
        first = by_seq[1]
        assert first.scheduled_time is not None
        assert first.time_confirmed is True
        assert first.is_return is False
        # Legs suivants (B-C, C-A) : heure à définir, scheduled_time=null.
        for seq in (2, 3):
            leg_booking = by_seq[seq]
            assert leg_booking.scheduled_time is None
            assert leg_booking.time_confirmed is False
            assert leg_booking.is_return is False
            assert leg_booking.parent_booking_id is None
            assert leg_booking.route_group_id == request.route_group_id

    def test_accept_makes_other_offers_unavailable(
        self,
        client,
        db,
        sample_institution,
        sample_company,
        sample_company_2,
        company_auth_headers,
    ):
        """Test: Accepter rend les autres offres UNAVAILABLE."""
        company1, _user1 = sample_company
        company2, _user2 = sample_company_2

        # Créer une demande avec 2 offres
        request = TransportRequest()
        request.institution_id = sample_institution.id
        request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        request.pickup_location = "123 Rue Test"
        request.dropoff_location = "456 Avenue Dest"
        scheduled = datetime.now(UTC) + timedelta(days=2)
        request.mission_date = scheduled.date()
        request.scheduled_time = scheduled
        request.pickup_time_confirmed = True
        request.status = RequestStatus.SENT.value
        request.sent_at = datetime.now(UTC)
        db.session.add(request)
        db.session.flush()

        offer1 = RequestOffer(
            transport_request_id=request.id,
            company_id=company1.id,
            mode=OfferMode.BROADCAST.value,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=2),
        )
        offer2 = RequestOffer(
            transport_request_id=request.id,
            company_id=company2.id,
            mode=OfferMode.BROADCAST.value,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=2),
        )
        db.session.add(offer1)
        db.session.add(offer2)
        db.session.commit()

        # Company 1 accepte
        response = client.post(
            f"/api/v1/company/request-offers/{offer1.id}/accept",
            headers=company_auth_headers,
        )
        assert response.status_code == 200

        # Vérifier que l'offre 2 est UNAVAILABLE
        db.session.refresh(offer2)
        assert offer2.status == OfferStatus.UNAVAILABLE.value

    def test_second_accept_fails(
        self,
        client,
        db,
        sample_institution,
        sample_company,
        sample_company_2,
        company_auth_headers,
        company_2_auth_headers,
    ):
        """Test: Deuxième acceptation échoue (first accept wins)."""
        company1, _user1 = sample_company
        company2, _user2 = sample_company_2

        # Créer une demande avec 2 offres
        request = TransportRequest()
        request.institution_id = sample_institution.id
        request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        request.pickup_location = "123 Rue Test"
        request.dropoff_location = "456 Avenue Dest"
        scheduled = datetime.now(UTC) + timedelta(days=2)
        request.mission_date = scheduled.date()
        request.scheduled_time = scheduled
        request.pickup_time_confirmed = True
        request.status = RequestStatus.SENT.value
        request.sent_at = datetime.now(UTC)
        db.session.add(request)
        db.session.flush()

        offer1 = RequestOffer(
            transport_request_id=request.id,
            company_id=company1.id,
            mode=OfferMode.BROADCAST.value,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=2),
        )
        offer2 = RequestOffer(
            transport_request_id=request.id,
            company_id=company2.id,
            mode=OfferMode.BROADCAST.value,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=2),
        )
        db.session.add(offer1)
        db.session.add(offer2)
        db.session.commit()

        # Company 1 accepte
        response1 = client.post(
            f"/api/v1/company/request-offers/{offer1.id}/accept",
            headers=company_auth_headers,
        )
        assert response1.status_code == 200

        # Company 2 essaie d'accepter -> échec (offre UNAVAILABLE)
        response2 = client.post(
            f"/api/v1/company/request-offers/{offer2.id}/accept",
            headers=company_2_auth_headers,
        )
        assert response2.status_code == 409
        body2 = response2.get_json()
        assert body2.get("code") in ("OFFER_UNAVAILABLE", "REQUEST_CONVERTED")

    def test_accept_after_reject_returns_offer_rejected(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
    ):
        """Refus puis tentative d'acceptation → OFFER_REJECTED (409)."""
        _request, offer = sample_request_with_offer

        reject_response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/reject",
            headers=company_auth_headers,
            json={"reason": "Indisponible"},
        )
        assert reject_response.status_code == 200

        accept_response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
        )
        assert accept_response.status_code == 409
        body = accept_response.get_json()
        assert body.get("code") == "OFFER_REJECTED"

    def test_accept_rdv_only_without_proposed_pickup_returns_422(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
    ):
        """Cas Khalid : RDV seul sans départ — Valider interdit (Planifier requis)."""
        request, offer = sample_request_with_offer
        past_rdv = datetime.now(UTC) - timedelta(hours=2)
        request.pickup_time_confirmed = False
        request.scheduled_time = past_rdv
        request.scheduled_time_type = "arrival"
        request.appointment_time_confirmed = True
        request.is_urgent = False
        db.session.commit()

        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
        )

        assert response.status_code == 422
        body = response.get_json()
        assert body.get("code") == "PROPOSED_PICKUP_REQUIRED"

    def test_accept_rdv_only_with_proposed_pickup_creates_booking(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
    ):
        """Planifier : accept avec proposed_pickup_time sur RDV seul."""
        request, offer = sample_request_with_offer
        past_rdv = datetime.now(UTC) - timedelta(hours=2)
        proposed = datetime.now(UTC) + timedelta(minutes=15)
        request.pickup_time_confirmed = False
        request.scheduled_time = past_rdv
        request.scheduled_time_type = "arrival"
        request.appointment_time_confirmed = True
        db.session.commit()

        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
            json={"proposed_pickup_time": proposed.isoformat()},
        )

        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        assert data.get("booking_id") is not None

    def test_accept_expired_offer_returns_410_even_with_proposed_pickup(
        self,
        client,
        db,
        sample_request_with_offer,
        company_auth_headers,
    ):
        """Offre expirée (urgente ou non) — gate avant accept, même avec proposed_pickup_time."""
        _request, offer = sample_request_with_offer
        offer.expires_at = datetime.now(UTC) - timedelta(minutes=5)
        db.session.commit()

        proposed = (datetime.now(UTC) + timedelta(minutes=15)).isoformat()
        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_auth_headers,
            json={"proposed_pickup_time": proposed},
        )

        assert response.status_code == 410
        body = response.get_json()
        assert body.get("code") == "OFFER_EXPIRED"

    def test_cannot_accept_other_company_offer(
        self, client, db, sample_request_with_offer, company_2_auth_headers
    ):
        """Test: Une entreprise ne peut pas accepter l'offre d'une autre."""
        _request, offer = sample_request_with_offer

        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/accept",
            headers=company_2_auth_headers,
        )

        assert response.status_code == 403


class TestRejectOffer:
    """Tests pour le rejet d'offres avec escalade."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Reject Test"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_companies(self, db):
        """Crée 3 entreprises de test."""
        companies = []
        for i in range(3):
            user = User()
            user.email = f"company{i}_{uuid.uuid4().hex[:8]}@test.com"
            user.username = user.email
            user.password = "test"
            user.role = UserRole.COMPANY.value
            db.session.add(user)
            db.session.flush()

            company = Company()
            company.name = f"Transport Reject Test {i}"
            company.user_id = user.id
            company.is_approved = True
            company.dispatch_enabled = True
            db.session.add(company)
            db.session.flush()

            companies.append((company, user))

        return companies

    @pytest.fixture
    def company_auth_headers(self, sample_companies):
        """Headers JWT pour la première entreprise."""
        company, user = sample_companies[0]
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": UserRole.COMPANY.value,
                "aud": "atmr-api",
                "user_id": user.id,
                "company_id": company.id,
            },
        )
        return {"Authorization": f"Bearer {token}"}

    def test_reject_sequential_triggers_escalade(
        self, client, db, sample_institution, sample_companies, company_auth_headers
    ):
        """Test: Rejeter une offre séquentielle déclenche l'escalade."""
        company1, _user1 = sample_companies[0]
        company2, _user2 = sample_companies[1]
        company3, _user3 = sample_companies[2]

        # Définir les préférences
        InstitutionTransportPreference.set_preferences(
            institution_id=sample_institution.id,
            company_ids=[company1.id, company2.id, company3.id],
        )
        db.session.commit()

        # Créer une demande avec offre séquentielle (première préférence)
        request = TransportRequest()
        request.institution_id = sample_institution.id
        request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
        request.pickup_location = "123 Rue Test"
        request.dropoff_location = "456 Avenue Dest"
        scheduled = datetime.now(UTC) + timedelta(days=2)
        request.mission_date = scheduled.date()
        request.scheduled_time = scheduled
        request.status = RequestStatus.SENT.value
        request.sent_at = datetime.now(UTC)
        db.session.add(request)
        db.session.flush()

        offer = RequestOffer(
            transport_request_id=request.id,
            company_id=company1.id,
            mode=OfferMode.SEQUENTIAL.value,
            order=1,
            status=OfferStatus.PENDING.value,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        db.session.add(offer)
        db.session.commit()

        # Company 1 rejette
        response = client.post(
            f"/api/v1/company/request-offers/{offer.id}/reject",
            headers=company_auth_headers,
            json={"reason": "Pas disponible"},
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["success"] is True
        assert data["escalated"] is True
        assert data["next_offer_id"] is not None

        # Vérifier l'offre rejetée
        db.session.refresh(offer)
        assert offer.status == OfferStatus.REJECTED.value
        assert offer.rejection_reason == "Pas disponible"

        # Vérifier la nouvelle offre (company 2)
        new_offer = RequestOffer.query.get(data["next_offer_id"])
        assert new_offer is not None
        assert new_offer.company_id == company2.id
        assert new_offer.order == 2
        assert new_offer.status == OfferStatus.PENDING.value


class TestTransportPreferences:
    """Tests pour la gestion des préférences de transport."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique Prefs Test"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_user(self, db, sample_institution):
        """Crée un utilisateur institution admin."""
        user = User()
        user.email = f"admin_{uuid.uuid4().hex[:8]}@test.com"
        user.username = user.email
        user.password = "test"
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
            identity=str(sample_user.public_id),
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "aud": "atmr-api",
                "user_id": sample_user.id,
                "institution_id": sample_institution.id,
                "institution_role": "institution_admin",
            },
        )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def sample_companies(self, db):
        """Crée des entreprises de test."""
        companies = []
        for i in range(3):
            user = User()
            user.email = f"company{i}_{uuid.uuid4().hex[:8]}@test.com"
            user.username = user.email
            user.password = "test"
            user.role = UserRole.COMPANY.value
            db.session.add(user)
            db.session.flush()

            company = Company()
            company.name = f"Transport Pref Test {i}"
            company.user_id = user.id
            company.is_approved = True
            company.dispatch_enabled = True
            db.session.add(company)
            db.session.flush()

            companies.append(company)

        return companies

    def test_get_preferences_empty(self, client, db, sample_institution, auth_headers):
        """Test: GET préférences retourne liste vide si pas définies."""
        response = client.get(
            "/api/v1/institutions/settings/transport-preferences",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["preferences"] == []
        assert data["total"] == 0

    def test_set_preferences(
        self, client, db, sample_institution, auth_headers, sample_companies
    ):
        """Test: PUT définit les préférences."""
        company_ids = [c.id for c in sample_companies]

        response = client.put(
            "/api/v1/institutions/settings/transport-preferences",
            headers=auth_headers,
            json={"company_ids": company_ids},
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["total"] == 3
        assert len(data["preferences"]) == 3

        # Vérifier l'ordre
        for i, pref in enumerate(data["preferences"]):
            assert pref["order"] == i + 1
            assert pref["company_id"] == company_ids[i]

    def test_set_preferences_replaces_existing(
        self, client, db, sample_institution, auth_headers, sample_companies
    ):
        """Test: PUT remplace les préférences existantes."""
        # Définir des préférences initiales
        client.put(
            "/api/v1/institutions/settings/transport-preferences",
            headers=auth_headers,
            json={"company_ids": [sample_companies[0].id, sample_companies[1].id]},
        )

        # Remplacer par d'autres
        response = client.put(
            "/api/v1/institutions/settings/transport-preferences",
            headers=auth_headers,
            json={"company_ids": [sample_companies[2].id]},
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["total"] == 1
        assert data["preferences"][0]["company_id"] == sample_companies[2].id

    def test_get_eligible_companies(
        self, client, db, sample_institution, auth_headers, sample_companies
    ):
        """Test: GET eligible-companies retourne les entreprises éligibles."""
        response = client.get(
            "/api/v1/institutions/settings/eligible-companies",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["total"] >= len(sample_companies)
        assert all("is_preferred" in c for c in data["companies"])
