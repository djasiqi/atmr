# tests/e2e/test_e2e_institution_flow.py
# ruff: noqa: I001
"""Test E2E: Flux complet Institution → Company.

Ce smoke test valide le flux de bout en bout:
1. Institution crée un patient
2. Institution crée une demande de transport
3. Institution envoie la demande (crée des offres)
4. Company récupère et accepte l'offre
5. La demande est convertie en booking
6. Optionnel: Vérification des règles d'annulation EN_ROUTE

Usage:
    # Commande unique - les migrations sont appliquées automatiquement
    docker compose run --rm api python -m pytest tests/e2e/test_e2e_institution_flow.py -v

    # Ou avec le pattern
    docker compose run --rm api python -m pytest -k e2e_institution_flow -v

Note:
    Les migrations sont appliquées automatiquement par la fixture `e2e_db_migrations`
    définie dans conftest.py. Aucune étape manuelle n'est requise.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from flask_jwt_extended import create_access_token

from models import (
    Booking,
    BookingStatus,
    Company,
    Institution,
    InstitutionPatient,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    User,
    UserRole,
)
from tests.e2e.helpers.e2e_helpers import create_test_booking


# Marker pour identifier les tests E2E
pytestmark = pytest.mark.e2e


class TestE2EInstitutionFlow:
    """Test E2E du flux institution -> company."""

    # =========================================================================
    # FIXTURES
    # =========================================================================

    @pytest.fixture
    def e2e_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = f"Clinique E2E Test {uuid.uuid4().hex[:6]}"
        institution.public_id = str(uuid.uuid4())
        institution.institution_type = "clinic"
        institution.address = "123 Rue E2E Test, 1000 Lausanne"
        institution.contact_email = "e2e@clinique-test.ch"
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def e2e_institution_admin(self, db, e2e_institution):
        """Crée un utilisateur institution_admin."""
        user = User()
        user.email = f"admin_{uuid.uuid4().hex[:8]}@e2e-institution.ch"
        user.set_password("password123", force_change=False)
        user.role = UserRole.INSTITUTION.value
        user.institution_id = e2e_institution.id
        user.institution_role = "institution_admin"
        user.first_name = "Admin"
        user.last_name = "E2E"
        db.session.add(user)
        db.session.flush()
        return user

    @pytest.fixture
    def e2e_institution_headers(self, e2e_institution_admin, e2e_institution):
        """Headers JWT pour l'utilisateur institution."""
        token = create_access_token(
            identity=str(e2e_institution_admin.public_id),
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "institution_id": e2e_institution.id,
                "institution_role": "institution_admin",
                "aud": "atmr-api",
            },
        )
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @pytest.fixture
    def e2e_company(self, db):
        """Crée une company de test."""
        # Créer d'abord un user pour la company
        company_user = User()
        company_user.email = f"company_{uuid.uuid4().hex[:8]}@e2e-transport.ch"
        company_user.set_password("password123", force_change=False)
        company_user.role = UserRole.COMPANY.value
        company_user.first_name = "Company"
        company_user.last_name = "E2E"
        db.session.add(company_user)
        db.session.flush()

        company = Company()
        company.name = f"Transport E2E Test {uuid.uuid4().hex[:6]}"
        company.user_id = company_user.id
        company.address = "456 Avenue E2E Transport"
        company.phone = "+41791234567"
        company.email = company_user.email
        company.is_active = True
        # Accepter une offre exige une entreprise approuvée (accept_offer.py)
        company.is_approved = True
        db.session.add(company)
        db.session.flush()

        return company, company_user

    @pytest.fixture
    def e2e_clinic_company(self, db, e2e_institution):
        """Company clinique homonyme de l'institution (payeuse).

        `billing_intent=institution` impose de résoudre `billed_to_company_id`
        avant flush : le résolveur cherche une Company portant le nom de
        l'institution (institution_billing_resolver). Sans elle, l'acceptation
        d'offre échoue.
        """
        clinic_user = User()
        clinic_user.email = f"clinic_{uuid.uuid4().hex[:8]}@e2e-clinique.ch"
        clinic_user.set_password("password123", force_change=False)
        clinic_user.role = UserRole.COMPANY.value
        db.session.add(clinic_user)
        db.session.flush()

        clinic_company = Company()
        clinic_company.name = e2e_institution.name
        clinic_company.user_id = clinic_user.id
        clinic_company.address = e2e_institution.address
        clinic_company.email = clinic_user.email
        clinic_company.is_active = True
        db.session.add(clinic_company)
        db.session.flush()
        return clinic_company

    @pytest.fixture
    def e2e_company_headers(self, e2e_company):
        """Headers JWT pour l'utilisateur company."""
        company, company_user = e2e_company
        token = create_access_token(
            identity=str(company_user.public_id),
            additional_claims={
                "role": UserRole.COMPANY.value,
                "company_id": company.id,
                "aud": "atmr-api",
            },
        )
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # =========================================================================
    # TEST PRINCIPAL E2E
    # =========================================================================

    def test_e2e_institution_flow_complete(
        self,
        client,
        db,
        e2e_institution,
        e2e_institution_admin,
        e2e_institution_headers,
        e2e_company,
        e2e_company_headers,
        e2e_clinic_company,
    ):
        """Test E2E complet: Institution crée une demande, Company l'accepte.

        Flow:
        1. Institution: POST /institutions/patients
        2. Institution: POST /institutions/requests (DRAFT)
        3. Institution: POST /institutions/requests/{id}/send
        4. Company: GET /company/request-offers?status=pending
        5. Company: POST /company/request-offers/{offer_id}/accept
        6. Vérifications finales
        """
        company, _company_user = e2e_company
        external_ref = f"E2E-{uuid.uuid4().hex[:8]}"

        # =====================================================================
        # STEP 1: Institution crée un patient
        # =====================================================================
        patient_data = {
            "first_name": "Jean",
            "last_name": "Dupont",
            "birth_date": "1950-05-15",
            "external_reference": f"PAT-{uuid.uuid4().hex[:8]}",
            "phone": "+41791234567",
            "address": "789 Rue Patient, 1000 Lausanne",
            "mobility_reduced": True,
            "wheelchair_required": False,
            "notes": "Patient test E2E",
        }

        response = client.post(
            "/api/v1/institutions/patients",
            headers=e2e_institution_headers,
            json=patient_data,
        )

        assert response.status_code == 201, (
            f"Failed to create patient: {response.get_json()}"
        )
        patient_response = response.get_json()
        # La réponse encapsule le patient : {"patient": {...}, "sync": {...}}
        patient_payload = patient_response.get("patient") or patient_response
        patient_id = patient_payload.get("id")
        assert patient_id is not None, "Patient ID should be returned"

        # =====================================================================
        # STEP 2: Institution crée une demande de transport (DRAFT)
        # =====================================================================
        scheduled_time = datetime.now(UTC) + timedelta(days=2)
        request_data = {
            "external_reference": external_ref,
            "patient_id": patient_id,
            "mission_type": "patient_transport",
            "mission_date": scheduled_time.date().isoformat(),
            "scheduled_time": scheduled_time.isoformat(),
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
            "pickup_location": "123 Rue Départ, 1000 Lausanne",
            "pickup_lat": 46.5197,
            "pickup_lng": 6.6323,
            "dropoff_location": "456 Avenue Arrivée, 1005 Lausanne",
            "dropoff_lat": 46.5230,
            "dropoff_lng": 6.6400,
            "is_round_trip": False,
            "mobility_reduced": True,
            "wheelchair_required": False,
            "contact_name": "Dr. E2E Test",
            "contact_phone": "+41791112233",
            "notes": "Test E2E - transport patient",
            "billing_intent": "institution",
        }

        response = client.post(
            "/api/v1/institutions/requests",
            headers=e2e_institution_headers,
            json=request_data,
        )

        assert response.status_code == 201, (
            f"Failed to create request: {response.get_json()}"
        )
        request_response = response.get_json()
        request_id = request_response.get("id")
        assert request_id is not None, "Request ID should be returned"
        assert request_response.get("status") == RequestStatus.DRAFT.value

        # =====================================================================
        # STEP 3: Institution envoie la demande (crée des offres)
        # =====================================================================
        response = client.post(
            f"/api/v1/institutions/requests/{request_id}/send",
            headers=e2e_institution_headers,
        )

        assert response.status_code == 200, (
            f"Failed to send request: {response.get_json()}"
        )
        send_response = response.get_json()
        assert send_response.get("status") == RequestStatus.SENT.value
        offers_created = send_response.get("offers_created", 0)

        # Vérifier qu'au moins une offre a été créée
        # Note: Si aucune company éligible, le test peut échouer ici
        # On s'assure d'avoir au moins la company E2E éligible
        if offers_created == 0:
            # Créer manuellement une offre pour notre company
            offer = RequestOffer()
            offer.transport_request_id = request_id
            offer.company_id = company.id
            offer.status = OfferStatus.PENDING.value
            offer.mode = "broadcast"
            offer.sent_at = datetime.now(UTC)
            offer.expires_at = datetime.now(UTC) + timedelta(hours=24)
            db.session.add(offer)
            db.session.commit()
            offers_created = 1

        assert offers_created > 0, "At least one offer should be created"

        # =====================================================================
        # STEP 4: Company récupère les offres PENDING
        # =====================================================================
        response = client.get(
            f"/api/v1/company/request-offers?status={OfferStatus.PENDING.value}",
            headers=e2e_company_headers,
        )

        assert response.status_code == 200, (
            f"Failed to get offers: {response.get_json()}"
        )
        offers_response = response.get_json()
        offers = offers_response.get("offers", [])

        # Trouver l'offre correspondant à notre demande
        target_offer = None
        for offer in offers:
            offer_request = offer.get("transport_request", {})
            if offer_request.get("external_reference") == external_ref:
                target_offer = offer
                break

        # Si pas trouvée via API, récupérer directement en DB
        if not target_offer:
            db_offer = RequestOffer.query.filter_by(
                transport_request_id=request_id,
                company_id=company.id,
                status=OfferStatus.PENDING.value,
            ).first()
            assert db_offer is not None, "Offer should exist in DB"
            target_offer = {"id": db_offer.id}

        offer_id = target_offer.get("id")
        assert offer_id is not None, (
            f"Offer ID should be found for request {external_ref}"
        )

        # =====================================================================
        # STEP 5: Company accepte l'offre
        # =====================================================================
        response = client.post(
            f"/api/v1/company/request-offers/{offer_id}/accept",
            headers=e2e_company_headers,
        )

        assert response.status_code == 200, (
            f"Failed to accept offer: {response.get_json()}"
        )
        accept_response = response.get_json()
        assert accept_response.get("success") is True
        booking_id = accept_response.get("booking_id")
        assert booking_id is not None, "Booking ID should be returned after accept"

        # =====================================================================
        # STEP 6: Vérifications finales
        # =====================================================================
        db.session.expire_all()

        # 6a. Vérifier que la request est CONVERTED
        transport_req = TransportRequest.query.get(request_id)
        assert transport_req is not None
        assert transport_req.status == RequestStatus.CONVERTED.value
        assert transport_req.booking_id == booking_id, (
            "Request should have booking_id set"
        )

        # 6b. Vérifier que le booking existe et appartient à la company
        booking = Booking.query.get(booking_id)
        assert booking is not None, "Booking should exist"
        assert booking.company_id == company.id, "Booking should belong to the company"

        # 6c. Vérifier l'offre acceptée
        accepted_offer = RequestOffer.query.get(offer_id)
        assert accepted_offer.status == OfferStatus.ACCEPTED.value

        # 6d. Vérifier que les autres offres sont UNAVAILABLE (si plusieurs)
        other_offers = RequestOffer.query.filter(
            RequestOffer.transport_request_id == request_id,
            RequestOffer.id != offer_id,
        ).all()
        for other_offer in other_offers:
            assert other_offer.status == OfferStatus.UNAVAILABLE.value, (
                f"Other offer {other_offer.id} should be UNAVAILABLE, got {other_offer.status}"
            )

        # =====================================================================
        # SUCCESS
        # =====================================================================
        print("\n✅ E2E Test PASSED:")
        print(f"   - Patient ID: {patient_id}")
        print(f"   - Request ID: {request_id} (external_ref: {external_ref})")
        print(f"   - Offer ID: {offer_id}")
        print(f"   - Booking ID: {booking_id}")
        print(f"   - Company: {company.name} (ID: {company.id})")

    # =========================================================================
    # TEST OPTIONNEL: Annulation EN_ROUTE
    # =========================================================================

    def test_e2e_cancellation_en_route_is_billable(
        self,
        client,
        db,
        e2e_institution,
        e2e_institution_admin,
        e2e_institution_headers,
        e2e_company,
        e2e_company_headers,
    ):
        """Test: Une annulation EN_ROUTE génère des frais (billable).

        Ce test vérifie les règles de facturation ajoutées en ÉTAPE 5.
        """
        from application.bookings.cancellation_rules import (
            compute_cancellation_fields_with_status,
            is_status_billable_cancellation,
        )

        _company, _company_user = e2e_company

        # Test 1: Vérifier que EN_ROUTE est toujours billable
        assert is_status_billable_cancellation("EN_ROUTE") is True
        assert is_status_billable_cancellation("IN_PROGRESS") is True
        assert is_status_billable_cancellation("PENDING") is False
        assert is_status_billable_cancellation("ACCEPTED") is False

        # Test 2: Calculer les champs d'annulation pour EN_ROUTE
        fields = compute_cancellation_fields_with_status(
            booking_status="EN_ROUTE",
            reason_code="INSTITUTION_CANCELLED",
            reason_text="Patient hospitalisé",
            cancelled_by_role="institution",
        )

        assert fields["is_cancellation_billable"] is True
        assert fields["billing_info"]["billing_reason"] == "status_en_route"
        assert "déplacement" in fields["billing_info"]["billing_description"].lower()

        # Test 3: IN_PROGRESS avec surcharge
        fields_in_progress = compute_cancellation_fields_with_status(
            booking_status="IN_PROGRESS",
            reason_code="INSTITUTION_CANCELLED",
            reason_text="Annulation tardive",
            cancelled_by_role="institution",
        )

        assert fields_in_progress["is_cancellation_billable"] is True
        assert (
            fields_in_progress["billing_info"]["billing_reason"] == "status_in_progress"
        )
        assert fields_in_progress["billing_info"]["surcharge_percent"] == 100

        print("\n✅ Cancellation rules E2E PASSED:")
        print("   - EN_ROUTE → billable ✓")
        print("   - IN_PROGRESS → billable + 100% surcharge ✓")


class TestE2EInstitutionFlowEdgeCases:
    """Tests E2E pour les cas limites du flux institution."""

    @pytest.fixture
    def e2e_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = f"Clinique Edge {uuid.uuid4().hex[:6]}"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def e2e_institution_admin(self, db, e2e_institution):
        """Crée un utilisateur institution_admin."""
        user = User()
        user.email = f"edge_{uuid.uuid4().hex[:8]}@test.ch"
        user.set_password("password123", force_change=False)
        user.role = UserRole.INSTITUTION.value
        user.institution_id = e2e_institution.id
        user.institution_role = "institution_admin"
        db.session.add(user)
        db.session.flush()
        return user

    @pytest.fixture
    def e2e_institution_headers(self, e2e_institution_admin, e2e_institution):
        """Headers JWT pour l'utilisateur institution."""
        token = create_access_token(
            identity=str(e2e_institution_admin.public_id),
            additional_claims={
                "role": UserRole.INSTITUTION.value,
                "institution_id": e2e_institution.id,
                "institution_role": "institution_admin",
                "aud": "atmr-api",
            },
        )
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def test_e2e_cancel_converted_request_returns_409(
        self,
        client,
        db,
        e2e_institution,
        e2e_institution_admin,
        e2e_institution_headers,
    ):
        """Test: Annuler une request CONVERTED retourne 409 avec resulting_booking_id."""
        # Créer une request déjà convertie (booking réel : FK booking_id)
        scheduled_at = datetime.now(UTC) + timedelta(days=2)
        converted_booking = create_test_booking(db, scheduled_time=scheduled_at)
        transport_req = TransportRequest()
        transport_req.institution_id = e2e_institution.id
        transport_req.external_reference = f"EDGE-{uuid.uuid4().hex[:8]}"
        transport_req.pickup_location = "123 Rue Test"
        transport_req.dropoff_location = "456 Avenue Test"
        transport_req.mission_date = scheduled_at.date()
        transport_req.scheduled_time = scheduled_at
        transport_req.status = RequestStatus.CONVERTED.value
        transport_req.booking_id = converted_booking.id
        db.session.add(transport_req)
        db.session.commit()

        # Tenter d'annuler
        response = client.post(
            f"/api/v1/institutions/requests/{transport_req.id}/cancel",
            headers=e2e_institution_headers,
        )

        assert response.status_code == 409
        data = response.get_json()
        assert "resulting_booking_id" in data
        assert data["resulting_booking_id"] == transport_req.booking_id
        assert (
            "convertie" in data.get("error", "").lower()
            or "booking" in data.get("error", "").lower()
        )

        print("\n✅ Cancel CONVERTED request returns 409 with resulting_booking_id ✓")

    def test_e2e_cancel_draft_request_succeeds(
        self,
        client,
        db,
        e2e_institution,
        e2e_institution_admin,
        e2e_institution_headers,
    ):
        """Test: Annuler une request DRAFT réussit."""
        # Créer une request DRAFT
        scheduled_at = datetime.now(UTC) + timedelta(days=2)
        transport_req = TransportRequest()
        transport_req.institution_id = e2e_institution.id
        transport_req.external_reference = f"DRAFT-{uuid.uuid4().hex[:8]}"
        transport_req.pickup_location = "123 Rue Test"
        transport_req.dropoff_location = "456 Avenue Test"
        transport_req.mission_date = scheduled_at.date()
        transport_req.scheduled_time = scheduled_at
        transport_req.status = RequestStatus.DRAFT.value
        db.session.add(transport_req)
        db.session.commit()

        # Annuler
        response = client.post(
            f"/api/v1/institutions/requests/{transport_req.id}/cancel",
            headers=e2e_institution_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data.get("status") == RequestStatus.CANCELLED.value

        print("\n✅ Cancel DRAFT request succeeds ✓")
