"""INSTITUTION-07 — ACL + workflow contrôle facturation (C01–C18)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token

from application.institutions.billing_control.mutations import (
    mark_booking_control_anomaly,
    reopen_booking_control,
    validate_booking_control,
)
from application.institutions.billing_control.status import effective_control_status
from application.institutions.change_booking_payer import change_booking_payer
from ext import db
from models import (
    BillingParty,
    Booking,
    BookingChangeEvent,
    Client,
    ClinicBillingPartyMapping,
    Company,
    Institution,
    InstitutionPatient,
    TransportRequest,
    User,
)
from models.enums import (
    BillingPartyType,
    BookingStatus,
    InstitutionBillingControlStatus,
    RequestStatus,
    UserRole,
)
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_institution_patient,
)


def _institution(db, name: str | None = None) -> Institution:
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = name or f"Clinique {uuid.uuid4().hex[:6]}"
    inst.institution_type = "clinic"
    inst.address = "Rue Test 1, 1200 Genève"
    inst.billing_address = inst.address
    db.session.add(inst)
    db.session.flush()
    return inst


def _institution_user(
    db,
    institution: Institution,
    *,
    role: str,
    email_prefix: str,
) -> User:
    user = User()
    user.username = f"{email_prefix}_{uuid.uuid4().hex[:6]}"
    user.email = f"{email_prefix}_{uuid.uuid4().hex[:6]}@test.ch"
    user.role = UserRole.INSTITUTION
    user.public_id = str(uuid.uuid4())
    user.institution_id = institution.id
    user.institution_role = role
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _auth_headers(
    db,
    user: User,
    institution: Institution,
    role: str,
) -> dict[str, str]:
    from models.web_session import WebSession

    now = datetime.now(UTC)
    session = WebSession()
    session.id = str(uuid.uuid4())
    session.user_id = int(user.id)
    session.institution_id = institution.id
    session.created_at = now
    session.expires_at = now + timedelta(hours=8)
    session.last_interactive_activity_at = now
    db.session.add(session)
    db.session.flush()

    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "role": UserRole.INSTITUTION.value,
            "institution_id": institution.id,
            "institution_role": role,
            "sid": session.id,
            "aud": "atmr-api",
        },
    )
    return {"Authorization": f"Bearer {token}"}


def _converted_booking(
    db,
    institution: Institution,
    *,
    company: Company | None = None,
) -> tuple[Booking, TransportRequest]:
    suffix = uuid.uuid4().hex[:8]
    scheduled = datetime.now(UTC) + timedelta(days=3)

    if company is None:
        cu = User()
        cu.username = f"co_{suffix}"
        cu.email = f"co_{suffix}@test.ch"
        cu.role = UserRole.company
        cu.public_id = str(uuid.uuid4())
        cu.set_password("password123", force_change=False)
        db.session.add(cu)
        db.session.flush()
        company = Company()
        company.name = f"Transport {suffix}"
        company.address = "Rue Transport 1"
        company.contact_phone = "0210000000"
        company.contact_email = f"t_{suffix}@test.ch"
        company.user_id = cu.id
        db.session.add(company)
        db.session.flush()

    icu = User()
    icu.username = f"icli_{suffix}"
    icu.email = f"icli_{suffix}@test.ch"
    icu.role = UserRole.client
    icu.public_id = str(uuid.uuid4())
    icu.set_password("password123", force_change=False)
    db.session.add(icu)
    db.session.flush()

    institution_client = Client()
    institution_client.user_id = icu.id
    institution_client.company_id = company.id
    institution_client.is_institution = True
    institution_client.institution_name = institution.name
    institution_client.billing_address = institution.address
    db.session.add(institution_client)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = "Alice"
    patient.last_name = "TEST"
    patient.address = "Rue Patient"
    patient.postal_code = "1200"
    patient.city = "Genève"
    db.session.add(patient)
    db.session.flush()

    patient_bp = get_or_create_billing_party_for_institution_patient(
        company_id=company.id,
        institution_patient=patient,
    )

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = institution_client.id
    booking.customer_name = "Alice TEST"
    booking.pickup_location = "Domicile"
    booking.dropoff_location = "Clinique"
    booking.scheduled_time = scheduled
    booking.completed_at = scheduled
    booking.status = BookingStatus.COMPLETED.value
    booking.amount = Decimal("75.00")
    booking.billed_to_type = "patient"
    booking.billing_party_id = int(patient_bp.id)
    booking.institution_patient_id = patient.id
    db.session.add(booking)
    db.session.flush()

    transport_req = TransportRequest()
    transport_req.public_id = str(uuid.uuid4())
    transport_req.institution_id = institution.id
    transport_req.patient_id = patient.id
    transport_req.external_reference = f"CTRL-{suffix}"
    transport_req.pickup_location = booking.pickup_location
    transport_req.dropoff_location = booking.dropoff_location
    transport_req.scheduled_time = scheduled
    transport_req.mission_date = scheduled.date()
    transport_req.pickup_time_confirmed = True
    transport_req.status = RequestStatus.CONVERTED.value
    transport_req.billing_intent = "patient"
    transport_req.booking_id = booking.id
    db.session.add(transport_req)
    db.session.flush()
    db.session.commit()
    return booking, transport_req


@pytest.fixture
def control_institution(db):
    return _institution(db)


@pytest.fixture
def other_institution(db):
    return _institution(db, name="Autre Clinique")


@pytest.fixture
def admin_user(db, control_institution):
    return _institution_user(
        db, control_institution, role="institution_admin", email_prefix="admin"
    )


@pytest.fixture
def billing_user(db, control_institution):
    return _institution_user(
        db, control_institution, role="institution_billing", email_prefix="billing"
    )


@pytest.fixture
def requester_user(db, control_institution):
    return _institution_user(
        db, control_institution, role="institution_requester", email_prefix="requester"
    )


@pytest.fixture
def reader_user(db, control_institution):
    return _institution_user(
        db, control_institution, role="institution_reader", email_prefix="reader"
    )


@pytest.fixture
def admin_headers(db, admin_user, control_institution):
    return _auth_headers(db, admin_user, control_institution, "institution_admin")


@pytest.fixture
def billing_headers(db, billing_user, control_institution):
    return _auth_headers(db, billing_user, control_institution, "institution_billing")


@pytest.fixture
def requester_headers(db, requester_user, control_institution):
    return _auth_headers(
        db, requester_user, control_institution, "institution_requester"
    )


@pytest.fixture
def reader_headers(db, reader_user, control_institution):
    return _auth_headers(db, reader_user, control_institution, "institution_reader")


@pytest.fixture
def control_booking(db, control_institution):
    booking, tr = _converted_booking(db, control_institution)
    return {"booking": booking, "transport_request": tr}


LIST_URL = "/api/v1/institutions/billing/control/bookings"


class TestBillingControlACL:
    def test_c01_admin_list_200(self, client, admin_headers, control_booking):
        r = client.get(LIST_URL, headers=admin_headers)
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] >= 1

    def test_c02_billing_list_200(self, client, billing_headers, control_booking):
        r = client.get(LIST_URL, headers=billing_headers)
        assert r.status_code == 200

    def test_c03_requester_list_403(self, client, requester_headers, control_booking):
        r = client.get(LIST_URL, headers=requester_headers)
        assert r.status_code == 403

    def test_c04_reader_list_403(self, client, reader_headers, control_booking):
        r = client.get(LIST_URL, headers=reader_headers)
        assert r.status_code == 403

    def test_c05_requester_payer_change_403(
        self, client, requester_headers, control_booking
    ):
        bid = control_booking["booking"].id
        r = client.put(
            f"/api/v1/institutions/billing/bookings/{bid}",
            headers=requester_headers,
            json={
                "billing_intent": "institution",
                "billing_change_reason_code": "ADMIN_CORRECTION",
                "override_reason": "Test ACL requester",
            },
        )
        assert r.status_code == 403

    def test_c06_billing_payer_change_allowed_if_unlocked(
        self, client, db, billing_headers, control_institution
    ):
        clinic_co = Company()
        clinic_co.name = f"Clinique payeuse {uuid.uuid4().hex[:4]}"
        clinic_co.address = "Clinique addr"
        clinic_co.contact_phone = "0220000000"
        clinic_co.contact_email = f"c_{uuid.uuid4().hex[:6]}@test.ch"
        cu = User()
        cu.username = f"clu_{uuid.uuid4().hex[:6]}"
        cu.email = f"{cu.username}@test.ch"
        cu.role = UserRole.company
        cu.public_id = str(uuid.uuid4())
        cu.set_password("password123", force_change=False)
        db.session.add(cu)
        db.session.flush()
        clinic_co.user_id = cu.id
        db.session.add(clinic_co)
        db.session.flush()

        booking, _tr = _converted_booking(db, control_institution)
        booking.client.default_billed_to_company_id = clinic_co.id
        db.session.flush()

        transport = db.session.get(Company, booking.company_id)
        assert transport is not None
        clinic_bp = BillingParty()
        clinic_bp.company_id = transport.id
        clinic_bp.type = BillingPartyType.CLINIC
        clinic_bp.display_name = clinic_co.name
        clinic_bp.billing_address = clinic_co.address
        clinic_bp.external_ref = f"institution:{control_institution.id}"
        db.session.add(clinic_bp)
        db.session.flush()
        mapping = ClinicBillingPartyMapping()
        mapping.company_id = transport.id
        mapping.clinic_company_id = clinic_co.id
        mapping.billing_party_id = clinic_bp.id
        mapping.is_active = True
        db.session.add(mapping)
        db.session.commit()

        r = client.put(
            f"/api/v1/institutions/billing/bookings/{booking.id}",
            headers=billing_headers,
            json={
                "billing_intent": "institution",
                "billing_change_reason_code": "ADMIN_CORRECTION",
                "override_reason": "Correction payeur billing role",
            },
        )
        assert r.status_code == 200
        db.session.refresh(booking)
        assert booking.billed_to_type == "clinic"
        assert effective_control_status(booking) == "pending_review"

    @pytest.mark.parametrize("topology", ["route_group", "parent"])
    def test_payer_change_resolves_return_without_direct_transport_request(
        self,
        client,
        db,
        billing_headers,
        control_institution,
        topology,
    ):
        """P0 : retour sans TR directe (route_group / parent) — plus de 404 PUT payeur."""
        outbound, tr = _converted_booking(db, control_institution)

        return_booking = Booking()
        return_booking.company_id = outbound.company_id
        return_booking.client_id = outbound.client_id
        return_booking.customer_name = outbound.customer_name
        return_booking.pickup_location = outbound.dropoff_location
        return_booking.dropoff_location = outbound.pickup_location
        return_booking.scheduled_time = outbound.scheduled_time + timedelta(hours=2)
        return_booking.completed_at = return_booking.scheduled_time
        return_booking.status = BookingStatus.COMPLETED.value
        return_booking.amount = outbound.amount
        return_booking.billed_to_type = "patient"
        return_booking.billing_party_id = outbound.billing_party_id
        return_booking.institution_patient_id = outbound.institution_patient_id

        if topology == "route_group":
            route_group_id = str(uuid.uuid4())
            outbound.route_group_id = route_group_id
            outbound.route_sequence_number = 1
            return_booking.route_group_id = route_group_id
            return_booking.route_sequence_number = 2
            tr.route_group_id = route_group_id
        else:
            return_booking.parent_booking_id = outbound.id

        db.session.add(return_booking)
        db.session.commit()

        assert (
            TransportRequest.query.filter_by(
                booking_id=return_booking.id,
                institution_id=control_institution.id,
            ).first()
            is None
        )

        r = client.put(
            f"/api/v1/institutions/billing/bookings/{return_booking.id}",
            headers=billing_headers,
            json={
                "billing_intent": "patient",
                "billing_change_reason_code": "ADMIN_CORRECTION",
                "override_reason": "Correction payeur retour sans TR directe",
            },
        )

        assert r.status_code == 200, r.get_json()

    def test_c07_admin_validate(self, client, admin_headers, control_booking):
        bid = control_booking["booking"].id
        r = client.post(
            f"/api/v1/institutions/billing/control/bookings/{bid}/validate",
            headers=admin_headers,
            json={"actor_display_name": "Admin Validate"},
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["control"]["control_status"] == "validated"

    def test_c08_billing_anomaly(self, client, billing_headers, control_booking):
        bid = control_booking["booking"].id
        r = client.post(
            f"/api/v1/institutions/billing/control/bookings/{bid}/anomaly",
            headers=billing_headers,
            json={
                "anomaly_reason_code": "PAYER_NOT_FOUND",
                "comment": "Payeur introuvable",
                "actor_display_name": "Billing Anomaly",
            },
        )
        assert r.status_code == 200
        assert r.get_json()["control"]["control_status"] == "anomaly"

    def test_c09_cross_tenant_404(
        self, client, db, billing_headers, control_booking, other_institution
    ):
        other_booking, _ = _converted_booking(db, other_institution)
        r = client.get(
            f"/api/v1/institutions/billing/control/bookings/{other_booking.id}",
            headers=billing_headers,
        )
        assert r.status_code == 404

    def test_c10_invoiced_booking_financial_mutation_409(
        self, client, billing_headers, control_booking
    ):
        booking = control_booking["booking"]
        booking.billing_locked_at = datetime.now(UTC)
        db.session.commit()
        r = client.put(
            f"/api/v1/institutions/billing/bookings/{booking.id}",
            headers=billing_headers,
            json={
                "billing_intent": "institution",
                "billing_change_reason_code": "ADMIN_CORRECTION",
                "override_reason": "Doit échouer verrouillé",
            },
        )
        assert r.status_code == 409


class TestBillingControlWorkflow:
    def test_c11_pending_to_validated(self, db, control_institution, control_booking):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        assert effective_control_status(booking) == "pending_review"
        result = validate_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="Validateur Test",
        )
        assert result.ok is True
        db.session.commit()
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.VALIDATED
        )

    def test_c12_audit_validated_by_exact(
        self, db, control_institution, control_booking
    ):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        actor = _institution_user(
            db, control_institution, role="institution_billing", email_prefix="audit"
        )
        validate_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=int(actor.id),
            actor_role="institution_billing",
            actor_display_name="Marie Audit",
        )
        db.session.commit()
        event = (
            BookingChangeEvent.query.filter_by(
                booking_id=booking.id,
                action_type="billing_control_validated",
            )
            .order_by(BookingChangeEvent.id.desc())
            .first()
        )
        assert event is not None
        assert int(event.actor_user_id) == int(actor.id)
        assert event.after_snapshot["control_status"] == "validated"
        assert booking.institution_control_validated_by_display_name == "Marie Audit"

    def test_c13_pending_to_anomaly(self, db, control_institution, control_booking):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        result = mark_booking_control_anomaly(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name="Anomaly User",
            anomaly_reason_code="FINANCIAL_INCONSISTENCY",
            anomaly_comment="Montant incohérent",
        )
        assert result.ok is True
        db.session.commit()
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.ANOMALY
        )

    def test_c14_anomaly_reason_preserved(
        self, db, control_institution, control_booking
    ):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        mark_booking_control_anomaly(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name=None,
            anomaly_reason_code="MISSING_BLOCKING_DATA",
            anomaly_comment="Adresse manquante",
        )
        db.session.commit()
        assert "MISSING_BLOCKING_DATA" in (
            booking.institution_control_anomaly_reason or ""
        )
        event = BookingChangeEvent.query.filter_by(
            booking_id=booking.id, action_type="billing_control_anomaly"
        ).first()
        assert event is not None
        assert event.reason is not None

    def test_c15_payer_correction_not_auto_validate(
        self, db, control_institution, control_booking
    ):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        booking.institution_control_status = InstitutionBillingControlStatus.VALIDATED
        booking.institution_control_validated_at = datetime.now(UTC)
        db.session.commit()

        suffix = uuid.uuid4().hex[:6]
        clinic_co = Company()
        clinic_co.name = f"Clinique payeuse WF {suffix}"
        clinic_co.address = "Addr"
        clinic_co.contact_phone = "0221111111"
        clinic_co.contact_email = f"cwf_{suffix}@test.ch"
        cu = User()
        cu.username = f"clu_wf_{suffix}"
        cu.email = f"clu_wf_{suffix}@test.ch"
        cu.role = UserRole.company
        cu.public_id = str(uuid.uuid4())
        cu.set_password("password123", force_change=False)
        db.session.add(cu)
        db.session.flush()
        clinic_co.user_id = cu.id
        db.session.add(clinic_co)
        db.session.flush()
        booking.client.default_billed_to_company_id = clinic_co.id
        transport = db.session.get(Company, booking.company_id)
        clinic_bp = BillingParty()
        clinic_bp.company_id = transport.id
        clinic_bp.type = BillingPartyType.CLINIC
        clinic_bp.display_name = clinic_co.name
        clinic_bp.billing_address = clinic_co.address
        clinic_bp.external_ref = f"institution:{control_institution.id}"
        db.session.add(clinic_bp)
        db.session.flush()
        db.session.add(
            ClinicBillingPartyMapping(
                company_id=transport.id,
                clinic_company_id=clinic_co.id,
                billing_party_id=clinic_bp.id,
                is_active=True,
            )
        )
        db.session.commit()

        result = change_booking_payer(
            booking,
            target_payer="institution",
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name="Corrector",
            override_reason="Correction après validation",
            billing_change_reason_code="ADMIN_CORRECTION",
        )
        assert result.ok is True
        db.session.commit()
        assert effective_control_status(booking) == "pending_review"
        assert booking.institution_control_validated_at is None

    def test_c16_validate_outbound_only(self, db, control_institution):
        scheduled = datetime.now(UTC) + timedelta(days=2)
        booking_out, tr = _converted_booking(db, control_institution)
        return_booking = Booking()
        return_booking.company_id = booking_out.company_id
        return_booking.client_id = booking_out.client_id
        return_booking.customer_name = "Alice TEST"
        return_booking.pickup_location = "Clinique"
        return_booking.dropoff_location = "Domicile"
        return_booking.scheduled_time = scheduled.replace(hour=15)
        return_booking.status = BookingStatus.COMPLETED.value
        return_booking.amount = Decimal("75.00")
        return_booking.billed_to_type = "patient"
        return_booking.billing_party_id = booking_out.billing_party_id
        return_booking.is_return = True
        return_booking.parent_booking_id = booking_out.id
        return_booking.route_group_id = getattr(booking_out, "route_group_id", None)
        db.session.add(return_booking)
        db.session.commit()

        validate_booking_control(
            booking_out,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="Admin",
        )
        db.session.commit()
        assert (
            booking_out.institution_control_status
            == InstitutionBillingControlStatus.VALIDATED
        )
        assert return_booking.institution_control_status is None
        assert effective_control_status(return_booking) == "pending_review"

    def test_c17_multi_leg_independent_state(self, db, control_institution):
        b1, tr1 = _converted_booking(db, control_institution)
        b2, tr2 = _converted_booking(db, control_institution)
        mark_booking_control_anomaly(
            b1,
            transport_request=tr1,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name=None,
            anomaly_reason_code="OTHER",
        )
        validate_booking_control(
            b2,
            transport_request=tr2,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="V",
        )
        db.session.commit()
        assert effective_control_status(b1) == "anomaly"
        assert effective_control_status(b2) == "validated"

    def test_c18_locked_readonly_control_mutations(
        self, db, control_institution, control_booking
    ):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        booking.billing_locked_at = datetime.now(UTC)
        db.session.commit()
        for fn, kwargs in (
            (
                validate_booking_control,
                {},
            ),
            (
                mark_booking_control_anomaly,
                {"anomaly_reason_code": "OTHER"},
            ),
            (
                reopen_booking_control,
                {"reason": "reopen"},
            ),
        ):
            booking.institution_control_status = InstitutionBillingControlStatus.ANOMALY
            db.session.commit()
            result = fn(
                booking,
                transport_request=tr,
                institution_id=control_institution.id,
                actor_user_id=1,
                actor_role="institution_billing",
                actor_display_name=None,
                **kwargs,
            )
            assert result.ok is False
            assert result.status_code == 409

    def test_null_legacy_effective_pending_review(self, db, control_booking):
        booking = control_booking["booking"]
        assert booking.institution_control_status is None
        assert effective_control_status(booking) == "pending_review"

    def test_reopen_anomaly_to_pending(self, db, control_institution, control_booking):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        mark_booking_control_anomaly(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name=None,
            anomaly_reason_code="TRANSPORT_DISPUTED",
        )
        db.session.commit()
        result = reopen_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="Reopen",
        )
        assert result.ok is True
        db.session.commit()
        assert effective_control_status(booking) == "pending_review"
        assert booking.institution_control_anomaly_reason is None

    def test_reopen_validated_to_pending(
        self, db, control_institution, control_booking
    ):
        """REOPEN-VALIDATED : Validé → À vérifier (annuler une validation trop rapide)."""
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        validate_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="Marc Validate",
        )
        db.session.commit()
        assert effective_control_status(booking) == "validated"
        assert booking.institution_control_validated_at is not None

        result = reopen_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name="Marc Reopen",
            reason="Validation trop rapide",
        )
        assert result.ok is True
        db.session.commit()
        assert effective_control_status(booking) == "pending_review"
        assert booking.institution_control_validated_at is None
        assert booking.institution_control_validated_by_user_id is None
        assert booking.institution_control_validated_by_display_name is None

    def test_reopen_pending_rejected(self, db, control_institution, control_booking):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        assert effective_control_status(booking) == "pending_review"
        result = reopen_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name=None,
        )
        assert result.ok is False
        assert result.status_code == 409

    def test_reopen_validated_locked_rejected(
        self, db, control_institution, control_booking
    ):
        """REOPEN-VALIDATED : Validé + verrouillé → 409 (readonly avant statut)."""
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        validate_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="Marc",
        )
        booking.billing_locked_at = datetime.now(UTC)
        db.session.commit()
        assert effective_control_status(booking) == "validated"

        result = reopen_booking_control(
            booking,
            transport_request=tr,
            institution_id=control_institution.id,
            actor_user_id=1,
            actor_role="institution_billing",
            actor_display_name=None,
            reason="Ne doit pas passer",
        )
        assert result.ok is False
        assert result.status_code == 409
        assert effective_control_status(booking) == "validated"
        assert booking.institution_control_validated_at is not None


class TestBillingControlAPID01D12:
    """Gate API BILLING CONTROL — contrat liste/détail (D01–D12)."""

    def test_d01_admin_list_contract(self, client, admin_headers, control_booking):
        r = client.get(LIST_URL, headers=admin_headers)
        assert r.status_code == 200
        data = r.get_json()
        assert "items" in data
        assert "pagination" in data
        assert "summary" in data
        assert data["summary"]["total"] >= 1
        item = next(
            i for i in data["items"] if i["booking_id"] == control_booking["booking"].id
        )
        assert item["control"]["effective_status"] == "pending_review"
        assert item["payer"]["type"] == "patient"
        assert item["billing"]["editable"] is True

    def test_d02_billing_list_contract(self, client, billing_headers, control_booking):
        r = client.get(LIST_URL, headers=billing_headers)
        assert r.status_code == 200
        assert r.get_json()["summary"]["total"] >= 1

    def test_d03_period_filter(self, client, admin_headers, control_booking):
        booking = control_booking["booking"]
        st = booking.scheduled_time
        period = f"{st.year}-{st.month:02d}"
        r = client.get(f"{LIST_URL}?period={period}", headers=admin_headers)
        assert r.status_code == 200
        ids = {i["booking_id"] for i in r.get_json()["items"]}
        assert booking.id in ids

        wrong = f"{st.year}-{(st.month % 12) + 1:02d}"
        r2 = client.get(f"{LIST_URL}?period={wrong}", headers=admin_headers)
        ids2 = {i["booking_id"] for i in r2.get_json()["items"]}
        assert booking.id not in ids2

    def test_d04_control_status_filter(
        self, client, db, admin_headers, control_booking
    ):
        booking = control_booking["booking"]
        tr = control_booking["transport_request"]
        validate_booking_control(
            booking,
            transport_request=tr,
            institution_id=tr.institution_id,
            actor_user_id=1,
            actor_role="institution_admin",
            actor_display_name="D04",
        )
        db.session.commit()

        r_val = client.get(
            f"{LIST_URL}?control_status=validated", headers=admin_headers
        )
        assert booking.id in {i["booking_id"] for i in r_val.get_json()["items"]}

        r_pend = client.get(
            f"{LIST_URL}?control_status=pending_review", headers=admin_headers
        )
        assert booking.id not in {i["booking_id"] for i in r_pend.get_json()["items"]}

    def test_d05_patient_transport_payer_filters(
        self, client, admin_headers, control_booking
    ):
        booking = control_booking["booking"]
        patient_id = booking.institution_patient_id
        company_id = booking.company_id

        r = client.get(
            f"{LIST_URL}?patient={patient_id}&transport_company={company_id}&payer_type=patient",
            headers=admin_headers,
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["summary"]["total"] >= 1
        assert all(
            i["booking_id"] == booking.id for i in data["items"]
        ) or booking.id in {i["booking_id"] for i in data["items"]}

    def test_d06_pagination_total_coherent(
        self, client, db, admin_headers, control_institution
    ):
        _converted_booking(db, control_institution)
        _converted_booking(db, control_institution)
        r = client.get(f"{LIST_URL}?page=1&page_size=1", headers=admin_headers)
        data = r.get_json()
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 1
        assert data["pagination"]["total"] >= 2
        assert len(data["items"]) == 1
        assert data["summary"]["total"] == data["pagination"]["total"]

    def test_d07_summary_matches_population(
        self, client, admin_headers, control_booking
    ):
        r = client.get(LIST_URL, headers=admin_headers)
        data = r.get_json()
        s = data["summary"]
        assert s["total"] == data["pagination"]["total"]
        assert s["pending_review"] + s["validated"] + s["anomaly"] == s["total"]

    def test_d08_round_trip_distinct_linked(
        self, client, db, admin_headers, control_institution
    ):
        outbound, _tr = _converted_booking(db, control_institution)
        scheduled = outbound.scheduled_time
        ret = Booking()
        ret.company_id = outbound.company_id
        ret.client_id = outbound.client_id
        ret.customer_name = outbound.customer_name
        ret.pickup_location = "Clinique"
        ret.dropoff_location = "Domicile"
        ret.scheduled_time = scheduled.replace(hour=15)
        ret.status = BookingStatus.COMPLETED.value
        ret.amount = Decimal("75.00")
        ret.billed_to_type = "patient"
        ret.billing_party_id = outbound.billing_party_id
        ret.institution_patient_id = outbound.institution_patient_id
        ret.is_return = True
        ret.parent_booking_id = outbound.id
        db.session.add(ret)
        db.session.commit()

        period = f"{scheduled.year}-{scheduled.month:02d}"
        r = client.get(f"{LIST_URL}?period={period}", headers=admin_headers)
        items = {i["booking_id"]: i for i in r.get_json()["items"]}
        assert outbound.id in items
        assert ret.id in items
        assert items[outbound.id]["segment_type"] == "outbound"
        assert items[ret.id]["segment_type"] == "return"
        sib_ids = {
            s["booking_id"] for s in items[outbound.id]["relationship"]["siblings"]
        }
        assert ret.id in sib_ids

    def test_d09_detail_payer_control_billing(
        self, client, admin_headers, control_booking
    ):
        bid = control_booking["booking"].id
        r = client.get(
            f"/api/v1/institutions/billing/control/bookings/{bid}",
            headers=admin_headers,
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["payer"]["type"] == "patient"
        assert data["control"]["effective_status"] == "pending_review"
        assert "locked" in data["billing"]
        assert "editable" in data["billing"]

    def test_d10_cross_tenant_list_excludes_other(
        self, client, billing_headers, control_booking, other_institution, db
    ):
        other_booking, _ = _converted_booking(db, other_institution)
        r = client.get(LIST_URL, headers=billing_headers)
        ids = {i["booking_id"] for i in r.get_json()["items"]}
        assert other_booking.id not in ids

    def test_d11_legacy_null_pending_without_write(
        self, client, admin_headers, control_booking, db
    ):
        booking = control_booking["booking"]
        booking.institution_control_status = None
        db.session.commit()

        r = client.get(LIST_URL, headers=admin_headers)
        item = next(i for i in r.get_json()["items"] if i["booking_id"] == booking.id)
        assert item["control"]["effective_status"] == "pending_review"
        db.session.refresh(booking)
        assert booking.institution_control_status is None

    def test_d12_non_eligible_period_preview_absent(
        self, client, admin_headers, control_institution, db
    ):
        scheduled = datetime.now(UTC) + timedelta(days=3)
        booking, _tr = _converted_booking(db, control_institution)
        booking.status = BookingStatus.PENDING.value
        booking.scheduled_time = scheduled
        db.session.commit()

        period = f"{scheduled.year}-{scheduled.month:02d}"
        r = client.get(f"{LIST_URL}?period={period}", headers=admin_headers)
        ids = {i["booking_id"] for i in r.get_json()["items"]}
        assert booking.id not in ids

        r_all = client.get(LIST_URL, headers=admin_headers)
        assert booking.id in {i["booking_id"] for i in r_all.get_json()["items"]}
