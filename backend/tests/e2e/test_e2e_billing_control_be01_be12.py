"""Smoke E2E INSTITUTION-07 — Contrôle facturation BE1→BE12.

Parcours intégré HTTP (même contrat que la page UI) :
correction payeur → contrôle → period-preview.

Exécution :
    docker compose -f docker-compose.test.yml run --rm backend_tests sh -c \\
      "flask db upgrade heads && python -m pytest tests/e2e/test_e2e_billing_control_be01_be12.py -v"
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from application.institutions.billing_control.status import effective_control_status
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Booking
from models.enums import BookingStatus, InstitutionBillingControlStatus
from tests.e2e.helpers.billing_control_e2e import (
    LIST_URL,
    assert_triplet_coherent,
    institution_auth_headers,
    make_clinic_payer_company,
    make_eligible_control_booking,
    make_institution,
    make_institution_user,
    make_transport_company,
    period_param,
    setup_clinic_billing_mapping,
)

pytestmark = pytest.mark.e2e


@pytest.fixture
def bc_institution(db):
    return make_institution(db)


@pytest.fixture
def bc_company(db):
    return make_transport_company(db)


@pytest.fixture
def bc_admin(db, bc_institution):
    return make_institution_user(
        db, bc_institution, role="institution_admin", prefix="admin"
    )


@pytest.fixture
def bc_billing(db, bc_institution):
    return make_institution_user(
        db, bc_institution, role="institution_billing", prefix="billing"
    )


@pytest.fixture
def bc_requester(db, bc_institution):
    return make_institution_user(
        db, bc_institution, role="institution_requester", prefix="requester"
    )


@pytest.fixture
def bc_reader(db, bc_institution):
    return make_institution_user(
        db, bc_institution, role="institution_reader", prefix="reader"
    )


@pytest.fixture
def bc_eligible(db, bc_institution, bc_company):
    scheduled = datetime.now(UTC).replace(
        day=5, hour=10, minute=0, second=0, microsecond=0
    )
    if scheduled < datetime.now(UTC):
        scheduled = scheduled + timedelta(days=30)
    booking, tr, patient = make_eligible_control_booking(
        db,
        bc_institution,
        transport_company=bc_company,
        scheduled=scheduled,
    )
    return {
        "booking": booking,
        "transport_request": tr,
        "patient": patient,
        "period": period_param(scheduled),
        "scheduled": scheduled,
    }


class TestE2EBillingControlBE01BE12:
    def test_be01_admin_list_summary_and_eligible(
        self, client, db, bc_institution, bc_admin, bc_eligible
    ):
        headers = institution_auth_headers(
            bc_admin, bc_institution, "institution_admin"
        )
        period = bc_eligible["period"]
        r = client.get(f"{LIST_URL}?period={period}", headers=headers)
        assert r.status_code == 200
        data = r.get_json()
        assert data["summary"]["total"] >= 1
        assert data["summary"]["total"] == data["pagination"]["total"]
        ids = {i["booking_id"] for i in data["items"]}
        assert bc_eligible["booking"].id in ids

    def test_be02_billing_list_access(
        self, client, db, bc_institution, bc_billing, bc_eligible
    ):
        headers = institution_auth_headers(
            bc_billing, bc_institution, "institution_billing"
        )
        r = client.get(
            f"{LIST_URL}?period={bc_eligible['period']}",
            headers=headers,
        )
        assert r.status_code == 200

    def test_be03_requester_and_reader_denied(
        self, client, db, bc_institution, bc_requester, bc_reader, bc_eligible
    ):
        bid = bc_eligible["booking"].id
        for user, role in (
            (bc_requester, "institution_requester"),
            (bc_reader, "institution_reader"),
        ):
            headers = institution_auth_headers(user, bc_institution, role)
            assert client.get(LIST_URL, headers=headers).status_code == 403
            assert client.get(f"{LIST_URL}/{bid}", headers=headers).status_code == 403
            assert (
                client.post(
                    f"/api/v1/institutions/billing/control/bookings/{bid}/validate",
                    headers=headers,
                    json={},
                ).status_code
                == 403
            )

    def test_be04_be05_be12_payer_correction_chain(
        self, client, db, bc_institution, bc_billing, bc_company, bc_eligible
    ):
        """BE4+BE5+BE12 : Patient→Clinique, pending_review, triplet = period-preview."""
        booking = bc_eligible["booking"]
        institution = bc_institution
        clinic_co = make_clinic_payer_company(db)
        setup_clinic_billing_mapping(
            db,
            transport_company=bc_company,
            clinic_company=clinic_co,
            institution=institution,
        )
        booking.client.default_billed_to_company_id = clinic_co.id
        db.session.commit()

        headers = institution_auth_headers(
            bc_billing, institution, "institution_billing"
        )
        period = bc_eligible["period"]
        y, m = map(int, period.split("-"))

        r_put = client.put(
            f"/api/v1/institutions/billing/bookings/{booking.id}",
            headers=headers,
            json={
                "billing_intent": "institution",
                "billing_change_reason_code": "ADMIN_CORRECTION",
                "override_reason": "E2E BE4 correction payeur",
            },
        )
        assert r_put.status_code == 200, r_put.get_json()
        db.session.refresh(booking)

        r_detail = client.get(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}",
            headers=headers,
        )
        assert r_detail.status_code == 200
        detail = r_detail.get_json()
        assert_triplet_coherent(booking, detail)
        assert detail["control"]["effective_status"] == "pending_review"
        assert effective_control_status(booking) == "pending_review"
        assert booking.billed_to_type == "clinic"
        assert booking.billing_party_id is not None

        prev = build_period_invoice_preview(
            company_id=int(booking.company_id),
            period_year=y,
            period_month=m,
            clinic_company_id=int(booking.billed_to_company_id),
            include_line_details=True,
        )
        preview_ids = {line.booking_id for line in prev.preview_lines}
        assert int(booking.id) in preview_ids
        assert detail["payer"]["type"] == "clinic"
        assert detail["payer"]["billing_party_id"] == booking.billing_party_id

    def test_be06_validate_with_actor(
        self, client, db, bc_institution, bc_admin, bc_eligible
    ):
        booking = bc_eligible["booking"]
        headers = institution_auth_headers(
            bc_admin, bc_institution, "institution_admin"
        )
        r = client.post(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}/validate",
            headers=headers,
            json={"actor_display_name": "Marc E2E"},
        )
        assert r.status_code == 200
        db.session.refresh(booking)
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.VALIDATED
        )
        detail = client.get(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}",
            headers=headers,
        ).get_json()
        assert detail["control"]["effective_status"] == "validated"
        assert detail["control"]["validated_by_display_name"] == "Marc E2E"
        assert detail["control"]["validated_at"] is not None

    def test_be07_be08_anomaly_and_reopen(
        self, client, db, bc_institution, bc_billing, bc_eligible
    ):
        booking = bc_eligible["booking"]
        headers = institution_auth_headers(
            bc_billing, bc_institution, "institution_billing"
        )
        r_an = client.post(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}/anomaly",
            headers=headers,
            json={
                "anomaly_reason_code": "FINANCIAL_INCONSISTENCY",
                "comment": "Motif E2E BE7",
            },
        )
        assert r_an.status_code == 200
        db.session.refresh(booking)
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.ANOMALY
        )

        r_re = client.post(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}/reopen",
            headers=headers,
            json={},
        )
        assert r_re.status_code == 200
        db.session.refresh(booking)
        assert effective_control_status(booking) == "pending_review"

    def test_be09_round_trip_independent_validation(
        self, client, db, bc_institution, bc_admin, bc_company
    ):
        scheduled = datetime.now(UTC).replace(
            day=8, hour=9, minute=0, second=0, microsecond=0
        )
        if scheduled < datetime.now(UTC):
            scheduled += timedelta(days=30)
        outbound, _tr, _patient = make_eligible_control_booking(
            db, bc_institution, transport_company=bc_company, scheduled=scheduled
        )
        ret = Booking()
        ret.company_id = outbound.company_id
        ret.client_id = outbound.client_id
        ret.customer_name = outbound.customer_name
        ret.pickup_location = "Clinique"
        ret.dropoff_location = "Domicile"
        ret.scheduled_time = scheduled.replace(hour=15)
        ret.completed_at = ret.scheduled_time
        ret.status = BookingStatus.COMPLETED.value
        ret.amount = outbound.amount
        ret.billed_to_type = "patient"
        ret.billing_party_id = outbound.billing_party_id
        ret.institution_patient_id = outbound.institution_patient_id
        ret.is_return = True
        ret.parent_booking_id = outbound.id
        db.session.add(ret)
        db.session.commit()

        headers = institution_auth_headers(
            bc_admin, bc_institution, "institution_admin"
        )
        assert (
            client.post(
                f"/api/v1/institutions/billing/control/bookings/{outbound.id}/validate",
                headers=headers,
                json={"actor_display_name": "Admin E2E"},
            ).status_code
            == 200
        )
        db.session.refresh(outbound)
        db.session.refresh(ret)
        assert (
            outbound.institution_control_status
            == InstitutionBillingControlStatus.VALIDATED
        )
        assert ret.institution_control_status is None
        assert effective_control_status(ret) == "pending_review"

    def test_be10_locked_readonly_and_409(
        self, client, db, bc_institution, bc_billing, bc_eligible
    ):
        booking = bc_eligible["booking"]
        booking.billing_locked_at = datetime.now(UTC)
        db.session.commit()
        headers = institution_auth_headers(
            bc_billing, bc_institution, "institution_billing"
        )
        detail = client.get(
            f"/api/v1/institutions/billing/control/bookings/{booking.id}",
            headers=headers,
        ).get_json()
        assert detail["billing"]["locked"] is True
        assert detail["billing"]["editable"] is False
        assert (
            client.post(
                f"/api/v1/institutions/billing/control/bookings/{booking.id}/validate",
                headers=headers,
                json={},
            ).status_code
            == 409
        )

    def test_be11_ineligible_absent_from_period_list(
        self, client, db, bc_institution, bc_admin, bc_company
    ):
        scheduled = datetime.now(UTC).replace(
            day=12, hour=10, minute=0, second=0, microsecond=0
        )
        if scheduled < datetime.now(UTC):
            scheduled += timedelta(days=30)
        pending, _tr, _p = make_eligible_control_booking(
            db,
            bc_institution,
            transport_company=bc_company,
            scheduled=scheduled,
            status=BookingStatus.PENDING.value,
        )
        headers = institution_auth_headers(
            bc_admin, bc_institution, "institution_admin"
        )
        period = period_param(scheduled)
        data = client.get(f"{LIST_URL}?period={period}", headers=headers).get_json()
        assert pending.id not in {i["booking_id"] for i in data["items"]}

    def test_be12_control_population_matches_preview_eligibility(
        self, client, db, bc_institution, bc_admin, bc_company, bc_eligible
    ):
        """Invariant canary : IDs control (période) ⊆ éligibles period-preview patient."""
        from application.invoices.billing_period_eligibility import (
            booking_matches_period_preview_eligibility,
        )

        booking = bc_eligible["booking"]
        period = bc_eligible["period"]
        y, m = map(int, period.split("-"))
        headers = institution_auth_headers(
            bc_admin, bc_institution, "institution_admin"
        )
        data = client.get(
            f"{LIST_URL}?period={period}&page_size=200",
            headers=headers,
        ).get_json()
        assert data["summary"]["total"] == data["pagination"]["total"]

        for item in data["items"]:
            b = db.session.get(Booking, item["booking_id"])
            assert b is not None
            assert booking_matches_period_preview_eligibility(
                b,
                company_id=int(b.company_id),
                period_year=y,
                period_month=m,
                billed_to_type=str(b.billed_to_type or "patient"),
            )
            if int(b.id) == int(booking.id):
                prev = build_period_invoice_preview(
                    company_id=int(b.company_id),
                    period_year=y,
                    period_month=m,
                    client_id=int(b.client_id),
                    institution_patient_id=int(b.institution_patient_id),
                    include_line_details=True,
                )
                assert int(b.id) in {line.booking_id for line in prev.preview_lines}
