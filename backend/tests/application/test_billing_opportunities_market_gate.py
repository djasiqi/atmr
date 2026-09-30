"""Registre « Direct patient » × gate Market LIRIE — parité liste / preview / génération.

Cas réel (30.09.2026) : course #40543 terminée, payeur patient, créée par une
institution (CLHDA) et pas encore validée par elle. La liste des patients
annonçait « 1 transport — 45 CHF » alors que la prévisualisation et la
génération retenaient la course (« Aucune réservation trouvée pour cette
période »). La liste doit appliquer le même gate et expliquer la retenue.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from application.invoices.billing_opportunities import (
    list_billing_opportunities,
    opportunities_to_dict,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Booking, Client, InstitutionPatient, User
from models.enums import (
    BookingCreatedVia,
    BookingStatus,
    InstitutionBillingControlStatus,
    UserRole,
)
from models.invoice import CompanyBillingSettings
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_institution_patient,
)
from tests.e2e.helpers.billing_control_e2e import (
    make_institution,
    make_transport_company,
)

ZURICH = ZoneInfo("Europe/Zurich")
PERIOD_YEAR = 2026
PERIOD_MONTH = 9
# Veille de fin de mois, course du jour terminée à 10:01.
BEFORE_RELEASE = datetime(2026, 9, 30, 10, 37, tzinfo=ZURICH)
# Premier instant du mois suivant : libération automatique.
AT_RELEASE = datetime(2026, 10, 1, 0, 0, tzinfo=ZURICH)


def _make_world(db) -> dict:
    institution = make_institution(db, name="Clinique les Hauts d'Anières")
    transport = make_transport_company(db)

    settings = CompanyBillingSettings()
    settings.company_id = transport.id
    settings.payment_terms_days = 30
    settings.vat_applicable = False
    settings.vat_rate = None
    db.session.add(settings)
    db.session.flush()

    # Client porteur (institution) rattaché au transporteur — comme client_id=23 en prod.
    suffix = uuid.uuid4().hex[:6]
    carrier_user = User()
    carrier_user.username = f"clhda_{suffix}"
    carrier_user.email = f"clhda_{suffix}@test.ch"
    carrier_user.role = UserRole.client
    carrier_user.public_id = str(uuid.uuid4())
    carrier_user.set_password("password123", force_change=False)
    db.session.add(carrier_user)
    db.session.flush()

    carrier = Client()
    carrier.user_id = carrier_user.id
    carrier.company_id = transport.id
    carrier.is_institution = True
    carrier.institution_name = institution.name
    carrier.linked_institution_id = institution.id
    carrier.billing_address = institution.address
    db.session.add(carrier)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = "Anna"
    patient.last_name = "SBORDONE"
    patient.address = "Avenue de Vaudagne 47"
    patient.postal_code = "1217"
    patient.city = "Meyrin"
    db.session.add(patient)
    db.session.flush()
    bp = get_or_create_billing_party_for_institution_patient(
        company_id=transport.id, institution_patient=patient
    )

    booking = Booking()
    booking.company_id = transport.id
    booking.client_id = carrier.id
    booking.customer_name = "Anna SBORDONE"
    booking.pickup_location = "Chemin des Courbes 9, 1247, Anières"
    booking.dropoff_location = "Avenue de Vaudagne 47, Meyrin"
    booking.scheduled_time = datetime(2026, 9, 30, 9, 30)
    booking.completed_at = datetime(2026, 9, 30, 10, 1)
    booking.status = BookingStatus.COMPLETED.value
    booking.amount = Decimal("45.00")
    booking.billed_to_type = "patient"
    booking.billing_party_id = bp.id
    booking.billed_to_company_id = None
    booking.institution_patient_id = patient.id
    # Origine institution (Market LIRIE), contrôle institution pas encore fait.
    booking.billing_origin = "LIRIE_MARKETPLACE"
    booking.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    booking.institution_control_status = None
    db.session.add(booking)
    db.session.flush()
    db.session.commit()

    return {
        "transport": transport,
        "carrier": carrier,
        "patient": patient,
        "bp": bp,
        "booking": booking,
    }


@pytest.fixture
def world(db):
    return _make_world(db)


def _patient_item(world, *, now: datetime):
    res = list_billing_opportunities(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        now=now,
    )
    assert len(res.patient_items) == 1
    return res, res.patient_items[0]


def _preview_count(world, *, now: datetime) -> int:
    preview = build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        client_id=world["carrier"].id,
        institution_patient_id=world["patient"].id,
        now=now,
    )
    return preview.transports_count


def test_pending_market_leg_is_listed_as_held_not_billable(world):
    """Avant la libération : le patient reste visible, mais rien n'est annoncé facturable."""
    res, item = _patient_item(world, now=BEFORE_RELEASE)

    assert item.identity_status == "resolved"
    assert item.recipient_status == "ready"
    assert item.segments_count == 0
    assert item.transports_count == 0
    assert item.unbilled_total_amount == 0.0
    assert item.can_generate is False
    # La seule raison est le gate institution — pas « identité / destinataire ».
    assert item.blocked_reason == "pending_institution_validation"
    assert item.pending_validation_count == 1
    assert item.pending_validation_amount == 45.0
    assert item.pending_validation_release_at == "2026-10-01T00:00:00+02:00"
    assert item.disputed_count == 0
    assert res.total_draft_would_create == 0

    payload = opportunities_to_dict(res)["patient_payers"][0]
    assert payload["blocked_reason"] == "pending_institution_validation"
    assert payload["pending_validation_count"] == 1
    assert payload["pending_validation_amount"] == 45.0
    assert payload["pending_validation_release_at"] == "2026-10-01T00:00:00+02:00"

    # Parité avec la prévisualisation (même horloge).
    assert _preview_count(world, now=BEFORE_RELEASE) == 0


def test_generate_explains_institution_hold_instead_of_generic_not_found(world):
    _, item = _patient_item(world, now=BEFORE_RELEASE)
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"

    result = GenerateInvoiceUseCase(pdf_service=pdf).execute(
        GenerateInvoiceInput(
            company_id=world["transport"].id,
            client_id=world["carrier"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
            billing_opportunity_key=item.opportunity_key,
        ),
        now=BEFORE_RELEASE,
    )

    assert result.success is False
    message = str((result.error or {}).get("error") or "")
    assert "Aucune réservation trouvée pour cette période" not in message
    assert message == (
        "Aucune prestation facturable pour l'instant : 1 prestation en attente de "
        "validation par l'institution (Market LIRIE) — facturable après validation "
        "ou automatiquement dès le 01.10.2026."
    )


def test_auto_release_on_first_of_next_month_makes_it_billable(world):
    res, item = _patient_item(world, now=AT_RELEASE)

    assert item.segments_count == 1
    assert item.unbilled_total_amount == 45.0
    assert item.can_generate is True
    assert item.blocked_reason is None
    assert item.pending_validation_count == 0
    assert item.pending_validation_release_at is None
    assert res.total_draft_would_create == 1
    assert _preview_count(world, now=AT_RELEASE) == 1


def test_institution_validation_releases_immediately(db, world):
    booking = world["booking"]
    booking.institution_control_status = InstitutionBillingControlStatus.VALIDATED
    db.session.commit()

    _, item = _patient_item(world, now=BEFORE_RELEASE)
    assert item.segments_count == 1
    assert item.can_generate is True
    assert item.blocked_reason is None
    assert item.pending_validation_count == 0
    assert _preview_count(world, now=BEFORE_RELEASE) == 1


def test_disputed_market_leg_reported_as_disputed(db, world):
    booking = world["booking"]
    booking.institution_control_status = InstitutionBillingControlStatus.ANOMALY
    db.session.commit()

    _, item = _patient_item(world, now=AT_RELEASE)
    # Une contestation n'est jamais libérée automatiquement.
    assert item.segments_count == 0
    assert item.can_generate is False
    assert item.blocked_reason == "disputed"
    assert item.disputed_count == 1
    assert item.pending_validation_count == 0
    assert _preview_count(world, now=AT_RELEASE) == 0


def test_selecteur_exclut_la_course_retenue_par_le_controle(world, monkeypatch):
    monkeypatch.setattr("ext.redis_client", None)
    monkeypatch.setattr(
        "application.invoices.institution_invoice_eligibility._now_zurich",
        lambda now=None: now or BEFORE_RELEASE,
    )
    from application.invoices.invoice_candidates import list_patient_invoice_candidates

    payload = list_patient_invoice_candidates(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
    )
    assert payload["period"] == "2026-09"
    assert payload["patients"] == []


def test_selecteur_affiche_la_course_validee_puis_l_exclut_si_facturee(
    db, world, monkeypatch
):
    monkeypatch.setattr("ext.redis_client", None)
    from application.invoices.invoice_candidates import list_patient_invoice_candidates
    from models.enums import InvoiceLineType
    from models.invoice import Invoice, InvoiceLine

    booking = world["booking"]
    booking.institution_control_status = InstitutionBillingControlStatus.VALIDATED
    db.session.commit()

    def _load():
        return list_patient_invoice_candidates(
            company_id=world["transport"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
        )

    visible = _load()
    assert len(visible["patients"]) == 1
    assert visible["patients"][0]["billable_count"] == 1
    assert visible["patients"][0]["amount"] == 45.0

    invoice = Invoice()
    invoice.company_id = world["transport"].id
    invoice.client_id = world["carrier"].id
    invoice.period_year = PERIOD_YEAR
    invoice.period_month = PERIOD_MONTH
    invoice.invoice_number = f"T-{uuid.uuid4().hex[:8]}"
    invoice.due_date = datetime(2026, 10, 31, tzinfo=ZURICH)
    db.session.add(invoice)
    db.session.flush()

    line = InvoiceLine()
    line.invoice_id = invoice.id
    line.type = InvoiceLineType.RIDE
    line.description = "Course déjà facturée"
    line.qty = Decimal("1.00")
    line.unit_price = Decimal("45.00")
    line.line_total = Decimal("45.00")
    db.session.add(line)
    db.session.flush()

    booking.invoice_line_id = line.id
    db.session.commit()

    assert _load()["patients"] == []


def test_selecteur_ne_lit_pas_une_autre_entreprise(world, monkeypatch):
    monkeypatch.setattr("ext.redis_client", None)
    from application.invoices.invoice_candidates import list_patient_invoice_candidates

    payload = list_patient_invoice_candidates(
        company_id=int(world["transport"].id) + 99999,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
    )
    assert payload["patients"] == []
