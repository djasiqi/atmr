"""Seed déterministe du tenant documentation Institution (Lot 2).

Reset contrôlé : ne touche qu'au tenant docs identifié explicitement.
Aucune donnée réelle, aucun e-mail réel, aucune réservation LIRIE.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from sqlalchemy import or_

from ext import db
from models import (
    Institution,
    InstitutionNotification,
    InstitutionPatient,
    TransportRequest,
    TransportRequestLeg,
    User,
)
from models.enums import (
    BillingIntent,
    CarrierSource,
    InstitutionRole,
    MissionType,
    RequestStatus,
    ScheduledTimeType,
    UserRole,
)

logger = logging.getLogger(__name__)

DOCS_EMAIL_DOMAIN = "docs.lirie.local"
DOCS_TIMEZONE = ZoneInfo("Europe/Zurich")
DOCS_MISSION_DATE = date(2026, 3, 16)

DOCS_INSTITUTION_NAME = "Établissement de démonstration LIRIE"
DOCS_INSTITUTION_EMAIL = f"institution@{DOCS_EMAIL_DOMAIN}"
DOCS_INSTITUTION_PUBLIC_ID = "d0c50000-1613-4000-8000-000000000001"
DOCS_INSTITUTION_PHONE = "+41 22 000 00 10"
DOCS_INSTITUTION_ADDRESS = "Route de Démonstration 10, 1200 Genève"

DOCS_USER_EMAIL = f"docs.institution@{DOCS_EMAIL_DOMAIN}"
DOCS_USER_USERNAME = "docs.institution"
DOCS_USER_PUBLIC_ID = "d0c50000-1613-4000-8000-000000000010"
DOCS_USER_FIRST_NAME = "Test"
DOCS_USER_LAST_NAME = "Documentation"

DOCS_PICKUP = "Route de Démonstration 10, 1200 Genève"
DOCS_HOSPITAL = "Hôpital de démonstration, Rue Exemple Médical 20, 1200 Genève"
DOCS_MEDICAL_CENTER = "Centre médical Démo, Rue Exemple 30, 1200 Genève"

_BLOCKED_ENV = frozenset({"production", "prod", "staging"})
_BLOCKED_DB_MARKERS = ("prod", "staging")
_ALLOWED_DB_MARKERS = ("demo", "docs", "test", "atmr")
_ALLOWED_APP_ENV = frozenset({"", "development", "dev", "testing", "test", "local"})


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _database_name_from_url(database_url: str) -> str:
    parsed = urlparse(database_url)
    return parsed.path.lstrip("/").strip().lower()


def assert_institution_docs_seed_environment() -> None:
    """Refuse production / staging et les bases aux noms officiels."""
    env = (
        os.getenv("ENVIRONMENT")
        or os.getenv("FLASK_CONFIG")
        or os.getenv("FLASK_ENV")
        or os.getenv("APP_ENV")
        or ""
    ).strip().lower()
    if env in _BLOCKED_ENV:
        raise RuntimeError(
            "Seed institution-docs bloqué: environnement production/staging détecté."
        )

    database_url = (
        os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or ""
    ).strip()
    db_name = _database_name_from_url(database_url)
    blocked_name = bool(db_name) and any(
        marker in db_name for marker in _BLOCKED_DB_MARKERS
    )
    if blocked_name and "test" not in db_name:
        raise RuntimeError(
            f"Seed institution-docs bloqué: base '{db_name}' non autorisée "
            "(prod/staging)."
        )

    if _is_truthy(os.getenv("ALLOW_INSTITUTION_DOCS_SEED")) or _is_truthy(
        os.getenv("ALLOW_NON_DEMO_SEED")
    ):
        return

    try:
        from flask import current_app, has_app_context

        if has_app_context() and current_app.config.get("TESTING"):
            return
    except Exception:
        pass

    if (
        db_name
        and any(marker in db_name for marker in _ALLOWED_DB_MARKERS)
        and env in _ALLOWED_APP_ENV
    ):
        return

    raise RuntimeError(
        "Seed institution-docs bloqué: base non autorisée. "
        "Utilisez une base locale/demo/docs/test ou "
        "ALLOW_INSTITUTION_DOCS_SEED=true."
    )


def get_institution_docs_password() -> str:
    password = os.getenv("INSTITUTION_DOCS_PASSWORD") or ""
    if not password.strip():
        raise RuntimeError(
            "INSTITUTION_DOCS_PASSWORD doit être défini pour le seed "
            "institution-docs. Aucun mot de passe n'est hardcodé."
        )
    return password


def _docs_dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 16, hour, minute, tzinfo=DOCS_TIMEZONE)


def _docs_naive(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 16, hour, minute)


def _is_docs_email(value: str | None) -> bool:
    email = (value or "").strip().lower()
    return email.endswith(f"@{DOCS_EMAIL_DOMAIN}")


def _is_docs_institution(institution: Institution) -> bool:
    if institution.public_id == DOCS_INSTITUTION_PUBLIC_ID:
        return True
    if _is_docs_email(institution.contact_email):
        return True
    return (
        institution.name == DOCS_INSTITUTION_NAME
        and (institution.contact_email or "") == DOCS_INSTITUTION_EMAIL
    )


def _collect_docs_institutions() -> list[Institution]:
    found: list[Institution] = []
    seen: set[int] = set()

    for inst in Institution.query.filter(
        or_(
            Institution.public_id == DOCS_INSTITUTION_PUBLIC_ID,
            Institution.contact_email == DOCS_INSTITUTION_EMAIL,
        )
    ).all():
        if inst.id not in seen and _is_docs_institution(inst):
            found.append(inst)
            seen.add(inst.id)

    user = User.query.filter_by(email=DOCS_USER_EMAIL).first()
    if user and user.institution_id and user.institution_id not in seen:
        inst = db.session.get(Institution, user.institution_id)
        if inst is not None and _is_docs_institution(inst):
            found.append(inst)
            seen.add(inst.id)

    return found


def _collect_docs_user_ids(institution_ids: list[int]) -> list[int]:
    filters = [
        User.email == DOCS_USER_EMAIL,
        User.username == DOCS_USER_USERNAME,
        User.public_id == DOCS_USER_PUBLIC_ID,
    ]
    if institution_ids:
        filters.append(User.institution_id.in_(institution_ids))
    users = User.query.filter(or_(*filters)).all()
    ids: list[int] = []
    for user in users:
        email = (user.email or "").strip().lower()
        if email and not _is_docs_email(email) and email != DOCS_USER_EMAIL:
            raise RuntimeError(
                f"Seed institution-docs aborté: utilisateur hors domaine docs "
                f"rattaché au tenant ({email})."
            )
        ids.append(int(user.id))
    return ids


def _purge_docs_tenant(institutions: list[Institution], user_ids: list[int]) -> None:
    """Supprime uniquement le tenant docs et ses objets liés."""
    from models.control_plane import OrganizationMembership, PlatformOrganization
    from models.institution_api_key import InstitutionApiKey
    from models.institution_reserved_username import InstitutionReservedUsername
    from models.institution_settings import InstitutionSettings
    from models.institution_transport_preference import InstitutionTransportPreference
    from models.institution_user_audit_event import InstitutionUserAuditEvent
    from models.web_session import WebSession

    institution_ids = [int(inst.id) for inst in institutions]
    if not institution_ids and not user_ids:
        return

    if institution_ids:
        org_ids = [
            int(org.id)
            for org in PlatformOrganization.query.filter(
                PlatformOrganization.institution_id.in_(institution_ids)
            ).all()
        ]
        if org_ids:
            OrganizationMembership.query.filter(
                OrganizationMembership.organization_id.in_(org_ids)
            ).delete(synchronize_session=False)
            PlatformOrganization.query.filter(
                PlatformOrganization.id.in_(org_ids)
            ).delete(synchronize_session=False)

        InstitutionNotification.query.filter(
            InstitutionNotification.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        TransportRequest.query.filter(
            TransportRequest.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionPatient.query.filter(
            InstitutionPatient.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionSettings.query.filter(
            InstitutionSettings.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionTransportPreference.query.filter(
            InstitutionTransportPreference.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionApiKey.query.filter(
            InstitutionApiKey.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionUserAuditEvent.query.filter(
            InstitutionUserAuditEvent.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        InstitutionReservedUsername.query.filter(
            InstitutionReservedUsername.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)
        WebSession.query.filter(
            WebSession.institution_id.in_(institution_ids)
        ).delete(synchronize_session=False)

    if user_ids:
        WebSession.query.filter(WebSession.user_id.in_(user_ids)).delete(
            synchronize_session=False
        )
        User.query.filter(User.id.in_(user_ids)).delete(synchronize_session=False)

    for inst in institutions:
        db.session.delete(inst)
    db.session.flush()


def _create_institution() -> Institution:
    institution = Institution()
    institution.public_id = DOCS_INSTITUTION_PUBLIC_ID
    institution.name = DOCS_INSTITUTION_NAME
    institution.institution_type = "clinic"
    institution.address = DOCS_INSTITUTION_ADDRESS
    institution.contact_email = DOCS_INSTITUTION_EMAIL
    institution.contact_phone = DOCS_INSTITUTION_PHONE
    db.session.add(institution)
    db.session.flush()
    return institution


def _create_user(institution: Institution, password: str) -> User:
    user = User()
    user.public_id = DOCS_USER_PUBLIC_ID
    user.username = DOCS_USER_USERNAME
    user.email = DOCS_USER_EMAIL
    user.first_name = DOCS_USER_FIRST_NAME
    user.last_name = DOCS_USER_LAST_NAME
    user.role = UserRole.INSTITUTION
    user.institution_id = institution.id
    user.institution_role = InstitutionRole.ADMIN.value
    user.account_status = "active"
    user.set_password(password, force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _create_patient(
    institution: Institution,
    *,
    external_reference: str,
    public_id: str,
    first_name: str,
    last_name: str,
    dob: date,
    phone: str,
    address: str,
    notes: str | None = None,
) -> InstitutionPatient:
    patient = InstitutionPatient()
    patient.public_id = public_id
    patient.institution_id = institution.id
    patient.external_reference = external_reference
    patient.first_name = first_name
    patient.last_name = last_name
    patient.dob = dob
    patient.phone = phone
    patient.address = address
    patient.postal_code = "1200"
    patient.city = "Genève"
    patient.notes = notes
    db.session.add(patient)
    db.session.flush()
    return patient


def _add_simple_leg(
    request: TransportRequest,
    *,
    dropoff_establishment: str | None = None,
    dropoff_service: str | None = None,
) -> None:
    if not request.scheduled_time:
        return
    leg = TransportRequestLeg()
    leg.transport_request_id = request.id
    leg.sequence_index = 0
    leg.route_sequence_number = 1
    leg.pickup_location = request.pickup_location
    leg.dropoff_location = request.dropoff_location
    leg.dropoff_establishment = dropoff_establishment
    leg.dropoff_service = dropoff_service
    leg.scheduled_time = request.scheduled_time
    leg.time_confirmed = True
    db.session.add(leg)


def _create_request(
    institution: Institution,
    user: User,
    *,
    external_reference: str,
    public_id: str,
    patient: InstitutionPatient | None,
    mission_type: str,
    status: str,
    hour: int,
    minute: int,
    dropoff_location: str,
    delivery_description: str | None = None,
    billing_intent: str = BillingIntent.INSTITUTION.value,
    carrier_source: str = CarrierSource.LIRIE.value,
    mobility: dict[str, Any] | None = None,
    pickup_type: str = "institution",
    dropoff_type: str = "other",
    created_at: datetime | None = None,
    sent_at: datetime | None = None,
    accepted_at: datetime | None = None,
    assigned_externally_at: datetime | None = None,
    external_carrier: dict[str, str] | None = None,
    dropoff_establishment: str | None = None,
    dropoff_service: str | None = None,
) -> TransportRequest:
    request = TransportRequest()
    request.public_id = public_id
    request.institution_id = institution.id
    request.created_by_user_id = user.id
    request.created_by_display_name = f"{user.first_name} {user.last_name}"
    request.external_reference = external_reference
    request.patient_id = patient.id if patient else None
    request.mission_type = mission_type
    request.delivery_description = delivery_description
    request.mission_date = DOCS_MISSION_DATE
    request.scheduled_time = _docs_naive(hour, minute)
    request.pickup_time_confirmed = True
    request.scheduled_time_type = ScheduledTimeType.DEPARTURE.value
    request.pickup_location = DOCS_PICKUP
    request.dropoff_location = dropoff_location
    request.pickup_type = pickup_type
    request.dropoff_type = dropoff_type
    request.billing_intent = billing_intent
    request.status = status
    request.carrier_source = carrier_source
    request.mobility = mobility
    request.contact_on_site = {
        "requester_name": f"{DOCS_USER_FIRST_NAME} {DOCS_USER_LAST_NAME}",
        "requester_phone": DOCS_INSTITUTION_PHONE,
        "requester_service": "Documentation",
    }
    if created_at is not None:
        request.created_at = created_at
    if sent_at is not None:
        request.sent_at = sent_at
    if accepted_at is not None:
        request.accepted_at = accepted_at
    if assigned_externally_at is not None:
        request.assigned_externally_at = assigned_externally_at
        request.externalized_by_user_id = user.id
    if external_carrier:
        request.external_carrier_name = external_carrier.get("name")
        request.external_carrier_phone = external_carrier.get("phone")
        request.external_carrier_email = external_carrier.get("email")
        request.external_carrier_reason = external_carrier.get("reason")
    db.session.add(request)
    db.session.flush()
    _add_simple_leg(
        request,
        dropoff_establishment=dropoff_establishment,
        dropoff_service=dropoff_service,
    )
    return request


def _create_notification(
    institution: Institution,
    *,
    event_type: str,
    title: str,
    message: str,
    created_at: datetime,
    dedupe_key: str,
    metadata: dict[str, Any] | None = None,
) -> InstitutionNotification:
    notif = InstitutionNotification()
    notif.institution_id = institution.id
    notif.event_type = event_type
    notif.title = title
    notif.message = message
    notif.metadata_json = metadata or {}
    notif.dedupe_key = dedupe_key
    notif.is_read = False
    notif.created_at = created_at
    db.session.add(notif)
    return notif


def reset_and_seed_institution_docs(*, commit: bool = True) -> dict[str, Any]:
    """Reset du tenant docs uniquement, puis reconstruction exacte."""
    assert_institution_docs_seed_environment()
    password = get_institution_docs_password()

    institutions = _collect_docs_institutions()
    user_ids = _collect_docs_user_ids([int(inst.id) for inst in institutions])
    _purge_docs_tenant(institutions, user_ids)

    institution = _create_institution()
    user = _create_user(institution, password)

    patient_test = _create_patient(
        institution,
        external_reference="DOCS-PAT-001",
        public_id="d0c50000-1613-4000-8000-000000000021",
        first_name="TEST",
        last_name="Test",
        dob=date(1980, 1, 1),
        phone="+41 22 000 00 01",
        address="Rue Exemple 1",
        notes="E-mail fictif : test.test@docs.lirie.local",
    )
    patient_alice = _create_patient(
        institution,
        external_reference="DOCS-PAT-002",
        public_id="d0c50000-1613-4000-8000-000000000022",
        first_name="Alice",
        last_name="Exemple",
        dob=date(1945, 2, 2),
        phone="+41 22 000 00 02",
        address="Rue Exemple 2",
    )
    _create_patient(
        institution,
        external_reference="DOCS-PAT-003",
        public_id="d0c50000-1613-4000-8000-000000000023",
        first_name="Marc",
        last_name="Démonstration",
        dob=date(1950, 3, 3),
        phone="+41 22 000 00 03",
        address="Rue Exemple 3",
    )

    req_001 = _create_request(
        institution,
        user,
        external_reference="DOCS-REQ-001",
        public_id="d0c50000-1613-4000-8000-000000000031",
        patient=patient_test,
        mission_type=MissionType.PATIENT_TRANSPORT.value,
        status=RequestStatus.SENT.value,
        hour=10,
        minute=30,
        dropoff_location=DOCS_HOSPITAL,
        mobility={"wheelchair": True, "needs_assistance": True},
        created_at=_docs_dt(9, 20),
        sent_at=_docs_dt(9, 35),
        dropoff_establishment="Hôpital de démonstration",
        dropoff_service="Radiologie",
    )
    req_002 = _create_request(
        institution,
        user,
        external_reference="DOCS-REQ-002",
        public_id="d0c50000-1613-4000-8000-000000000032",
        patient=None,
        mission_type=MissionType.MATERIAL_DELIVERY.value,
        status=RequestStatus.SENT.value,
        hour=11,
        minute=45,
        dropoff_location=DOCS_HOSPITAL,
        delivery_description="Livraison de documents",
        created_at=_docs_dt(9, 40),
        sent_at=_docs_dt(9, 42),
        dropoff_establishment="Hôpital de démonstration",
    )
    req_003 = _create_request(
        institution,
        user,
        external_reference="DOCS-REQ-003",
        public_id="d0c50000-1613-4000-8000-000000000033",
        patient=patient_test,
        mission_type=MissionType.PATIENT_TRANSPORT.value,
        status=RequestStatus.EXTERNAL_ASSIGNED.value,
        hour=14,
        minute=0,
        dropoff_location=DOCS_MEDICAL_CENTER,
        carrier_source=CarrierSource.EXTERNAL.value,
        created_at=_docs_dt(8, 50),
        assigned_externally_at=_docs_dt(9, 0),
        external_carrier={
            "name": "Taxi Démo Genève",
            "phone": "+41 22 000 00 99",
            "email": f"externe@{DOCS_EMAIL_DOMAIN}",
            "reason": "Transporteur habituel",
        },
        dropoff_establishment="Centre médical Démo",
        dropoff_service="Consultation",
    )
    req_004 = _create_request(
        institution,
        user,
        external_reference="DOCS-REQ-004",
        public_id="d0c50000-1613-4000-8000-000000000034",
        patient=patient_alice,
        mission_type=MissionType.PATIENT_TRANSPORT.value,
        status=RequestStatus.ACCEPTED.value,
        hour=15,
        minute=30,
        dropoff_location=DOCS_HOSPITAL,
        created_at=_docs_dt(7, 50),
        sent_at=_docs_dt(7, 55),
        accepted_at=_docs_dt(8, 0),
        dropoff_establishment="Hôpital de démonstration",
        dropoff_service="Consultation",
    )

    _create_notification(
        institution,
        event_type="request_sent",
        title="Demande envoyée",
        message="TEST Test — RDV 16.03.2026 10:30",
        created_at=_docs_dt(9, 35),
        dedupe_key="docs:request_sent:DOCS-REQ-001",
        metadata={"request_id": req_001.id, "external_reference": "DOCS-REQ-001"},
    )
    _create_notification(
        institution,
        event_type="request_converted",
        title="Transport confirmé",
        message="Alice Exemple — départ confirmé 16.03.2026 15:30",
        created_at=_docs_dt(8, 0),
        dedupe_key="docs:request_converted:DOCS-REQ-004",
        metadata={"request_id": req_004.id, "external_reference": "DOCS-REQ-004"},
    )
    _create_notification(
        institution,
        event_type="booking_message",
        title="Nouveau message",
        message="Transporteur Démo — TEST Test",
        created_at=_docs_dt(7, 30),
        dedupe_key="docs:booking_message:DOCS-REQ-001",
        metadata={"request_id": req_001.id, "external_reference": "DOCS-REQ-001"},
    )

    if commit:
        db.session.commit()
    else:
        db.session.flush()

    summary = {
        "institution": institution.name,
        "institution_public_id": institution.public_id,
        "user": DOCS_USER_EMAIL,
        "role": InstitutionRole.ADMIN.value,
        "patients": 3,
        "requests": [
            "DOCS-REQ-001",
            "DOCS-REQ-002",
            "DOCS-REQ-003",
            "DOCS-REQ-004",
        ],
        "notifications": 3,
        "external_request_id": req_003.id,
        "delivery_request_id": req_002.id,
    }
    logger.info("[institution-docs] seed terminé: %s", summary)
    return summary
