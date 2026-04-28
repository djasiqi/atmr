from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_

from ext import db
from models import (
    Booking,
    BookingStatus,
    Client,
    ClientType,
    Company,
    DemoAccess,
    DemoRequest,
    DispatchMode,
    Driver,
    Institution,
    InstitutionPatient,
    Invoice,
    InvoiceStatus,
    TransportRequest,
    User,
    UserRole,
)
from models.enums import (
    BillingIntent,
    InstitutionRole,
    LocationType,
    ManagementMode,
    RequestStatus,
    ScheduledTimeType,
)
from security.refresh_token_service import revoke_all_user_tokens
from services.demo.dispatcher import send_demo_access_ready_email
from services.demo.seed_service import (
    ensure_demo_reference_dataset,
    reset_and_seed_demo_dataset,
)
from services.demo.soft_delete_guard import company_is_demo
from services.demo.utils import get_demo_default_password

logger = logging.getLogger(__name__)


def _get_trace_id() -> str:
    """Récupère le trace_id pour corrélation (safe hors contexte requête)."""
    try:
        from middleware.trace_id import get_trace_id
        return get_trace_id()
    except Exception:
        return "-"


DEMO_ACCESS_DURATION_HOURS = 48


MAGIC_LINK_TTL_MINUTES = 48 * 60  # 48h, aligné avec durée compte démo
SESSION_RESET_DEBOUNCE_SECONDS = 20
MIN_DEMO_CLIENTS = 8
MIN_DEMO_DRIVERS = 3

DEMO_CLIENT_IDENTITIES: list[tuple[str, str]] = [
    ("Aline", "Morel"),
    ("Karim", "Haddad"),
    ("Sophie", "Vuille"),
    ("Nicolas", "Bernasconi"),
    ("Lea", "Rochat"),
    ("Omar", "Bensaid"),
    ("Camille", "Fournier"),
    ("Mathis", "Perrin"),
]

DEMO_DRIVER_IDENTITIES: list[tuple[str, str]] = [
    ("Yanis", "Dubois"),
    ("Maya", "Schmidt"),
    ("Romain", "Favre"),
    ("Sarah", "Aubert"),
]

DEMO_DRIVER_GPS_POINTS: list[tuple[str, str]] = [
    ("46.2049", "6.1437"),
    ("46.2106", "6.1289"),
    ("46.1959", "6.1396"),
    ("46.2181", "6.1117"),
]

DEMO_PICKUP_ADDRESSES: list[tuple[str, str, str, str, str]] = [
    ("Rue de Carouge 58", "Geneve", "1205", "46.1937", "6.1450"),
    ("Avenue Wendt 5", "Geneve", "1203", "46.2145", "6.1269"),
    ("Rue de Lausanne 71", "Geneve", "1202", "46.2162", "6.1478"),
    ("Route de Meyrin 33", "Geneve", "1203", "46.2178", "6.1115"),
    ("Avenue de la Praille 35", "Carouge", "1227", "46.1810", "6.1300"),
    ("Rue de Lyon 93", "Geneve", "1203", "46.2104", "6.1212"),
]

DEMO_DROPOFF_LOCATIONS: list[str] = [
    "Hopital cantonal, Rue Gabrielle-Perret-Gentil 4, 1205 Geneve",
    "Clinique de Carouge, Avenue Cardinal-Mermillod 1, 1227 Carouge",
    "Centre de soins de Chene, Route de Chene 100, 1224 Chene-Bougeries",
    "Policlinique des Acacias, Rue des Epinettes 19, 1227 Les Acacias",
]

DEMO_INSTITUTION_PATIENT_IDENTITIES: list[tuple[str, str]] = [
    ("Nadia", "Berset"),
    ("Louis", "Carrel"),
    ("Myriam", "Dufaux"),
    ("Pascal", "Jaquet"),
]


class DemoAccessError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class DemoProvisionResult:
    demo_request: DemoRequest
    demo_access: DemoAccess
    magic_token: str
    email_sent: bool
    email_error: str | None = None
    reused_existing_access: bool = False
    provision_summary: dict[str, Any] | None = None


INSTITUTION_TYPES = {"institution", "ems", "clinic", "hospital", "curatorship"}
TRANSPORT_TYPES = {"transport_company", "transport"}
ROLE_ALIASES = {
    "institution_admin": InstitutionRole.ADMIN.value,
    "company_admin": "company_admin",
    "dispatcher_demo": "dispatcher_demo",
}


def _clean_text(value: Any, *, max_len: int) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return raw[:max_len]


def _normalize_seed_context(raw_context: Any, demo_request: DemoRequest) -> dict[str, Any]:
    context = raw_context if isinstance(raw_context, dict) else {}
    return {
        "volume_range": str(context.get("volume_range") or demo_request.volume_range or "").strip()
        or None,
        "timing": str(context.get("timing") or demo_request.timing or "").strip() or None,
        "preferred_slot": str(
            context.get("preferred_slot") or demo_request.preferred_slot or ""
        ).strip()
        or None,
        "preferred_period": str(
            context.get("preferred_period") or demo_request.preferred_period or ""
        ).strip()
        or None,
        "integration_required": str(
            context.get("integration_required") or demo_request.integration_required or ""
        ).strip()
        or None,
    }


def _normalize_provision_profile(
    demo_request: DemoRequest, raw_profile: dict[str, Any] | None
) -> dict[str, Any]:
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    demo_login_email = _build_demo_user_email(
        _clean_text(profile.get("demo_login_email"), max_len=255) or demo_request.email
    )
    organization_type = (
        _clean_text(profile.get("organization_type"), max_len=64)
        or str(demo_request.organization_type or "").strip().lower()
        or "institution"
    ).lower()
    if organization_type == "curatelle":
        organization_type = "curatorship"
    if organization_type not in INSTITUTION_TYPES.union(TRANSPORT_TYPES).union({"other"}):
        organization_type = str(demo_request.organization_type or "institution").strip().lower()

    user_first_name, user_last_name = _split_person_name(
        _clean_text(profile.get("user_first_name"), max_len=100)
        or _clean_text(profile.get("user_last_name"), max_len=100)
        or demo_request.name
    )
    if _clean_text(profile.get("user_first_name"), max_len=100):
        user_first_name = _clean_text(profile.get("user_first_name"), max_len=100) or user_first_name
    if _clean_text(profile.get("user_last_name"), max_len=100):
        user_last_name = _clean_text(profile.get("user_last_name"), max_len=100) or user_last_name

    request_comment = _clean_text(demo_request.comment, max_len=1200)
    visible_demo_notes = _clean_text(profile.get("visible_demo_notes"), max_len=1200)
    if not visible_demo_notes and request_comment:
        visible_demo_notes = request_comment

    workspace_seed_notes = _clean_text(profile.get("workspace_seed_notes"), max_len=1200)
    if not workspace_seed_notes:
        workspace_seed_notes = (
            f"timing={demo_request.timing}; volume={demo_request.volume_range or '-'}; "
            f"slot={demo_request.preferred_slot}; period={demo_request.preferred_period}"
        )[:1200]

    persona = _clean_text(profile.get("demo_persona"), max_len=64)
    if not persona:
        persona = (
            "institution"
            if organization_type in INSTITUTION_TYPES
            else "transport_company" if organization_type in TRANSPORT_TYPES else "generic"
        )

    return {
        "organization_name": _clean_text(profile.get("organization_name"), max_len=200)
        or _clean_text(demo_request.organization, max_len=200)
        or f"Demo Workspace {demo_request.id}",
        "organization_type": organization_type,
        "organization_address": _clean_text(profile.get("organization_address"), max_len=255),
        "organization_contact_phone": _clean_text(
            profile.get("organization_contact_phone") or demo_request.phone, max_len=50
        ),
        "organization_contact_email": _clean_text(
            profile.get("organization_contact_email") or demo_request.email, max_len=255
        ),
        "workspace_display_name": _clean_text(
            profile.get("workspace_display_name") or demo_request.organization, max_len=200
        ),
        "demo_login_email": demo_login_email,
        "user_first_name": user_first_name,
        "user_last_name": user_last_name,
        "user_phone": _clean_text(profile.get("user_phone") or demo_request.phone, max_len=50),
        "user_role": ROLE_ALIASES.get(
            str(profile.get("user_role") or "").strip().lower(),
            str(profile.get("user_role") or "").strip().lower() or None,
        ),
        "provision_template": _clean_text(profile.get("provision_template"), max_len=64)
        or (
            "institution_demo"
            if organization_type in INSTITUTION_TYPES
            else "transport_company_demo"
            if organization_type in TRANSPORT_TYPES
            else "generic_demo"
        ),
        "demo_persona": persona,
        "guide_variant": _clean_text(profile.get("guide_variant"), max_len=64)
        or (
            "institution_quickstart"
            if persona == "institution"
            else "transport_dispatch_quickstart"
            if persona == "transport_company"
            else "generic_quickstart"
        ),
        "seed_context": _normalize_seed_context(profile.get("seed_context"), demo_request),
        "internal_admin_notes": _clean_text(profile.get("internal_admin_notes"), max_len=1200),
        "visible_demo_notes": visible_demo_notes,
        "workspace_seed_notes": workspace_seed_notes,
    }


def _split_person_name(raw_name: str | None) -> tuple[str, str]:
    value = str(raw_name or "").strip()
    if not value:
        return "Demo", "Access"
    parts = [p for p in value.split() if p]
    if len(parts) == 1:
        return parts[0][:100], "Access"
    first = parts[0]
    last = " ".join(parts[1:])
    return first[:100], last[:100]


def _normalize_phone(raw_phone: str | None) -> str | None:
    value = str(raw_phone or "").strip()
    return value or None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_non_demo_seed_guard_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "seed demo bloque" in message and "base non-demo detectee" in message


def _hash_magic_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_magic_token() -> tuple[str, str, datetime]:
    plain = secrets.token_urlsafe(32)
    return plain, _hash_magic_token(plain), _utc_now() + timedelta(minutes=MAGIC_LINK_TTL_MINUTES)


def _build_demo_user_email(raw_email: str | None) -> str:
    """Isolate demo identities from platform accounts using a demo- prefix."""
    value = str(raw_email or "").strip().lower()
    if not value:
        token = secrets.token_hex(4)
        return f"demo-{token}@demo.local"

    if "@" not in value:
        return f"demo-{value}@demo.local"

    local, domain = value.split("@", 1)
    local = local or "user"
    domain = domain or "demo.local"
    if not local.startswith("demo-"):
        local = f"demo-{local}"
    # Keep enough room for domain, honoring User.email max length (255).
    max_local_len = max(1, 255 - len(domain) - 1)
    local = local[:max_local_len]
    return f"{local}@{domain}"


def _create_or_reuse_demo_user(
    demo_request: DemoRequest, provision_profile: dict[str, Any] | None = None
) -> User:
    profile = provision_profile or {}
    demo_email = _build_demo_user_email(profile.get("demo_login_email") or demo_request.email)
    first_name = str(
        profile.get("user_first_name") or _split_person_name(demo_request.name)[0]
    ).strip()[:100]
    last_name = str(
        profile.get("user_last_name") or _split_person_name(demo_request.name)[1]
    ).strip()[:100]
    normalized_phone = _normalize_phone(profile.get("user_phone") or demo_request.phone)
    existing = User.query.filter_by(email=demo_email).first()
    if existing:
        existing.first_name = first_name
        existing.last_name = last_name
        existing.phone = normalized_phone
        return existing

    token = secrets.token_hex(8)
    username = f"demo_{demo_request.id}_{token}"
    user = User()
    user.username = username[:100]
    user.first_name = first_name
    user.last_name = last_name
    user.email = demo_email
    user.phone = normalized_phone
    user.role = UserRole.client
    user.account_status = "active"
    user.set_password(secrets.token_urlsafe(24))
    db.session.add(user)
    db.session.flush()
    return user


def _resolve_demo_journey(org_type_raw: str | None) -> str:
    org_type = str(org_type_raw or "").strip().lower()
    if org_type in {"transport_company", "transport"}:
        return "transport"
    if org_type in {"institution", "ems", "clinic", "hospital", "curatorship"}:
        return "institution"
    return "generic"


def _apply_demo_profile(
    demo_request: DemoRequest,
    demo_user: User,
    provision_profile: dict[str, Any] | None = None,
) -> str:
    profile = provision_profile or {}
    journey = _resolve_demo_journey(
        profile.get("organization_type")
        or getattr(demo_request, "organization_type", None)
    )
    demo_contact_email = (
        _clean_text(profile.get("organization_contact_email"), max_len=255)
        or demo_user.email
        or _build_demo_user_email(demo_request.email)
    )
    demo_contact_phone = _normalize_phone(
        profile.get("organization_contact_phone") or demo_request.phone
    )
    org_name = _clean_text(profile.get("organization_name"), max_len=200) or (
        demo_request.organization or f"Demo Workspace {demo_request.id}"
    )
    org_address = _clean_text(profile.get("organization_address"), max_len=255)
    visible_demo_notes = _clean_text(profile.get("visible_demo_notes"), max_len=1200)
    user_role = str(profile.get("user_role") or "").strip().lower()

    if journey == "transport":
        demo_user.role = UserRole.COMPANY
        demo_user.institution_id = None
        demo_user.institution_role = None
        company = getattr(demo_user, "company", None)
        if not company:
            company = Company()
            company.name = (org_name or f"Demo Transport {demo_request.id}")[:100]
            company.user_id = demo_user.id
            company.contact_email = demo_contact_email
            company.contact_phone = demo_contact_phone
            company.is_approved = True
            company.dispatch_enabled = True
            company.service_area = "Geneve"
            if org_address:
                company.address = org_address
            db.session.add(company)
            db.session.flush()
        else:
            company.name = (org_name or company.name or f"Demo Transport {demo_request.id}")[:100]
            company.contact_email = demo_contact_email
            company.contact_phone = demo_contact_phone
            company.is_approved = True
            company.dispatch_enabled = True
            if org_address:
                company.address = org_address
        company.dispatch_mode = DispatchMode.MANUAL
        return journey

    if journey == "institution":
        demo_user.role = UserRole.INSTITUTION
        demo_user.institution_role = (
            InstitutionRole.REQUESTER.value
            if user_role == "institution_requester"
            else InstitutionRole.ADMIN.value
        )
        institution = demo_user.institution
        if not institution:
            institution = Institution()
            institution.name = (org_name or f"Demo Institution {demo_request.id}")[:200]
            institution.institution_type = str(
                profile.get("organization_type") or demo_request.organization_type or "institution"
            )[:50]
            institution.contact_email = demo_contact_email
            institution.contact_phone = demo_contact_phone
            institution.address = org_address
            institution.notes = visible_demo_notes
            db.session.add(institution)
            db.session.flush()
            demo_user.institution_id = institution.id
        else:
            institution.name = (org_name or institution.name or f"Demo Institution {demo_request.id}")[:200]
            institution.contact_email = demo_contact_email
            institution.contact_phone = demo_contact_phone
            if org_address:
                institution.address = org_address
            if visible_demo_notes:
                institution.notes = visible_demo_notes
        return journey

    demo_user.role = UserRole.CLIENT
    demo_user.institution_id = None
    demo_user.institution_role = None
    return journey


def _seed_scale_from_context(seed_context: dict[str, Any] | None) -> tuple[int, int]:
    context = seed_context or {}
    volume_range = str(context.get("volume_range") or "").strip().lower()
    if volume_range == "100_plus":
        return 5, 20
    if volume_range == "20_100":
        return 4, 14
    if volume_range == "5_20":
        return 3, 10
    return MIN_DEMO_DRIVERS, MIN_DEMO_CLIENTS


def _seed_transport_demo_workspace(
    demo_request: DemoRequest,
    company: Company,
    *,
    seed_context: dict[str, Any] | None = None,
) -> None:
    now = _utc_now()
    base_email = str(demo_request.email or "demo.user@demo.local").strip().lower()
    local_part = base_email.split("@", 1)[0] or "demo.user"
    drivers_target, clients_target = _seed_scale_from_context(seed_context)

    drivers: list[Driver] = []
    for idx in range(drivers_target):
        driver_first_name, driver_last_name = DEMO_DRIVER_IDENTITIES[
            idx % len(DEMO_DRIVER_IDENTITIES)
        ]
        driver_email = f"demo-{local_part}.driver{idx + 1}@demo.local"
        driver_user = User.query.filter_by(email=driver_email).first()
        if not driver_user:
            driver_user = User()
            driver_user.email = driver_email
            driver_user.username = f"demo_driver_{company.id}_{idx + 1}_{secrets.token_hex(2)}"[:100]
        driver_user.role = UserRole.DRIVER
        driver_user.first_name = driver_first_name
        driver_user.last_name = driver_last_name
        driver_user.account_status = "active"
        driver_user.set_password(get_demo_default_password())
        db.session.add(driver_user)
        db.session.flush()

        driver = Driver.query.filter_by(user_id=driver_user.id).first()
        if not driver:
            driver = Driver()
            driver.user_id = driver_user.id
        driver.company_id = company.id
        driver.vehicle_assigned = f"Vehicule Demo {idx + 1}"
        driver.brand = "LIRIE Demo"
        driver.license_plate = f"DEMO-{company.id:03d}-{idx + 1:02d}"
        driver.is_active = True
        driver.is_available = idx != 0
        lat_raw, lon_raw = DEMO_DRIVER_GPS_POINTS[idx % len(DEMO_DRIVER_GPS_POINTS)]
        driver.latitude = Decimal(lat_raw)
        driver.longitude = Decimal(lon_raw)
        db.session.add(driver)
        db.session.flush()
        drivers.append(driver)

    clients: list[Client] = []
    for idx in range(clients_target):
        patient_first_name, patient_last_name = DEMO_CLIENT_IDENTITIES[
            idx % len(DEMO_CLIENT_IDENTITIES)
        ]
        patient_email = f"demo-{local_part}.patient{idx + 1}@demo.local"
        patient_user = User.query.filter_by(email=patient_email).first()
        if not patient_user:
            patient_user = User()
            patient_user.email = patient_email
            patient_user.username = f"demo_patient_{company.id}_{idx + 1}_{secrets.token_hex(2)}"[:100]
        patient_user.role = UserRole.CLIENT
        patient_user.first_name = patient_first_name
        patient_user.last_name = patient_last_name
        patient_user.account_status = "active"
        patient_user.set_password(get_demo_default_password())
        db.session.add(patient_user)
        db.session.flush()

        client = Client.query.filter_by(user_id=patient_user.id, company_id=company.id).first()
        if not client:
            client = Client()
            client.user_id = patient_user.id
            client.company_id = company.id

        address_row = DEMO_PICKUP_ADDRESSES[idx % len(DEMO_PICKUP_ADDRESSES)]
        domicile_address, domicile_city, domicile_zip, lat_raw, lon_raw = address_row
        domicile_lat = Decimal(lat_raw)
        domicile_lon = Decimal(lon_raw)

        client.contact_email = patient_user.email
        client.contact_phone = f"+41 79 000 0{idx:02d}"
        client.client_type = ClientType.TRANSPORT
        client.management_mode = ManagementMode.MANAGED
        client.domicile_address = domicile_address
        client.domicile_city = domicile_city
        client.domicile_zip = domicile_zip
        client.domicile_lat = domicile_lat
        client.domicile_lon = domicile_lon
        client.billing_address = domicile_address
        client.billing_lat = domicile_lat
        client.billing_lon = domicile_lon
        client.door_code = f"A{idx + 1:02d}B"
        client.floor = str((idx % 5) + 1)
        client.access_notes = "Ascenseur disponible. Sonner a l'interphone principal."
        client.gp_name = f"Dr {patient_last_name}"
        client.gp_phone = "+41 22 555 00 00"
        client.default_billed_to_type = "patient"
        client.default_billed_to_contact = (
            f"{patient_user.first_name or 'Patient'} {patient_user.last_name or ''}".strip()
        )
        client.residence_facility = f"Residence Les Tilleuls {((idx % 3) + 1)}"
        client.is_active = True
        db.session.add(client)
        db.session.flush()
        clients.append(client)

    existing_bookings = Booking.query.filter_by(company_id=company.id).count()
    if existing_bookings > 0:
        return

    statuses = [
        BookingStatus.ASSIGNED,
        BookingStatus.IN_PROGRESS,
        BookingStatus.ACCEPTED,
        BookingStatus.PENDING,
        BookingStatus.COMPLETED,
        BookingStatus.COMPLETED,
        BookingStatus.ASSIGNED,
        BookingStatus.PENDING,
        BookingStatus.ACCEPTED,
        BookingStatus.COMPLETED,
    ]
    for idx, status in enumerate(statuses):
        client = clients[idx % len(clients)]
        driver = drivers[idx % len(drivers)] if status != BookingStatus.PENDING else None
        scheduled = (now + timedelta(hours=(idx - 2))).replace(tzinfo=None, second=0, microsecond=0)
        booking = Booking()
        booking.user_id = client.user_id
        booking.client_id = client.id
        booking.company_id = company.id
        booking.driver_id = driver.id if driver else None
        booking.customer_name = f"{client.user.first_name} {client.user.last_name}".strip()
        booking.pickup_location = client.domicile_address
        booking.dropoff_location = DEMO_DROPOFF_LOCATIONS[
            idx % len(DEMO_DROPOFF_LOCATIONS)
        ]
        booking.scheduled_time = scheduled
        booking.amount = float(Decimal("55.00") + Decimal(idx))
        booking.status = status
        if status == BookingStatus.COMPLETED:
            booking.completed_at = now - timedelta(hours=max(1, idx))
        db.session.add(booking)

    for idx in range(2):
        client = clients[idx]
        invoice_number = f"DEMO-{company.id}-{now.strftime('%Y%m')}-{idx + 1:03d}"
        invoice = Invoice.query.filter_by(
            company_id=company.id, invoice_number=invoice_number
        ).first()
        if not invoice:
            invoice = Invoice()
            invoice.company_id = company.id
            invoice.client_id = client.id
            invoice.period_month = now.month
            invoice.period_year = now.year
            invoice.invoice_number = invoice_number
            amount = Decimal("180.00") + Decimal(idx * 35)
            invoice.currency = "CHF"
            invoice.subtotal_amount = amount
            invoice.total_amount = amount
            invoice.amount_paid = Decimal("0.00")
            invoice.balance_due = amount
            invoice.issued_at = now - timedelta(days=idx + 1)
            invoice.due_date = now + timedelta(days=30 - idx)
            invoice.status = InvoiceStatus.SENT if idx == 0 else InvoiceStatus.DRAFT
            db.session.add(invoice)


def _seed_institution_demo_workspace(
    demo_request: DemoRequest,
    demo_user: User,
    institution: Institution,
    *,
    seed_context: dict[str, Any] | None = None,
    visible_demo_notes: str | None = None,
) -> None:
    now = _utc_now()
    base_email = str(demo_request.email or "").strip().lower()
    local_part = base_email.split("@", 1)[0] or "demo.user"
    _drivers_target, patients_target = _seed_scale_from_context(seed_context)
    identity_count = max(1, min(patients_target, len(DEMO_INSTITUTION_PATIENT_IDENTITIES)))

    patients: list[InstitutionPatient] = []
    for idx, (first_name, last_name) in enumerate(
        DEMO_INSTITUTION_PATIENT_IDENTITIES[:identity_count]
    ):
        external_reference = f"INST-DEMO-{institution.id}-{idx + 1:03d}"
        patient = InstitutionPatient.query.filter_by(
            institution_id=institution.id,
            external_reference=external_reference,
        ).first()
        if not patient:
            patient = InstitutionPatient()
            patient.external_reference = external_reference

        patient.institution_id = institution.id
        patient.first_name = first_name
        patient.last_name = last_name
        patient.address = DEMO_PICKUP_ADDRESSES[idx % len(DEMO_PICKUP_ADDRESSES)][0]
        patient.city = "Geneve"
        patient.postal_code = DEMO_PICKUP_ADDRESSES[idx % len(DEMO_PICKUP_ADDRESSES)][2]
        patient.phone = f"+41 79 000 10{idx + 1}"
        patient.residence_name = institution.name
        patient.notes = "Patient demo institution - donnees virtuelles."
        if first_name == "Pascal" and last_name == "Jaquet":
            # Persona dédiée demandée: patient sous curatelle OPAD.
            patient.address = "Route de Meyrin 33, 1203, Geneve"
            patient.city = "Geneve"
            patient.postal_code = "1203"
            patient.phone = "+41 79 000 104"
            patient.residence_name = "Ems Les Marroniers"
            patient.has_guardianship = True
            patient.guardianship_type = "opad"
            patient.guardian_organization = "OPAD Geneve"
            patient.notes = "Patient demo sous curatelle OPAD - donnees virtuelles."
        db.session.add(patient)
        db.session.flush()
        patients.append(patient)

    # Reutiliser une entreprise demo existante pour les demandes converties.
    # CRITIQUE: Ne jamais utiliser une entreprise réelle - uniquement une entreprise démo
    # (contact_email @demo.lirie.ch, @demo.local, ou demo-*).
    _company_contact_demo_filter = or_(
        func.lower(func.coalesce(Company.contact_email, "")).like("%@demo.lirie.ch"),
        func.lower(func.coalesce(Company.contact_email, "")).like("%@demo.local"),
        func.lower(func.coalesce(Company.contact_email, "")).like("demo-%@%"),
    )
    _owner_user_demo_filter = or_(
        func.lower(func.coalesce(User.email, "")).like("%@demo.lirie.ch"),
        func.lower(func.coalesce(User.email, "")).like("%@demo.local"),
        func.lower(func.coalesce(User.email, "")).like("demo-%@%"),
    )
    demo_company = (
        Company.query.outerjoin(User, Company.user_id == User.id)
        .filter(or_(_company_contact_demo_filter, _owner_user_demo_filter))
        .order_by(Company.id.desc())
        .first()
    )
    if not demo_company:
        # Filet de sécurité: vérifier via helper métier.
        for c in Company.query.options(db.joinedload(Company.user)).yield_per(200):
            if company_is_demo(c):
                demo_company = c
                break
    if not demo_company:
        # Dernier recours: créer une entreprise démo dédiée pour l'institution.
        # Cela évite un 500 si le dataset partagé n'a pas encore été initialisé.
        owner_email = f"demo-inst-company-{institution.id}@demo.local"
        owner = User.query.filter_by(email=owner_email).first()
        if not owner:
            owner = User()
            owner.email = owner_email
            owner.username = (
                f"demo_inst_company_{institution.id}_{secrets.token_hex(3)}"[:100]
            )
            owner.role = UserRole.COMPANY
            owner.first_name = "Compte"
            owner.last_name = "Demo Institution"
            owner.account_status = "active"
            owner.set_password(get_demo_default_password())
            db.session.add(owner)
            db.session.flush()

        demo_company = Company.query.filter_by(user_id=owner.id).first()
        if not demo_company:
            demo_company = Company()
            demo_company.user_id = owner.id
            demo_company.name = (
                str(demo_request.organization_name or "").strip()
                or f"Demo Transport Institution {institution.id}"
            )[:100]
            demo_company.contact_email = owner_email
            demo_company.contact_phone = "+41 22 000 00 00"
            demo_company.service_area = "Geneve"
            demo_company.is_approved = True
            demo_company.dispatch_enabled = True
            demo_company.dispatch_mode = DispatchMode.MANUAL
            db.session.add(demo_company)
            db.session.flush()
    demo_driver = None
    demo_client = None
    if demo_company:
        demo_driver = Driver.query.filter_by(company_id=demo_company.id).first()
        # client_id est NOT NULL en base : créer ou réutiliser un Client pour les bookings institution
        demo_client_email = f"demo-inst-{institution.id}@demo.local"
        demo_client_user = User.query.filter_by(email=demo_client_email).first()
        if not demo_client_user:
            demo_client_user = User()
            demo_client_user.email = demo_client_email
            demo_client_user.username = f"demo_inst_client_{institution.id}_{secrets.token_hex(4)}"[:100]
            demo_client_user.role = UserRole.CLIENT
            demo_client_user.first_name = "Patient"
            demo_client_user.last_name = "Demo Institution"
            demo_client_user.account_status = "active"
            demo_client_user.set_password(get_demo_default_password())
            db.session.add(demo_client_user)
            db.session.flush()
        demo_client = Client.query.filter_by(
            user_id=demo_client_user.id, company_id=demo_company.id
        ).first()
        if not demo_client:
            demo_client = Client()
            demo_client.user_id = demo_client_user.id
            demo_client.company_id = demo_company.id
            demo_client.client_type = ClientType.TRANSPORT
            demo_client.management_mode = ManagementMode.MANAGED
            demo_client.contact_email = demo_client_user.email
            demo_client.domicile_address = DEMO_PICKUP_ADDRESSES[0][0]
            demo_client.domicile_city = "Geneve"
            demo_client.domicile_zip = DEMO_PICKUP_ADDRESSES[0][2]
            demo_client.domicile_lat = Decimal(DEMO_PICKUP_ADDRESSES[0][3])
            demo_client.domicile_lon = Decimal(DEMO_PICKUP_ADDRESSES[0][4])
            demo_client.default_billed_to_type = "patient"
            db.session.add(demo_client)
            db.session.flush()

    requests_payload = [
        {
            "status": RequestStatus.SENT.value,
            "hours_offset": 1,
            "pickup_idx": 0,
            "dropoff_idx": 0,
            "booking_status": None,
            "billing_intent": BillingIntent.INSTITUTION.value,
        },
        {
            "status": RequestStatus.SENT.value,
            "hours_offset": 3,
            "pickup_idx": 1,
            "dropoff_idx": 1,
            "booking_status": None,
            "billing_intent": BillingIntent.PATIENT.value,
            # Force >2h pour remonter en "attention".
            "created_hours_ago": 4,
        },
        {
            "status": RequestStatus.CONVERTED.value,
            "hours_offset": 2,
            "pickup_idx": 2,
            "dropoff_idx": 2,
            "booking_status": BookingStatus.IN_PROGRESS,
            "billing_intent": BillingIntent.INSTITUTION.value,
        },
        {
            "status": RequestStatus.CONVERTED.value,
            "hours_offset": -2,
            "pickup_idx": 3,
            "dropoff_idx": 3,
            "booking_status": BookingStatus.COMPLETED,
            "billing_intent": BillingIntent.PATIENT.value,
        },
    ]

    for idx, payload in enumerate(requests_payload):
        ext_ref = f"INST-{local_part.upper()}-{idx + 1:03d}"
        if TransportRequest.find_by_external_reference(institution.id, ext_ref):
            continue  # Idempotence: ne pas réinsérer une external_reference existante

        patient = patients[idx % len(patients)]
        pickup = DEMO_PICKUP_ADDRESSES[payload["pickup_idx"] % len(DEMO_PICKUP_ADDRESSES)]
        scheduled = (now + timedelta(hours=payload["hours_offset"])).replace(second=0, microsecond=0)

        booking = None
        if payload["booking_status"] is not None and demo_client is not None:
            booking = Booking()
            booking.user_id = demo_user.id
            booking.client_id = demo_client.id
            booking.company_id = demo_company.id
            booking.driver_id = demo_driver.id if demo_driver else None
            booking.customer_name = f"{patient.first_name} {patient.last_name}"
            booking.pickup_location = pickup[0]
            booking.dropoff_location = DEMO_DROPOFF_LOCATIONS[payload["dropoff_idx"] % len(DEMO_DROPOFF_LOCATIONS)]
            booking.scheduled_time = scheduled.replace(tzinfo=None)
            booking.amount = float(Decimal("62.00") + Decimal(idx))
            booking.status = payload["booking_status"]
            if payload["booking_status"] == BookingStatus.COMPLETED:
                booking.completed_at = now - timedelta(minutes=30)
            db.session.add(booking)
            db.session.flush()

        request_obj = TransportRequest()
        request_obj.institution_id = institution.id
        request_obj.created_by_user_id = demo_user.id
        request_obj.external_reference = ext_ref
        request_obj.patient_id = patient.id
        request_obj.mission_type = "patient_transport"
        request_obj.scheduled_time = scheduled
        request_obj.scheduled_time_type = ScheduledTimeType.DEPARTURE.value
        request_obj.pickup_location = pickup[0]
        request_obj.pickup_type = LocationType.DOMICILE.value
        request_obj.dropoff_location = DEMO_DROPOFF_LOCATIONS[payload["dropoff_idx"] % len(DEMO_DROPOFF_LOCATIONS)]
        request_obj.dropoff_type = LocationType.INSTITUTION.value
        request_obj.billing_intent = payload["billing_intent"]
        request_obj.status = payload["status"]
        request_obj.notes = "Demande virtuelle prechargee pour presentation institution."
        request_obj.booking_id = booking.id if booking else None
        request_obj.accepted_by_company_id = demo_company.id if booking and demo_company else None
        if payload["status"] == RequestStatus.SENT.value:
            request_obj.sent_at = now - timedelta(hours=1)
        if payload.get("created_hours_ago"):
            request_obj.created_at = now - timedelta(hours=payload["created_hours_ago"])
            request_obj.updated_at = now - timedelta(hours=payload["created_hours_ago"])
        db.session.add(request_obj)
    if visible_demo_notes:
        institution.notes = visible_demo_notes[:1200]


def _ensure_demo_workspace_seeded(
    demo_request: DemoRequest,
    demo_user: User,
    provision_profile: dict[str, Any] | None = None,
) -> None:
    role = (
        demo_user.role.value
        if hasattr(demo_user.role, "value")
        else str(demo_user.role or "")
    ).lower()
    if role not in {"company", "institution"}:
        return
    demo_request_id = getattr(demo_request, "id", None)
    company = getattr(demo_user, "company", None)
    institution = getattr(demo_user, "institution", None)
    # SQLAlchemy peut laisser la relation non résolue alors que la FK est déjà posée
    # (flush dans le même request) : le seed serait alors ignoré sans erreur.
    if role == "company" and company is None and getattr(demo_user, "id", None):
        logger.warning(
            "[demo_seed_diag] company relationship empty, fallback query user_id=%s",
            demo_user.id,
        )
        company = Company.query.filter_by(user_id=demo_user.id).first()
    if role == "institution" and institution is None and getattr(
        demo_user, "institution_id", None
    ):
        logger.warning(
            (
                "[demo_seed_diag] institution relationship empty before FK fallback "
                "user_id=%s institution_id=%s"
            ),
            getattr(demo_user, "id", None),
            demo_user.institution_id,
        )
        institution = db.session.get(Institution, demo_user.institution_id)
    if role == "institution" and institution is not None:
        logger.warning(
            "[demo_seed_diag] institution ready for seed institution_id=%s",
            institution.id,
        )
    elif role == "institution":
        logger.error(
            (
                "[demo_seed_diag] institution still missing after fallback "
                "user_id=%s institution_id=%s"
            ),
            getattr(demo_user, "id", None),
            getattr(demo_user, "institution_id", None),
        )
    company_id = getattr(company, "id", None)
    institution_id = getattr(institution, "id", None)
    user_id = getattr(demo_user, "id", None)
    seed_context = (provision_profile or {}).get("seed_context") or {}
    visible_demo_notes = (provision_profile or {}).get("visible_demo_notes")
    try:
        with db.session.begin_nested():
            if role == "company" and company:
                _seed_transport_demo_workspace(
                    demo_request, company, seed_context=seed_context
                )
            if role == "institution" and institution:
                _seed_institution_demo_workspace(
                    demo_request,
                    demo_user,
                    institution,
                    seed_context=seed_context,
                    visible_demo_notes=visible_demo_notes,
                )
            db.session.flush()
    except Exception as exc:
        logger.exception(
            "[demo_access] échec initialisation données démo (bookings, clients, etc.)",
            extra={
                "event_type": "demo_seed_failed",
                "demo_request_id": demo_request_id,
                "company_id": company_id,
                "institution_id": institution_id,
                "user_id": user_id,
                "journey_role": role,
                "trace_id": _get_trace_id(),
            },
        )
        raise DemoAccessError(
            "demo_seed_failed",
            "Impossible d'initialiser les données de démonstration. Veuillez réessayer ou contacter l'équipe.",
            status_code=500,
        ) from exc


def _disable_demo_user(user_id: int | None) -> None:
    if not user_id:
        return
    user = db.session.get(User, user_id)
    if not user:
        return

    # Ne jamais désactiver un compte plateforme "réel" par les routines démo.
    # Seuls les comptes isolés démo (email préfixé demo-) peuvent être désactivés ici.
    email = str(getattr(user, "email", "") or "").strip().lower()
    if not email.startswith("demo-"):
        logger.warning(
            "[demo_access] skip disable for non-demo account",
            extra={
                "event_type": "demo_access_disable_skipped",
                "user_id": user.id,
                "email": email,
                "role": str(getattr(user, "role", "")),
            },
        )
        return

    user.account_status = "disabled"
    try:
        revoke_all_user_tokens(user_id=user.id, reason="demo_access_disabled")
    except Exception:
        logger.exception("[demo_access] refresh token revocation failed")


def _canonical_request_email_from_demo_login(email: str | None) -> str:
    """Convertit demo-foo@bar vers foo@bar pour retrouver la DemoRequest source."""
    value = str(email or "").strip().lower()
    if not value or "@" not in value:
        return value
    local, domain = value.split("@", 1)
    if local.startswith("demo-"):
        local = local[len("demo-") :]
    return f"{local}@{domain}" if local else value


def enforce_demo_user_access_validity(  # noqa: PLR0911
    user: User | None,
) -> tuple[bool, str | None]:
    """Garantit qu'un compte demo-* reste valide uniquement pendant sa fenêtre d'accès.

    Retourne (True, None) pour les comptes non-demo ou les accès encore valides.
    En cas d'accès expiré/invalide, désactive le compte demo et retourne (False, message).
    """
    if not user:
        return False, "Compte introuvable."

    email = str(getattr(user, "email", "") or "").strip().lower()
    if not email.startswith("demo-"):
        return True, None

    now = _utc_now()
    access = (
        DemoAccess.query.filter(DemoAccess.demo_user_id == user.id)
        .order_by(DemoAccess.created_at.desc())
        .first()
    )

    if not access:
        # Tolérance legacy: certaines lignes DemoAccess historiques ne pointent pas encore
        # vers demo_user_id. On tente une réconciliation par email source de DemoRequest.
        request_email = _canonical_request_email_from_demo_login(email)
        if request_email:
            access = (
                DemoAccess.query.join(
                    DemoRequest, DemoAccess.demo_request_id == DemoRequest.id
                )
                .filter(DemoRequest.email == request_email)
                .order_by(DemoAccess.created_at.desc())
                .first()
            )
            if access:
                access.demo_user_id = user.id
                db.session.add(access)
                db.session.commit()
                logger.info(
                    "[demo_access] reconciled access->user binding",
                    extra={
                        "event_type": "demo_access_reconciled_binding",
                        "demo_access_id": access.id,
                        "demo_request_id": access.demo_request_id,
                        "user_id": user.id,
                        "email": email,
                    },
                )

    if not access:
        # Ne pas désactiver brutalement le compte sur un simple mismatch de binding.
        return False, "Acces demo introuvable ou invalide."

    if (
        access.status == "active"
    ):
        # Tolérance legacy: si demo_expires_at est absent, on le régénère.
        if access.demo_expires_at is None:
            base = access.provisioned_at or now
            access.demo_expires_at = base + timedelta(hours=DEMO_ACCESS_DURATION_HOURS)
            db.session.add(access)
            db.session.commit()
            return True, None
        if access.demo_expires_at > now:
            return True, None

    # Si l'accès courant n'est plus actif, tenter un fallback vers un accès actif de la même demande.
    if access.status != "active":
        active_access = _get_active_access_for_request(access.demo_request_id)
        if active_access:
            if active_access.demo_user_id != user.id:
                active_access.demo_user_id = user.id
                db.session.add(active_access)
                db.session.commit()
            if active_access.demo_expires_at is None or active_access.demo_expires_at > now:
                return True, None

    if access.status == "active":
        _apply_expiration(access, source="runtime_auth")
    else:
        _disable_demo_user(user.id)

    db.session.commit()
    return False, "Acces demo expire. Merci de demander un nouveau lien."


def _assert_request_exists(demo_request_id: int) -> DemoRequest:
    demo_request = db.session.get(DemoRequest, demo_request_id)
    if not demo_request:
        raise DemoAccessError(
            "request_not_found",
            "La demande de demo est introuvable.",
            status_code=404,
        )
    return demo_request


def _assert_access_exists(access_id: int) -> DemoAccess:
    access = db.session.get(DemoAccess, access_id)
    if not access:
        raise DemoAccessError(
            "access_not_found",
            "L'acces demo est introuvable.",
            status_code=404,
        )
    return access


def _get_active_access_for_request(demo_request_id: int) -> DemoAccess | None:
    return (
        DemoAccess.query.filter_by(demo_request_id=demo_request_id, status="active")
        .order_by(DemoAccess.created_at.desc())
        .first()
    )


def _ensure_access_is_active(access: DemoAccess) -> None:
    if access.status == "expired":
        raise DemoAccessError("access_expired", "Cet acces demo est expire.", status_code=409)
    if access.status == "revoked":
        raise DemoAccessError("access_revoked", "Cet acces demo a ete revoque.", status_code=409)
    if access.status != "active":
        raise DemoAccessError(
            "no_active_access",
            "Aucun acces actif disponible pour cette operation.",
            status_code=409,
        )
    if access.demo_expires_at and access.demo_expires_at <= _utc_now():
        raise DemoAccessError("access_expired", "Cet acces demo est expire.", status_code=409)


def _apply_expiration(access: DemoAccess, *, source: str = "scheduler") -> DemoAccess:
    if access.status != "active":
        return access
    now = _utc_now()
    access.status = "expired"
    access.expired_at = now
    access.magic_token_hash = None
    access.magic_token_expires_at = None
    _disable_demo_user(access.demo_user_id)
    logger.info(
        "[demo_access] expired",
        extra={
            "event_type": "demo_access_expired",
            "demo_request_id": access.demo_request_id,
            "demo_access_id": access.id,
            "actor_type": source,
            "actor_id": None,
        },
    )
    return access


def _reset_demo_dataset_on_session_start(access: DemoAccess) -> None:
    """Reset complet du dataset partagé au début d'une session démo."""
    allow_non_demo = os.getenv("ALLOW_NON_DEMO_SEED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_non_demo:
        try:
            reset_and_seed_demo_dataset(profile_name="sales", reset=True)
        except Exception as exc:
            if _is_non_demo_seed_guard_error(exc):
                logger.warning(
                    "[demo_access] full demo reset skipped: non-demo database guard active",
                    extra={
                        "event_type": "demo_reset_skipped_non_demo_guard",
                        "demo_access_id": access.id,
                        "demo_request_id": access.demo_request_id,
                    },
                )
            else:
                logger.exception("[demo_access] full demo reset failed on session start")
                raise DemoAccessError(
                    "demo_reset_failed",
                    "Reinitialisation de l'environnement demo impossible.",
                    status_code=500,
                ) from exc
    else:
        # ALLOW_NON_DEMO_SEED: garantir le socle démo (companies, drivers, etc.)
        # sans reset destructif. Sans cela, _seed_institution_demo_workspace ne trouve
        # aucune entreprise démo et lève RuntimeError.
        try:
            ensure_demo_reference_dataset(profile_name="sales")
        except Exception as exc:
            logger.exception(
                "[demo_access] ensure_demo_reference_dataset failed (ALLOW_NON_DEMO_SEED)",
                extra={
                    "event_type": "demo_reference_dataset_failed",
                    "demo_access_id": access.id,
                    "demo_request_id": access.demo_request_id,
                },
            )
            raise DemoAccessError(
                "demo_reference_dataset_failed",
                "Impossible d'initialiser les données de démonstration. Veuillez réessayer ou contacter l'équipe.",
                status_code=500,
            ) from exc

    demo_request = db.session.get(DemoRequest, access.demo_request_id) or access.demo_request
    if not demo_request:
        raise DemoAccessError(
            "request_not_found",
            "La demande de demo est introuvable.",
            status_code=404,
        )

    profile = _normalize_provision_profile(demo_request, None)
    org_type = str(profile.get("organization_type") or "").strip().lower()
    journey = _resolve_demo_journey(org_type)

    if journey == "generic":
        logger.error(
            "[demo_access] organization_type invalide ou vide - impossible d'initialiser le workspace",
            extra={
                "event_type": "demo_organization_type_invalid",
                "demo_request_id": demo_request.id,
                "demo_access_id": access.id,
                "organization_type": org_type or "(vide)",
            },
        )
        raise DemoAccessError(
            "invalid_organization_type",
            "Le type d'organisation de la demande est invalide ou manquant. Veuillez contacter l'équipe pour un nouveau lien.",
            status_code=400,
        )

    logger.info(
        "[demo_access] session start - initialisation du workspace",
        extra={
            "event_type": "demo_session_start",
            "demo_request_id": demo_request.id,
            "demo_access_id": access.id,
            "organization_type": org_type,
            "journey": journey,
        },
    )

    demo_user = _create_or_reuse_demo_user(demo_request, profile)
    _apply_demo_profile(demo_request, demo_user, profile)
    _ensure_demo_workspace_seeded(demo_request, demo_user, profile)
    demo_user.account_status = "active"
    demo_user.force_password_change = True
    db.session.flush()

    access.demo_user_id = demo_user.id
    access.demo_company_id = getattr(getattr(demo_user, "company", None), "id", None)
    db.session.add(access)
    db.session.commit()


def _ensure_demo_user_alignment_for_access(access: DemoAccess) -> User:
    """Garantit qu'un accès démo pointe vers un compte démo cohérent."""
    demo_request = db.session.get(DemoRequest, access.demo_request_id) or access.demo_request
    if not demo_request:
        raise DemoAccessError(
            "request_not_found",
            "La demande de demo est introuvable.",
            status_code=404,
        )

    profile = _normalize_provision_profile(demo_request, None)
    demo_user = access.demo_user or _create_or_reuse_demo_user(demo_request, profile)

    # Évite les comptes non-démo hérités d'anciens flux.
    if not str(demo_user.email or "").strip().lower().startswith("demo-"):
        demo_user.email = _build_demo_user_email(demo_request.email)

    _apply_demo_profile(demo_request, demo_user, profile)
    demo_user.account_status = "active"
    access.demo_user_id = demo_user.id
    access.demo_company_id = getattr(getattr(demo_user, "company", None), "id", None)
    db.session.add(access)
    db.session.flush()
    return demo_user


def _build_provision_summary(
    demo_user: User, demo_request: DemoRequest, provision_profile: dict[str, Any]
) -> dict[str, Any]:
    role_value = (
        demo_user.role.value
        if hasattr(demo_user.role, "value")
        else str(demo_user.role or "")
    )
    return {
        "workspace_display_name": provision_profile.get("workspace_display_name")
        or provision_profile.get("organization_name"),
        "organization_name": provision_profile.get("organization_name"),
        "organization_type": provision_profile.get("organization_type"),
        "demo_login_email": demo_user.email,
        "organization_contact_email": provision_profile.get("organization_contact_email"),
        "organization_contact_phone": provision_profile.get("organization_contact_phone"),
        "user_full_name": f"{demo_user.first_name or ''} {demo_user.last_name or ''}".strip(),
        "user_role": provision_profile.get("user_role") or role_value,
        "provision_template": provision_profile.get("provision_template"),
        "demo_persona": provision_profile.get("demo_persona"),
        "guide_variant": provision_profile.get("guide_variant"),
        "seed_context": provision_profile.get("seed_context") or {},
        "demo_request_id": demo_request.id,
    }


def provision_demo_access(
    *,
    demo_request_id: int,
    actor_id: int | None = None,
    provision_source: str = "manual",
    provisioning_mode: str = "shared_workspace",
    provision_profile: dict[str, Any] | None = None,
) -> DemoProvisionResult:
    try:
        ensure_demo_reference_dataset(profile_name="sales")
    except Exception:
        logger.exception("[demo_access] shared demo dataset ensure failed")

    demo_request = _assert_request_exists(demo_request_id)
    profile = _normalize_provision_profile(demo_request, provision_profile)
    org_type = str(profile.get("organization_type") or "").strip().lower()
    journey = _resolve_demo_journey(org_type)

    if journey == "generic":
        logger.error(
            "[demo_access] provision refusé: organization_type invalide ou vide",
            extra={
                "event_type": "demo_provision_organization_type_invalid",
                "demo_request_id": demo_request_id,
                "organization_type": org_type or "(vide)",
            },
        )
        raise DemoAccessError(
            "invalid_organization_type",
            "Le type d'organisation de la demande est invalide ou manquant. Veuillez corriger la demande avant de provisionner.",
            status_code=400,
        )

    active_access = _get_active_access_for_request(demo_request_id)
    if active_access and (not active_access.demo_expires_at or active_access.demo_expires_at > _utc_now()):
        logger.info(
            "[demo_access] provision reused existing access",
            extra={
                "event_type": "demo_access_reused",
                "demo_request_id": demo_request_id,
                "demo_access_id": active_access.id,
                "actor_type": "admin",
                "actor_id": actor_id,
            },
        )
        demo_user = active_access.demo_user or _create_or_reuse_demo_user(
            demo_request, profile
        )
        summary = _build_provision_summary(demo_user, demo_request, profile)
        return DemoProvisionResult(
            demo_request=demo_request,
            demo_access=active_access,
            magic_token="",
            email_sent=False,
            email_error=None,
            reused_existing_access=True,
            provision_summary=summary,
        )

    if not demo_request.email:
        raise DemoAccessError(
            "validation_error",
            "Adresse email de la demande manquante.",
            status_code=400,
        )

    # Phase 1: normalisation du profil de provisioning (défaut + overrides admin).
    demo_user = _create_or_reuse_demo_user(demo_request, profile)
    # Phase 2: création/mise à jour des entités workspace.
    _apply_demo_profile(demo_request, demo_user, profile)
    # Phase 3: seed contextualisé + persona/guide.
    _ensure_demo_workspace_seeded(demo_request, demo_user, profile)
    # Le dataset partagé est déjà garanti par ensure_demo_reference_dataset().
    demo_user.account_status = "active"
    demo_user.force_password_change = True
    now = _utc_now()
    plain_token, token_hash, token_expires_at = _new_magic_token()
    access = DemoAccess()
    access.demo_request_id = demo_request.id
    access.status = "active"
    access.magic_token_hash = token_hash
    access.magic_token_expires_at = token_expires_at
    access.demo_expires_at = now + timedelta(hours=DEMO_ACCESS_DURATION_HOURS)
    access.provisioned_at = now
    access.demo_user_id = demo_user.id
    access.demo_company_id = getattr(getattr(demo_user, "company", None), "id", None)
    access.provision_source = provision_source
    access.provisioning_mode = provisioning_mode
    db.session.add(access)
    db.session.flush()

    email_result = send_demo_access_ready_email(
        demo_request=demo_request,
        demo_access=access,
        magic_token=plain_token,
    )
    email_sent = bool(email_result.get("ok"))
    if email_sent:
        access.access_sent_at = now
        access.last_access_email_error = None
    else:
        access.last_access_email_error = str(email_result.get("error") or "email_error")[:1000]

    # Validation metier: une demande provisionnee devient qualifiee.
    demo_request.status = "qualified"

    db.session.commit()
    logger.info(
        "[demo_access] provisioned",
        extra={
            "event_type": "demo_access_provisioned",
            "demo_request_id": demo_request.id,
            "demo_access_id": access.id,
            "actor_type": "admin",
            "actor_id": actor_id,
            "provision_template": profile.get("provision_template"),
            "demo_persona": profile.get("demo_persona"),
            "guide_variant": profile.get("guide_variant"),
            "seed_context": profile.get("seed_context") or {},
            "provisioning_mode": provisioning_mode,
            "trace_id": _get_trace_id(),
        },
    )
    summary = _build_provision_summary(demo_user, demo_request, profile)
    return DemoProvisionResult(
        demo_request=demo_request,
        demo_access=access,
        magic_token=plain_token,
        email_sent=email_sent,
        email_error=access.last_access_email_error,
        reused_existing_access=False,
        provision_summary=summary,
    )


def resend_demo_access(*, access_id: int, actor_id: int | None = None) -> DemoProvisionResult:
    access = _assert_access_exists(access_id)
    _ensure_access_is_active(access)
    plain_token, token_hash, token_expires_at = _new_magic_token()
    access.magic_token_hash = token_hash
    access.magic_token_expires_at = token_expires_at
    access.magic_token_used_at = None

    email_result = send_demo_access_ready_email(
        demo_request=access.demo_request,
        demo_access=access,
        magic_token=plain_token,
    )
    email_sent = bool(email_result.get("ok"))
    if email_sent:
        access.access_sent_at = _utc_now()
        access.last_access_email_error = None
    else:
        access.last_access_email_error = str(email_result.get("error") or "email_error")[:1000]

    db.session.commit()
    logger.info(
        "[demo_access] resent",
        extra={
            "event_type": "demo_access_resent",
            "demo_request_id": access.demo_request_id,
            "demo_access_id": access.id,
            "actor_type": "admin",
            "actor_id": actor_id,
        },
    )
    return DemoProvisionResult(
        demo_request=access.demo_request,
        demo_access=access,
        magic_token=plain_token,
        email_sent=email_sent,
        email_error=access.last_access_email_error,
    )


def revoke_demo_access(*, access_id: int, actor_id: int | None = None) -> DemoAccess:
    access = _assert_access_exists(access_id)
    if access.status in {"expired", "revoked"}:
        return access

    access.status = "revoked"
    access.revoked_at = _utc_now()
    access.magic_token_hash = None
    access.magic_token_expires_at = None
    _disable_demo_user(access.demo_user_id)
    db.session.commit()
    logger.info(
        "[demo_access] revoked",
        extra={
            "event_type": "demo_access_revoked",
            "demo_request_id": access.demo_request_id,
            "demo_access_id": access.id,
            "actor_type": "admin",
            "actor_id": actor_id,
        },
    )
    return access


MAGIC_TOKEN_MAX_LENGTH = 256  # Limite anti-DoS (token_urlsafe(32) ~43 chars)


def consume_magic_link(token: str) -> dict[str, Any]:
    # Invariant d'architecture:
    # - le magic link represente un demo_access (source de verite demo)
    # - sa consommation ne doit jamais creer de session "app"
    # - la redirection finale doit rester en namespace /demo/*
    token = (token or "").strip()
    if not token:
        raise DemoAccessError("invalid_token", "Token invalide.", status_code=400)
    if len(token) > MAGIC_TOKEN_MAX_LENGTH:
        raise DemoAccessError("invalid_token", "Token invalide.", status_code=400)

    token_hash = _hash_magic_token(token)
    access = (
        DemoAccess.query.filter_by(magic_token_hash=token_hash)
        .order_by(DemoAccess.created_at.desc())
        .first()
    )
    if not access:
        raise DemoAccessError("invalid_token", "Token invalide.", status_code=404)

    _ensure_access_is_active(access)
    # Le reset/seed complet est déclenché au premier démarrage de session démo
    # (première consommation du magic link), puis l'appel reste idempotent.
    now = _utc_now()
    already_consumed = access.magic_token_used_at is not None
    if (
        not already_consumed
        and (not access.magic_token_expires_at or access.magic_token_expires_at <= now)
    ):
        raise DemoAccessError("token_expired", "Token expire.", status_code=409)

    should_reset_session = (
        (not already_consumed)
        or (
            access.magic_token_used_at is not None
            and (now - access.magic_token_used_at)
            > timedelta(seconds=SESSION_RESET_DEBOUNCE_SECONDS)
        )
    )

    if should_reset_session:
        # Debut de session demo: reset complet pour repartir d'un environnement propre.
        # Debounce court pour eviter un double reset sur doubles requetes front immediates.
        _reset_demo_dataset_on_session_start(access)
        access = db.session.get(DemoAccess, access.id) or access
        _ensure_access_is_active(access)
        now = _utc_now()
        access.magic_token_used_at = now
        # Garder le hash/token TTL pour rendre l'appel idempotent en cas de rechargement.
        db.session.commit()
        logger.info(
            "[demo_access] link consumed",
            extra={
                "event_type": "demo_magic_link_consumed",
                "demo_request_id": access.demo_request_id,
                "demo_access_id": access.id,
                "actor_type": "public_user",
                "actor_id": None,
                "trace_id": _get_trace_id(),
            },
        )

    demo_user = _ensure_demo_user_alignment_for_access(access)
    db.session.commit()

    return {
        "ok": True,
        "session_created": True,
        "already_consumed": already_consumed,
        "redirect_to": "/demo/home",
        "demo_access_id": access.id,
        "demo_request_id": access.demo_request_id,
        "demo_user_id": demo_user.id,
    }


def expire_due_demo_accesses() -> int:
    now = _utc_now()
    to_expire = DemoAccess.query.filter(
        DemoAccess.status == "active",
        DemoAccess.demo_expires_at.isnot(None),
        DemoAccess.demo_expires_at <= now,
    ).all()
    for access in to_expire:
        _apply_expiration(access, source="system")
    if to_expire:
        db.session.commit()
    return len(to_expire)
