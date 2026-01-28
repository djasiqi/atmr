from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ext import db
from models import (
    BillingParty,
    BillingPartyType,
    Client,
    ClientType,
    ClinicBillingPartyMapping,
    Company,
    User,
    UserRole,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SqlAlchemyClientWriter:
    """Adaptateur Infrastructure: création d'un client (User + Client)
    via SQLAlchemy."""

    def create_client_for_company(
        self,
        *,
        company_id: int,
        user_attrs: dict[str, Any],
        client_attrs: dict[str, Any],
    ) -> tuple[User, Client]:
        user = User()
        user.public_id = user_attrs.get("public_id")
        user.username = user_attrs.get("username")
        user.first_name = user_attrs.get("first_name") or ""
        user.last_name = user_attrs.get("last_name") or ""
        user.email = user_attrs.get("email")
        user.phone = user_attrs.get("phone")
        user.address = user_attrs.get("address")
        user.birth_date = user_attrs.get("birth_date")
        user.role = UserRole.client

        password = user_attrs.get("password")
        if password:
            user.set_password(password)  # nosem

        db.session.add(user)
        db.session.flush()

        ct = client_attrs.get("client_type") or "PRIVATE"
        ct_upper = str(ct).upper()
        client_type = (
            ClientType[ct_upper]
            if ct_upper in ClientType.__members__
            else ClientType.PRIVATE
        )

        preferential_rate = client_attrs.get("preferential_rate")
        if preferential_rate not in (None, ""):
            try:
                preferential_rate = Decimal(str(preferential_rate))
            except Exception:
                preferential_rate = None
        else:
            preferential_rate = None

        client = Client()
        client.user_id = user.id
        client.company_id = company_id
        client.client_type = client_type
        client.billing_address = client_attrs.get("billing_address")
        client.billing_lat = client_attrs.get("billing_lat")
        client.billing_lon = client_attrs.get("billing_lon")
        client.contact_email = client_attrs.get("contact_email")
        client.contact_phone = client_attrs.get("contact_phone")
        client.is_institution = bool(client_attrs.get("is_institution", False))
        client.institution_name = client_attrs.get("institution_name")
        client.domicile_address = client_attrs.get("domicile_address")
        client.domicile_zip = client_attrs.get("domicile_zip")
        client.domicile_city = client_attrs.get("domicile_city")
        client.domicile_lat = client_attrs.get("domicile_lat")
        client.domicile_lon = client_attrs.get("domicile_lon")
        client.preferential_rate = preferential_rate
        client.residence_facility = client_attrs.get("residence_facility")
        client.door_code = client_attrs.get("door_code")
        client.floor = client_attrs.get("floor")
        client.access_notes = client_attrs.get("access_notes")
        client.gp_name = client_attrs.get("gp_name")
        client.gp_phone = client_attrs.get("gp_phone")
        if client_attrs.get("default_billed_to_type"):
            client.default_billed_to_type = client_attrs.get("default_billed_to_type")
        client.default_billed_to_contact = client_attrs.get("default_billed_to_contact")
        if "is_active" in client_attrs:
            client.is_active = bool(client_attrs["is_active"])

        db.session.add(client)
        db.session.flush()  # Pour obtenir client.id

        # ✅ Si c'est une institution, créer automatiquement une Company et un BillingParty
        is_institution = bool(client_attrs.get("is_institution", False))
        institution_name = client_attrs.get("institution_name")
        if is_institution and institution_name:
            try:
                # Créer la Company pour la clinique (attributs assignés pour compatibilité type-checker)
                clinic_company = Company()
                clinic_company.name = client.institution_name
                clinic_company.user_id = user.id
                clinic_company.address = client.domicile_address or ""
                clinic_company.latitude = (
                    float(client.domicile_lat)
                    if getattr(client, "domicile_lat", None) is not None
                    else None
                )
                clinic_company.longitude = (
                    float(client.domicile_lon)
                    if getattr(client, "domicile_lon", None) is not None
                    else None
                )
                clinic_company.contact_email = client.contact_email or user.email or ""
                clinic_company.contact_phone = client.contact_phone or user.phone or ""
                clinic_company.service_area = ""
                clinic_company.max_daily_bookings = 50
                clinic_company.is_approved = False
                clinic_company.preferential_rate = preferential_rate
                db.session.add(clinic_company)
                db.session.flush()  # Pour obtenir clinic_company.id

                # Associer la Company au client
                client.default_billed_to_company_id = clinic_company.id

                # Créer le BillingParty pour la clinique
                billing_address = client.billing_address or client.domicile_address or ""
                if client.domicile_zip and client.domicile_city:
                    if billing_address:
                        billing_address = (
                            f"{billing_address}\n{client.domicile_zip} {client.domicile_city}"
                        )
                    else:
                        billing_address = f"{client.domicile_zip} {client.domicile_city}"

                billing_party = BillingParty()
                billing_party.company_id = company_id  # L'entreprise de transport
                billing_party.type = BillingPartyType.CLINIC
                billing_party.display_name = client.institution_name
                billing_party.billing_address = billing_address or "Adresse non renseignée"
                billing_party.contact_email = client.contact_email or user.email
                billing_party.contact_phone = client.contact_phone or user.phone
                billing_party.external_ref = f"clinic_company:{clinic_company.id}"
                billing_party.is_active = True
                db.session.add(billing_party)
                db.session.flush()  # Pour obtenir billing_party.id

                # Créer le mapping clinique → billing party
                mapping = ClinicBillingPartyMapping()
                mapping.company_id = company_id  # L'entreprise de transport
                mapping.clinic_company_id = clinic_company.id  # La clinique (payeur)
                mapping.billing_party_id = billing_party.id  # Le destinataire de facturation
                mapping.is_active = True
                db.session.add(mapping)

                logger.info(
                    (
                        "✅ Institution créée avec Company et BillingParty automatiques: "
                        "client_id=%s, clinic_company_id=%s, billing_party_id=%s, mapping_id=%s"
                    ),
                    client.id,
                    clinic_company.id,
                    billing_party.id,
                    mapping.id,
                )
            except Exception as e:
                logger.exception(
                    (
                        "❌ Erreur lors de la création automatique de Company/BillingParty "
                        "pour institution client_id=%s: %s"
                    ),
                    client.id,
                    str(e),
                )
                # Ne pas faire échouer la création du client si la Company/BillingParty échoue
                # L'utilisateur pourra les créer manuellement plus tard

        return user, client
