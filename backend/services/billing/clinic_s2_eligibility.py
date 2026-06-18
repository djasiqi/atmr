"""Filtres d'éligibilité facturation clinique S2 (transports portail institution inclus)."""

from __future__ import annotations

from sqlalchemy import and_, exists, func, or_
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import ColumnElement

from models import Booking, Client, Company
from models.enums import BookingCreatedVia


def institution_portal_clinic_booking_matches(
    clinic_company_id: int,
    transport_company_id: int,
) -> ColumnElement[bool]:
    """Bookings institution `billed_to_type=clinic` rattachés à cette clinique (même si `billed_to_company_id` erroné)."""
    ccid = int(clinic_company_id)
    carrier_id = int(transport_company_id)
    InstClient = aliased(Client)
    name_match = exists().where(
        Company.id == ccid,
        InstClient.id == Booking.client_id,
        InstClient.is_institution.is_(True),
        func.lower(Company.name) == func.lower(InstClient.institution_name),
    )
    return and_(
        Booking.created_via == BookingCreatedVia.INSTITUTION_PORTAL,
        Booking.billed_to_type == "clinic",
        Booking.company_id == carrier_id,
        exists().where(
            InstClient.id == Booking.client_id,
            InstClient.is_institution.is_(True),
            or_(
                InstClient.default_billed_to_company_id == ccid,
                name_match,
            ),
        ),
    )


def clinic_s2_billed_to_company_predicate(
    clinic_company_id: int,
    transport_company_id: int,
) -> ColumnElement[bool]:
    """Transport facturé à la clinique : champ direct ou portail institution corrigé."""
    ccid = int(clinic_company_id)
    return or_(
        Booking.billed_to_company_id == ccid,
        institution_portal_clinic_booking_matches(ccid, transport_company_id),
    )
