"""Règles d'identité patient institution — PATIENT-IDENTITY-01.

Civilité + DOB obligatoires. Date future interdite.
Mineur autorisé avec confirmation explicite (``minor_dob_confirmed``).
"""

from __future__ import annotations

from datetime import date

from marshmallow import ValidationError

MIN_ADULT_AGE_YEARS = 18
MINOR_DOB_CONFIRMATION_CODE = "MINOR_DOB_CONFIRMATION_REQUIRED"
MINOR_DOB_CONFIRMATION_MESSAGE = (
    "Patient mineur : confirmation de la date de naissance requise "
    "(minor_dob_confirmed=true)."
)


def parse_calendar_date(value: str) -> date:
    """Parse YYYY-MM-DD en date calendrier réelle (rejette 2026-02-31)."""
    raw = (value or "").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError("Date de naissance invalide.") from exc


def adult_dob_cutoff(today: date | None = None) -> date:
    """Date de naissance d'un patient ayant exactement 18 ans aujourd'hui."""
    ref = today or date.today()
    try:
        return ref.replace(year=ref.year - MIN_ADULT_AGE_YEARS)
    except ValueError:
        return ref.replace(year=ref.year - MIN_ADULT_AGE_YEARS, day=28)


def patient_age_years(dob: date, *, today: date | None = None) -> int:
    """Âge civil en années révolues."""
    ref = today or date.today()
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return years


def is_minor(dob: date, *, today: date | None = None) -> bool:
    """True si âge < 18 ans à la date de référence."""
    return patient_age_years(dob, today=today) < MIN_ADULT_AGE_YEARS


def validate_patient_dob(value: str, *, today: date | None = None) -> date:
    """Valide une DOB : calendrier réel, pas dans le futur.

    Ne refuse PAS les mineurs — la confirmation est gérée à part.
    """
    dob = parse_calendar_date(value)
    ref = today or date.today()
    if dob > ref:
        raise ValidationError("La date de naissance ne peut pas être dans le futur.")
    return dob


def requires_minor_dob_confirmation(
    *,
    new_dob: date | None,
    previous_dob: date | None = None,
    today: date | None = None,
) -> bool:
    """Indique si ``minor_dob_confirmed`` est obligatoire.

    - CREATE (previous_dob=None) : oui si nouvelle DOB mineure
    - UPDATE : oui seulement si DOB change vers une DOB mineure
      (ou devient mineure). DOB mineure inchangée → pas de reconfirmation.
    """
    if new_dob is None:
        return False
    if not is_minor(new_dob, today=today):
        return False
    return not (previous_dob is not None and previous_dob == new_dob)


DOMICILE_FIELDS = ("address", "postal_code", "city")


def _blank_to_none(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_domicile_triplet(
    *,
    address: object | None,
    postal_code: object | None,
    city: object | None,
) -> dict[str, str]:
    """Valide le triplet domicile complet. Messages FR métier."""
    cleaned_address = _blank_to_none(address)
    cleaned_postal = _blank_to_none(postal_code)
    cleaned_city = _blank_to_none(city)
    errors: dict[str, list[str]] = {}
    if not cleaned_address:
        errors["address"] = ["Adresse requise"]
    if not cleaned_postal:
        errors["postal_code"] = ["NPA requis"]
    if not cleaned_city:
        errors["city"] = ["Ville requise"]
    if errors:
        raise ValidationError(errors)
    return {
        "address": cleaned_address,
        "postal_code": cleaned_postal,
        "city": cleaned_city,
    }


def domicile_fields_touched(payload: dict) -> bool:
    """True si le payload touche au moins un champ du domicile."""
    return any(key in payload for key in DOMICILE_FIELDS)
