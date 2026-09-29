"""Contrat portail particulier → réservation exploitable par le transporteur.

Règles verrouillées :

- « Dès que possible » ne fabrique aucune heure.
- « Je dois arriver à » enregistre le rendez-vous, pas une prise en charge inventée.
- « Je souhaite partir à » enregistre l'heure de prise en charge.
- Un aller-retour sans heure reste ``scheduled_time is None`` (jamais 00:00).
- Fauteuil personnel et fauteuil à fournir s'excluent. L'assistance est indépendante.
- L'accès départ / destination reste dans ses colonnes, pas dans une note unique.
- Service ou médecin : champ libre côté client, classification seulement si elle est sûre.
"""

from __future__ import annotations

import re
from typing import Any

_DOCTOR_RE = re.compile(r"\b(dr\.?|docteur|médecin|medecin)\b", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\s+[–—-]\s+")


def _as_bool(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def decide_portal_schedule(validated: dict[str, Any]) -> dict[str, Any]:
    """Décide l'horaire aller sans inventer de prise en charge.

    ``time_confirmed`` vaut False pour « dès que possible » et pour un rendez-vous :
    l'heure stockée n'est alors pas une prise en charge confirmée.
    """
    asap = _as_bool(validated.get("asap")) or _as_bool(validated.get("is_urgent"))
    if asap:
        return {
            "kind": "asap",
            "scheduled_time_raw": None,
            "time_confirmed": False,
            "is_urgent": True,
            "scheduled_time_type": "departure",
        }
    kind = (validated.get("scheduled_time_type") or "departure").strip() or "departure"
    if kind not in {"arrival", "departure"}:
        kind = "departure"
    return {
        "kind": kind,
        "scheduled_time_raw": _blank(validated.get("scheduled_time")),
        "time_confirmed": kind == "departure",
        "is_urgent": False,
        "scheduled_time_type": kind,
    }


def decide_return_schedule(validated: dict[str, Any]) -> dict[str, Any]:
    """Aller-retour : heure vide → None, jamais minuit."""
    if not _as_bool(validated.get("is_round_trip")):
        return {
            "return_time_raw": None,
            "return_time_exact": False,
            "return_date": None,
        }
    return_time = _blank(validated.get("return_time"))
    return_date = _blank(validated.get("return_date"))
    return {
        "return_time_raw": return_time,
        "return_time_exact": return_time is not None,
        "return_date": return_date,
    }


def classify_destination_contact(raw: Any) -> dict[str, str]:
    """Un seul texte client. Classification seulement si un médecin est reconnaissable.

    Le texte brut reste dans ``destination_contact_detail``. Il n'est pas relégué
    à une note libre. Sans indice médecin, il est rangé dans ``hospital_service``.
    """
    text = _blank(raw) or ""
    if not text:
        return {
            "destination_contact_detail": "",
            "hospital_service": "",
            "doctor_name": "",
        }
    parts = [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
    if len(parts) >= 2 and _DOCTOR_RE.search(parts[-1]):
        return {
            "destination_contact_detail": text,
            "hospital_service": parts[0][:255],
            "doctor_name": " – ".join(parts[1:])[:200],
        }
    if len(parts) == 1 and _DOCTOR_RE.search(text):
        return {
            "destination_contact_detail": text,
            "hospital_service": "",
            "doctor_name": text[:200],
        }
    return {
        "destination_contact_detail": text,
        "hospital_service": text[:255],
        "doctor_name": "",
    }


def validate_medical_destination(data: dict[str, Any]) -> None:
    """Établissement obligatoire, et le texte saisi par le client suffit.

    Le particulier n'a pas à savoir si sa saisie est un service ou un médecin.
    Un texte rempli est accepté même s'il ne se classe pas.
    """
    if not _as_bool(data.get("medical_destination")):
        return
    facility = _blank(data.get("medical_facility"))
    detail = _blank(data.get("medical_destination_detail"))
    service = _blank(data.get("hospital_service"))
    doctor = _blank(data.get("doctor_name"))
    if not facility:
        raise ValueError("L'établissement est obligatoire pour une destination médicale.")
    if not (detail or service or doctor):
        raise ValueError(
            "Indiquez le service ou le médecin pour une destination médicale."
        )


def prepare_portal_extra_stop(step: dict[str, Any]) -> dict[str, Any]:
    """Étape supplémentaire : même contrat horaire que la course, sans deuxième système."""
    address = _blank(step.get("address") or step.get("dropoff_location"))
    if not address:
        raise ValueError("Chaque étape doit avoir une adresse.")
    validate_medical_destination(step)
    explicit_service = _blank(step.get("hospital_service")) or ""
    explicit_doctor = _blank(step.get("doctor_name")) or ""
    detail = _blank(step.get("medical_destination_detail")) or ""
    if explicit_service or explicit_doctor:
        classified = {
            "hospital_service": explicit_service[:255],
            "doctor_name": explicit_doctor[:200],
            "destination_contact_detail": detail
            or " – ".join(part for part in (explicit_service, explicit_doctor) if part),
        }
    else:
        classified = classify_destination_contact(detail)
    schedule = decide_portal_schedule(step)
    if schedule["kind"] != "asap" and not schedule["scheduled_time_raw"]:
        raise ValueError(
            "Chaque étape planifiée doit avoir une date et une heure, "
            "ou être marquée dès que possible."
        )
    return {
        "dropoff_location": address,
        "schedule": schedule,
        "medical_facility": (_blank(step.get("medical_facility")) or "")[:200],
        "hospital_service": classified["hospital_service"],
        "doctor_name": classified["doctor_name"],
        "medical_destination_detail": classified["destination_contact_detail"],
        "dropoff_access_notes": _blank(step.get("access_notes")),
    }


def read_company_portal_schedule(reservation: dict[str, Any]) -> dict[str, Any]:
    """Lecture entreprise du contrat PORTAL, sans reconstruire depuis une note.

    ``notes_medical`` est ignoré. Seuls ``is_urgent``, ``time_confirmed``,
    ``scheduled_time`` et ``is_return`` décident du libellé.
    """
    _ = reservation.get("notes_medical")
    is_urgent = _as_bool(reservation.get("is_urgent"))
    is_return = _as_bool(reservation.get("is_return"))
    scheduled = reservation.get("scheduled_time")
    if isinstance(scheduled, str) and not scheduled.strip():
        scheduled = None
    time_confirmed = reservation.get("time_confirmed")
    if is_urgent:
        return {
            "kind": "asap",
            "label": "Dès que possible",
            "scheduled_time": None,
            "time_confirmed": False,
            "pickup_time_fabricated": False,
        }
    if (not is_return) and time_confirmed is False and scheduled is not None:
        return {
            "kind": "appointment",
            "label": "Rendez-vous — prise en charge à proposer",
            "scheduled_time": scheduled,
            "time_confirmed": False,
            "pickup_time_fabricated": False,
        }
    return {
        "kind": "departure",
        "label": "Horaire",
        "scheduled_time": scheduled,
        "time_confirmed": time_confirmed is not False,
        "pickup_time_fabricated": False,
    }


def scrub_access_from_free_note(validated: dict[str, Any]) -> str:
    """Retire l'accès départ/destination d'une note libre éventuelle."""
    note = str(validated.get("client_note") or "")
    for key in ("pickup_access_notes", "dropoff_access_notes"):
        chunk = _blank(validated.get(key))
        if chunk:
            note = note.replace(chunk, "")
    cleaned = "\n".join(
        line.strip()
        for line in note.splitlines()
        if line.strip() and line.strip() not in {"Prise en charge :", "Destination :"}
    )
    return cleaned.strip()
