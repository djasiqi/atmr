"""Résolution du bénéficiaire d'une facture pour le bloc « Facturé à ».

Partagé entre le rendu PDF ReportLab et le constructeur de template HTML, afin que
les deux chemins produisent le même destinataire.

Invariant : le patient d'une facture se déduit uniquement de CETTE facture
(``Invoice.institution_patient_id`` puis ses bookings). Le déduire à l'échelle de
l'institution donnerait la même adresse à tous les résidents.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)


def iter_invoice_bookings(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
):
    """Bookings rattachés aux lignes de la facture.

    ``bookings_by_id`` est fourni par le pipeline PDF pour éviter les N+1 ; sans lui
    on retombe sur une requête ponctuelle par ligne.
    """
    from models import Booking

    for line in getattr(invoice, "lines", None) or []:
        # billed_booking est un backref InstrumentedList, pas un objet unique
        related = getattr(line, "billed_booking", None)
        if isinstance(related, list):
            if related:
                yield related[0]
            continue
        if related is not None:
            yield related
            continue
        reservation_id = getattr(line, "reservation_id", None)
        if not reservation_id:
            continue
        booking = (
            bookings_by_id.get(reservation_id)
            if bookings_by_id is not None
            else Booking.query.get(reservation_id)
        )
        if booking is not None:
            yield booking


def resolve_invoice_institution_patient(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
):
    """Patient institutionnel bénéficiaire de la facture, ou ``None``."""
    from models.institution_patient import InstitutionPatient

    patient_id = getattr(invoice, "institution_patient_id", None)
    if patient_id is None:
        for booking in iter_invoice_bookings(invoice, bookings_by_id=bookings_by_id):
            candidate = getattr(booking, "institution_patient_id", None)
            if candidate is None:
                resolve = getattr(booking, "_resolve_source_transport_request", None)
                request = resolve() if callable(resolve) else None
                candidate = getattr(request, "patient_id", None) if request else None
            if candidate is not None:
                patient_id = candidate
                break
    if patient_id is None:
        return None
    return InstitutionPatient.query.get(int(patient_id))


def institution_patient_billing_address(patient: Any) -> str:
    """Adresse de domicile du patient, qui fait office d'adresse de facturation."""
    if patient is None:
        return ""
    street = (getattr(patient, "address", None) or "").strip()
    postal = (getattr(patient, "postal_code", None) or "").strip()
    city = (getattr(patient, "city", None) or "").strip()
    postal_city = " ".join(part for part in (postal, city) if part)
    return ", ".join(part for part in (street, postal_city) if part)


def invoice_residence_label(
    *,
    patient: Any | None = None,
    client: Any | None = None,
) -> str:
    """Établissement de résidence (EMS, foyer) à coller sous le nom « Facturé à »."""
    if patient is not None:
        name = (getattr(patient, "residence_name", None) or "").strip()
        if name:
            return name
    if client is not None:
        return (getattr(client, "residence_facility", None) or "").strip()
    return ""


def append_residence_to_billed_to_name(
    name: str,
    residence: str,
    *,
    separator: str,
) -> str:
    """Ajoute la résidence sous le nom, sans doublon."""
    base = (name or "").strip()
    label = (residence or "").strip()
    if not label:
        return base
    if label.casefold() in base.casefold():
        return base
    return f"{base}{separator}{label}"


def live_patient_payer_identity(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Nom et adresse courants du payeur PATIENT (relecture live, pas snapshot BP).

    Un BillingParty PATIENT est un pointeur vers le bénéficiaire : si le client
    ou le patient institutionnel a été modifié après création de la facture, le
    PDF doit reprendre ces valeurs actuelles — jamais un ``display_name`` /
    ``billing_address`` périmé.
    """
    patient = resolve_invoice_institution_patient(
        invoice, bookings_by_id=bookings_by_id
    )
    if patient is not None:
        first = (getattr(patient, "first_name", None) or "").strip()
        last = (getattr(patient, "last_name", None) or "").strip()
        name = f"{first} {last}".strip() or None
        addr = institution_patient_billing_address(patient) or None
        return name, addr or None

    client = getattr(invoice, "client", None)
    if client is None:
        return None, None

    user = getattr(client, "user", None)
    name = None
    if user is not None:
        first = (getattr(user, "first_name", None) or "").strip()
        last = (getattr(user, "last_name", None) or "").strip()
        full = f"{first} {last}".strip()
        username = (getattr(user, "username", None) or "").strip()
        name = full or username or None
    if not name:
        first = (getattr(client, "first_name", None) or "").strip()
        last = (getattr(client, "last_name", None) or "").strip()
        name = f"{first} {last}".strip() or None
    if not name and bool(getattr(client, "is_institution", False)):
        name = (getattr(client, "institution_name", None) or "").strip() or None

    street = (getattr(client, "domicile_address", None) or "").strip()
    postal = (getattr(client, "domicile_zip", None) or "").strip()
    city = (getattr(client, "domicile_city", None) or "").strip()
    parts: list[str] = []
    if street:
        parts.append(street)
    postal_city = " ".join(part for part in (postal, city) if part)
    if postal_city:
        parts.append(postal_city)
    addr = "\n".join(parts) if parts else None
    if not addr:
        try:
            addr = (
                getattr(client, "billing_address_secure", None) or ""
            ).strip() or None
        except Exception:
            addr = (getattr(client, "billing_address", None) or "").strip() or None
    if not addr and user is not None:
        addr = (getattr(user, "address", None) or "").strip() or None
    return name, addr


def format_billing_party_recipient_name(
    bp_display_name: str,
    contact_name: str | None = None,
    *,
    separator: str = "\n",
) -> str:
    """Nom « Facturé à » : débiteur/organisme, puis contact facturation.

    Le type du tiers payeur (ex. ``curatorship``) ne qualifie jamais le contact
    comme représentant légal. On affiche uniquement :

    ``Hospice général``
    ``À l'att. de Mme Amandine HAUSER``
    """
    name = (bp_display_name or "Payeur").strip() or "Payeur"
    contact = (contact_name or "").strip()
    if not contact:
        return name
    if contact.casefold() == name.casefold():
        return name
    return f"{name}{separator}À l'att. de {contact}"


# ═══════════════════════════════════════════════════════════════════════════
# Source de vérité commune du bloc « Facturé à »
# ═══════════════════════════════════════════════════════════════════════════
#
# Trois notions distinctes, jamais confondues :
#   1. le patient / client concerné par la facture (``Invoice.client`` ou patient
#      institutionnel) ;
#   2. le payeur (``Invoice.billing_party``), identifiable séparément dans les
#      données comptables (registre, snapshots, QR-facture) ;
#   3. le destinataire administratif : à qui et à quelle adresse le courrier part.
#
# Règle : ``payer != patient`` ne remplace JAMAIS automatiquement le patient par le
# payeur. Le mode de correspondance est déduit du modèle de données existant :
#
#   patient_self         payeur = patient (ou aucun tiers) → patient (+ résidence),
#                        adresse du patient, aucune ligne « c/o ».
#   patient_care_of      la correspondance passe par un tiers (curatelle, OPAD,
#                        avocat, famille, autre) dont l'adresse de facturation est
#                        distincte de celle du patient → patient, « c/o tiers »,
#                        éventuellement « À l'att. de contact », adresse du tiers.
#   organization_debtor  établissement / organisme débiteur en nom propre
#                        (clinique, EMS, hôpital, assurance, facture S2 mensuelle)
#                        → organisme, éventuellement « À l'att. de contact »,
#                        adresse de l'organisme. Le patient figure dans les lignes.
#   legacy_institution   ``bill_to_client_id`` (client institution, ancien modèle).
#   unresolved_party     ``billing_party_id`` défini mais payeur introuvable.
#
# Si le tiers payeur est domicilié à l'adresse du patient (l'adresse de facturation
# appartient explicitement au patient), aucun « c/o » artificiel n'est ajouté.

BilledToMode = Literal[
    "patient_self",
    "patient_care_of",
    "organization_debtor",
    "legacy_institution",
    "unresolved_party",
]
BilledToSource = Literal["live", "snapshot", "legacy_live"]

# Tiers dont la correspondance s'adresse au patient « c/o » ce tiers.
# ``other`` : défaut PROVISOIRE (comportement historique < 16.09, cohérent avec SPC) ;
# un organisme débiteur en nom propre typé ``other`` doit le déclarer explicitement
# via ``recipient_mode`` (voir docs/facturation/bloc-facture-a-destinataire.md).
CARE_OF_PARTY_TYPES = frozenset({"curatorship", "opad", "lawyer", "family", "other"})
# Établissements / organismes facturés en nom propre (jamais de « c/o »).
ORGANIZATION_DEBTOR_PARTY_TYPES = frozenset({"clinic", "ems", "hospital", "insurance"})

# Attribut métier explicite (prioritaire sur toute inférence) lu sur le payeur
# (``BillingParty.recipient_mode``) puis sur le lien client↔payeur
# (``ClientBillingParty.recipient_mode``) : « care_of » = correspondance patient c/o
# tiers, « debtor » = organisme facturé en nom propre, « auto »/absent = inférence.
RECIPIENT_MODE_ATTR = "recipient_mode"
EXPLICIT_RECIPIENT_MODES: dict[str, BilledToMode] = {
    "care_of": "patient_care_of",
    "debtor": "organization_debtor",
}

# Snapshot immuable du bloc « Facturé à » (``Invoice.meta``).
BILLED_TO_SNAPSHOT_KEY = "billed_to_snapshot"
BILLED_TO_SNAPSHOT_VERSION = 1
# Seule une facture brouillon suit les master data ; dès qu'elle quitte DRAFT
# (envoyée, payée, en retard, annulée) le bloc est figé.
_DRAFT_STATUS_VALUE = "draft"

_COUNTRY_CODES: tuple[tuple[str, str, str], ...] = (
    (r"suisse|switzerland|schweiz|svizzera", "CH", "Suisse"),
    (r"france", "FR", "France"),
    (r"deutschland|germany|allemagne", "DE", "Allemagne"),
    (r"italy|italia|italie", "IT", "Italie"),
)

_CARE_OF_PREFIX_RE = re.compile(r"^\s*(?:c/o|c\.o\.|chez)\b\s*", re.IGNORECASE)
_COUNTRY_TOKENS_RE = re.compile(
    r"\b(?:suisse|switzerland|schweiz|svizzera|ch)\b", re.IGNORECASE
)
_HOUSE_NUMBER_RE = re.compile(
    r"^\d{1,3}\s*[a-zA-Z]?(?:\s*(?:bis|ter))?$", re.IGNORECASE
)
_STREET_COMMA_NUMBER_RE = re.compile(
    r"\s*,\s*(\d+\s*[a-zA-Z]?(?:\s*(?:bis|ter))?)$", re.IGNORECASE
)
_POSTAL_CODE_RE = re.compile(r"^\d{4,5}$")
_MIN_COMPARABLE_ADDRESS_LEN = 8


def join_street_and_number(street: str) -> str:
    """« Rue Patru, 2 » → « Rue Patru 2 » (format saisi « rue, numéro »)."""
    return _STREET_COMMA_NUMBER_RE.sub(r" \1", (street or "").strip())


def split_postal_address_lines(raw: str | None) -> list[str]:
    """Découpe une adresse « Rue, N, NPA, Ville » ou multi-lignes en lignes postales.

    Un numéro seul est recollé à la rue (« Rue Patru, 2 » → « Rue Patru 2 »), un NPA
    seul à la localité (« 1205, Genève » → « 1205 Genève ») ; les doublons
    consécutifs sont ignorés.
    """
    if not (raw or "").strip():
        return []
    tokens = [t.strip() for t in re.split(r"\r\n|\r|\n|,", str(raw)) if t.strip()]
    lines: list[str] = []
    pending_postal: str | None = None
    for token in tokens:
        if pending_postal is not None:
            lines.append(f"{pending_postal} {token}")
            pending_postal = None
            continue
        if _POSTAL_CODE_RE.match(token):
            pending_postal = token
            continue
        if _HOUSE_NUMBER_RE.match(token) and lines:
            lines[-1] = f"{lines[-1]} {token}"
            continue
        if lines and lines[-1].casefold() == token.casefold():
            continue
        lines.append(token)
    if pending_postal is not None:
        lines.append(pending_postal)
    return lines


@dataclass(frozen=True, slots=True)
class BilledToParty:
    """Résolution canonique (indépendante du rendu PDF / HTML) du bloc « Facturé à ».

    ``addressee`` : ligne 1 (patient ou organisme), brute (sans mise en majuscules).
    ``care_of`` : tiers de correspondance (sans le préfixe « c/o »).
    ``attention`` : contact facturation (sans le préfixe « À l'att. de »).
    ``residence`` : établissement de résidence du patient (EMS, foyer…).
    ``address`` : adresse postale brute de correspondance (texte, ``\\n`` ou virgules).
    ``address_owner`` : nom dont l'adresse peut déjà contenir le libellé (dédoublonnage).
    ``client_reference`` : référence du patient chez le payeur (ex. No SPC).
    """

    mode: BilledToMode
    addressee: str
    address: str = ""
    care_of: str | None = None
    attention: str | None = None
    residence: str | None = None
    address_owner: str | None = None
    client_reference: str | None = None
    client_reference_label: str | None = None
    billing_party: Any = None
    institution_patient: Any = None
    # Identité du patient de la facture (informatif, même en mode organisme).
    patient_name: str | None = None
    # Coordonnées du tiers payeur / organisme (rendu HTML uniquement).
    payer_contact_email: str | None = None
    payer_contact_phone: str | None = None
    payer_external_ref: str | None = None
    # ``live`` : master data courantes ; ``snapshot`` : bloc figé ; ``legacy_live`` :
    # facture figée sans snapshot (ancien modèle) rendue depuis les master data.
    source: BilledToSource = "live"
    # ``explicit`` : mode imposé par un attribut métier (``recipient_mode``) ;
    # ``inferred`` : déduit du type de payeur / des adresses (voir docs).
    mode_origin: Literal["explicit", "inferred", "snapshot"] = "inferred"

    @property
    def has_care_of(self) -> bool:
        return bool((self.care_of or "").strip())


def _party_type_value(party: Any) -> str:
    raw = getattr(party, "type", None)
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def _strategy_value(invoice: Any) -> str:
    raw = getattr(invoice, "billing_strategy", None)
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def normalize_postal_address(value: str | None) -> str:
    """Forme canonique d'une adresse pour comparaison (accents, ponctuation, pays)."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _COUNTRY_TOKENS_RE.sub(" ", text)
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def same_postal_address(left: str | None, right: str | None) -> bool:
    """Vrai si deux adresses désignent le même lieu (tolère ordre/ponctuation/pays)."""
    a = normalize_postal_address(left)
    b = normalize_postal_address(right)
    if len(a) < _MIN_COMPARABLE_ADDRESS_LEN or len(b) < _MIN_COMPARABLE_ADDRESS_LEN:
        return False
    return a == b or a in b or b in a


def is_care_of_party_type(party_type: str) -> bool:
    return (party_type or "").strip().lower() in CARE_OF_PARTY_TYPES


def is_organization_debtor_party(party: Any, *, strategy: str = "") -> bool:
    """Établissement facturé en nom propre : type établissement ou facture S2 mensuelle."""
    if (strategy or "").strip().lower() == "s2_clinic_monthly":
        return True
    return _party_type_value(party) in ORGANIZATION_DEBTOR_PARTY_TYPES


def explicit_recipient_mode(*holders: Any) -> BilledToMode | None:
    """Premier ``care_of`` / ``debtor`` rencontré parmi ``holders``, sinon ``None``.

    Precedence métier (appelant) : lien client↔payeur **puis** payeur. ``auto``
    et l'absence de champ sont ignorés (inférence par type).
    """
    for holder in holders:
        if holder is None:
            continue
        raw = getattr(holder, RECIPIENT_MODE_ATTR, None)
        value = str(getattr(raw, "value", raw) or "").strip().lower()
        if value in EXPLICIT_RECIPIENT_MODES:
            return EXPLICIT_RECIPIENT_MODES[value]
    return None


def _match_address_country(text: str | None) -> tuple[str, str, str] | None:
    """(pattern, code ISO, libellé) si un pays est mentionné dans ``text``."""
    for pattern, code, label in _COUNTRY_CODES:
        if re.search(rf"\b(?:{pattern})\b", text or "", re.IGNORECASE):
            return pattern, code, label
    return None


def detect_address_country(text: str | None) -> tuple[str, str] | None:
    """(code ISO, libellé) si un pays est mentionné dans ``text``."""
    found = _match_address_country(text)
    return (found[1], found[2]) if found else None


def structure_postal_address(raw: str | None) -> dict[str, Any]:
    """Décompose une adresse brute en ``street`` / ``postal_code`` / ``city`` / ``country``.

    ``raw`` est conservé tel quel : le rendu rejoue toujours la même entrée.
    """
    lines = split_postal_address_lines(raw)
    street_parts: list[str] = []
    extra_parts: list[str] = []
    postal_code: str | None = None
    city: str | None = None
    country: tuple[str, str] | None = None
    for line in lines:
        match = re.match(r"^(\d{4,5})\s+(.+)$", line)
        if match and postal_code is None:
            postal_code = match.group(1)
            city_text = match.group(2).strip()
            found = _match_address_country(city_text)
            if found:
                country = (found[1], found[2])
                city_text = re.sub(
                    rf"\s*\b(?:{found[0]})\b\s*$", "", city_text, flags=re.IGNORECASE
                ).strip(" ,")
            city = city_text or None
            continue
        found = _match_address_country(line)
        if found and len(line.split()) <= 2:
            country = (found[1], found[2])
            continue
        (street_parts if postal_code is None else extra_parts).append(line)
    return {
        "raw": (raw or "").strip(),
        "street": ", ".join(street_parts) or None,
        "postal_code": postal_code,
        "city": city,
        "country": country[0] if country else None,
        "country_label": country[1] if country else None,
        "extra": extra_parts or None,
    }


def _compose_address_from_structure(structured: dict[str, Any]) -> str:
    """Adresse brute équivalente (si ``raw`` absent du snapshot)."""
    parts: list[str] = []
    if structured.get("street"):
        parts.append(str(structured["street"]))
    postal_city = " ".join(
        str(p) for p in (structured.get("postal_code"), structured.get("city")) if p
    )
    if postal_city:
        parts.append(postal_city)
    for extra in structured.get("extra") or []:
        parts.append(str(extra))
    if structured.get("country_label"):
        parts.append(str(structured["country_label"]))
    return "\n".join(parts)


def _normalize_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def invoice_billed_to_is_frozen(invoice: Any) -> bool:
    """Vrai dès que la facture a quitté DRAFT (envoyée, payée, en retard, annulée)."""
    status = getattr(invoice, "status", None)
    value = str(getattr(status, "value", status) or "").strip().lower()
    return bool(value) and value != _DRAFT_STATUS_VALUE


def read_billed_to_snapshot(invoice: Any) -> dict[str, Any] | None:
    snap = _normalize_meta(getattr(invoice, "meta", None)).get(BILLED_TO_SNAPSHOT_KEY)
    return snap if isinstance(snap, dict) and snap.get("mode") else None


def billed_to_snapshot_from_party(
    party: BilledToParty,
    invoice: Any,
    *,
    reason: str,
    frozen: bool,
) -> dict[str, Any]:
    """Snapshot complet et autonome : permet de reproduire le bloc sans master data."""
    bp = party.billing_party
    now = datetime.now(UTC).isoformat()
    return {
        "version": BILLED_TO_SNAPSHOT_VERSION,
        "mode": party.mode,
        "mode_origin": party.mode_origin,
        "addressee": party.addressee,
        "care_of": party.care_of,
        "attention": party.attention,
        "residence": party.residence,
        "patient_display_name": party.patient_name,
        "payer_display_name": (getattr(bp, "display_name", None) or None)
        if bp is not None
        else None,
        "payer_type": (_party_type_value(bp) or None) if bp is not None else None,
        "payer_contact_email": party.payer_contact_email,
        "payer_contact_phone": party.payer_contact_phone,
        "payer_external_ref": party.payer_external_ref,
        "address": structure_postal_address(party.address),
        "address_owner": party.address_owner,
        "client_reference": party.client_reference,
        "client_reference_label": party.client_reference_label,
        "billing_party_id": getattr(invoice, "billing_party_id", None),
        "client_id": getattr(invoice, "client_id", None),
        "institution_patient_id": getattr(party.institution_patient, "id", None)
        or getattr(invoice, "institution_patient_id", None),
        "captured_at": now,
        "captured_reason": reason,
        "frozen_at": now if frozen else None,
    }


def billed_to_party_from_snapshot(snap: dict[str, Any]) -> BilledToParty:
    """Rejoue le bloc depuis le snapshot — aucune lecture des master data."""
    address = snap.get("address")
    if isinstance(address, dict):
        raw_address = str(address.get("raw") or "").strip() or (
            _compose_address_from_structure(address)
        )
    else:
        raw_address = str(address or "").strip()
    mode = str(snap.get("mode") or "patient_self")
    return BilledToParty(
        mode=mode,  # type: ignore[arg-type]
        addressee=str(snap.get("addressee") or "Client"),
        address=raw_address,
        care_of=snap.get("care_of") or None,
        attention=snap.get("attention") or None,
        residence=snap.get("residence") or None,
        address_owner=snap.get("address_owner") or None,
        client_reference=snap.get("client_reference") or None,
        client_reference_label=snap.get("client_reference_label") or None,
        patient_name=snap.get("patient_display_name") or None,
        payer_contact_email=snap.get("payer_contact_email") or None,
        payer_contact_phone=snap.get("payer_contact_phone") or None,
        payer_external_ref=snap.get("payer_external_ref") or None,
        source="snapshot",
        mode_origin="snapshot",
    )


def _write_billed_to_snapshot(invoice: Any, snap: dict[str, Any]) -> None:
    meta = _normalize_meta(getattr(invoice, "meta", None))
    meta[BILLED_TO_SNAPSHOT_KEY] = snap
    invoice.meta = meta


def refresh_billed_to_snapshot_if_draft(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
    reason: str = "pdf_generation",
) -> dict[str, Any] | None:
    """Brouillon : le snapshot suit les master data (il reflète le dernier PDF construit).

    Facture figée : jamais réécrit. Sans snapshot (ancienne facture), rien n'est
    créé a posteriori : le rendu bascule en ``legacy_live`` explicitement journalisé.
    """
    if invoice_billed_to_is_frozen(invoice):
        return read_billed_to_snapshot(invoice)
    party = _resolve_invoice_billed_to_live(invoice, bookings_by_id=bookings_by_id)
    snap = billed_to_snapshot_from_party(party, invoice, reason=reason, frozen=False)
    _write_billed_to_snapshot(invoice, snap)
    from services.documents.qr_creditor import refresh_qr_creditor_snapshot_if_draft
    from services.documents.qr_debtor import refresh_qr_debtor_snapshot_if_draft

    refresh_qr_debtor_snapshot_if_draft(invoice, reason=reason)
    refresh_qr_creditor_snapshot_if_draft(invoice, reason=reason)
    return snap


def freeze_billed_to_snapshot(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
    reason: str = "status_transition",
) -> dict[str, Any]:
    """À appeler quand la facture quitte DRAFT (envoi e-mail / papier / lot).

    Un snapshot existant (celui du PDF effectivement construit) est conservé et
    marqué figé ; sinon il est capturé maintenant depuis les master data.
    """
    from services.documents.qr_creditor import freeze_qr_creditor_snapshot
    from services.documents.qr_debtor import freeze_qr_debtor_snapshot

    existing = read_billed_to_snapshot(invoice)
    if existing is not None:
        if not existing.get("frozen_at"):
            frozen = dict(existing)
            frozen["frozen_at"] = datetime.now(UTC).isoformat()
            frozen["frozen_reason"] = reason
            _write_billed_to_snapshot(invoice, frozen)
            freeze_qr_debtor_snapshot(invoice, reason=reason)
            freeze_qr_creditor_snapshot(invoice, reason=reason)
            return frozen
        freeze_qr_debtor_snapshot(invoice, reason=reason)
        freeze_qr_creditor_snapshot(invoice, reason=reason)
        return existing
    party = _resolve_invoice_billed_to_live(invoice, bookings_by_id=bookings_by_id)
    snap = billed_to_snapshot_from_party(party, invoice, reason=reason, frozen=True)
    snap["frozen_reason"] = reason
    _write_billed_to_snapshot(invoice, snap)
    freeze_qr_debtor_snapshot(invoice, reason=reason)
    freeze_qr_creditor_snapshot(invoice, reason=reason)
    logger.info(
        "[Facturé à] Snapshot figé à la transition (invoice_id=%s, mode=%s, reason=%s).",
        getattr(invoice, "id", None),
        party.mode,
        reason,
    )
    return snap


def _patient_own_addresses(invoice: Any, *, live_address: str | None) -> list[str]:
    """Adresses appartenant explicitement au patient (domicile courant, facturation client)."""
    candidates: list[str] = []
    if live_address:
        candidates.append(live_address)
    client = getattr(invoice, "client", None)
    if client is not None:
        try:
            explicit = getattr(client, "billing_address_secure", None) or ""
        except Exception:
            explicit = getattr(client, "billing_address", None) or ""
        if explicit and str(explicit).strip():
            candidates.append(str(explicit).strip())
    return candidates


def party_address_belongs_to_patient(
    invoice: Any, party_address: str | None, *, live_address: str | None
) -> bool:
    """Vrai si l'adresse de facturation du tiers est en réalité celle du patient."""
    if not (party_address or "").strip():
        return False
    return any(
        same_postal_address(party_address, own)
        for own in _patient_own_addresses(invoice, live_address=live_address)
    )


def _institution_booking_customer_name(
    invoice: Any, *, bookings_by_id: dict[int, Any] | None
) -> str | None:
    """Nom du patient porté par les bookings (clients institution sans patient résolu)."""
    for booking in iter_invoice_bookings(invoice, bookings_by_id=bookings_by_id):
        name = (getattr(booking, "customer_name", None) or "").strip()
        if name:
            return name
    return None


def _resolve_patient_identity(
    invoice: Any, *, bookings_by_id: dict[int, Any] | None
) -> tuple[str | None, str | None, Any]:
    """(nom, adresse, patient institutionnel) du patient de CETTE facture."""
    patient = resolve_invoice_institution_patient(
        invoice, bookings_by_id=bookings_by_id
    )
    name, address = live_patient_payer_identity(invoice, bookings_by_id=bookings_by_id)
    client = getattr(invoice, "client", None)
    if (
        client is not None
        and bool(getattr(client, "is_institution", False))
        and patient is None
    ):
        booking_name = _institution_booking_customer_name(
            invoice, bookings_by_id=bookings_by_id
        )
        if booking_name:
            # Compte institution partagé : le patient réel vient du booking, et son
            # adresse ne doit jamais être celle de l'institution.
            name = booking_name
            if _strategy_value(invoice) == "s1_patient":
                address = None
    return name, address, patient


def _patient_self_party(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None,
    billing_party: Any = None,
    fallback_address: str | None = None,
) -> BilledToParty:
    name, address, patient = _resolve_patient_identity(
        invoice, bookings_by_id=bookings_by_id
    )
    client = getattr(invoice, "client", None)
    if not name:
        name = (getattr(billing_party, "display_name", None) or "").strip() or "Client"
    if not address:
        address = (fallback_address or "").strip() or ""
    if not address and billing_party is not None:
        address = (getattr(billing_party, "billing_address", None) or "").strip()
    return BilledToParty(
        mode="patient_self",
        addressee=name,
        address=address,
        residence=invoice_residence_label(patient=patient, client=client) or None,
        address_owner=name,
        billing_party=billing_party,
        institution_patient=patient,
        patient_name=name,
    )


def _payer_contact_fields(party: Any) -> dict[str, str | None]:
    """Coordonnées du tiers payeur telles que connues au moment de la résolution."""

    def _clean(attr: str) -> str | None:
        return (getattr(party, attr, None) or "").strip() or None

    return {
        "payer_contact_email": _clean("contact_email"),
        "payer_contact_phone": _clean("contact_phone"),
        "payer_external_ref": _clean("external_ref"),
    }


def _client_reference_for(party: Any, link: Any) -> tuple[str | None, str | None]:
    """Référence du patient chez le payeur (ex. No SPC) si renseignée sur le lien."""
    reference = (getattr(link, "client_reference", None) or "").strip() if link else ""
    if not reference:
        return None, None
    display = (getattr(party, "display_name", None) or "").upper()
    if "SPC" in display:
        return reference, "No. SPC"
    return reference, "Référence"


def resolve_invoice_billed_to(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
    use_snapshot: bool = True,
) -> BilledToParty:
    """Résout le destinataire du bloc « Facturé à ».

    Unique point de décision pour le PDF ReportLab, le constructeur HTML, la
    régénération et l'envoi : les renderers ne font que formater le résultat.

    - Brouillon : master data courantes (contrat « Régénérer PDF »).
    - Facture figée (hors DRAFT) avec snapshot : snapshot uniquement — les master
      data (nom du patient, adresse du tiers, contact…) peuvent changer sans
      toucher au document émis.
    - Facture figée sans snapshot (ancien modèle) : fallback ``legacy_live``,
      journalisé explicitement.
    """
    if use_snapshot and invoice_billed_to_is_frozen(invoice):
        snap = read_billed_to_snapshot(invoice)
        if snap is not None:
            return billed_to_party_from_snapshot(snap)
        logger.warning(
            "[Facturé à] Facture figée sans snapshot (invoice_id=%s, status=%s) : "
            "rendu depuis les master data courantes (legacy_live).",
            getattr(invoice, "id", None),
            getattr(getattr(invoice, "status", None), "value", None),
        )
        live = _resolve_invoice_billed_to_live(invoice, bookings_by_id=bookings_by_id)
        return replace(live, source="legacy_live")
    return _resolve_invoice_billed_to_live(invoice, bookings_by_id=bookings_by_id)


def _resolve_invoice_billed_to_live(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> BilledToParty:
    """Résolution depuis le modèle de données courant (patient, payeur, lien)."""
    from models import BillingParty as BillingPartyModel
    from models.billing_party import ClientBillingParty

    invoice_id = getattr(invoice, "id", None)
    client_id = getattr(invoice, "client_id", None)
    bp_id = getattr(invoice, "billing_party_id", None)

    if bp_id:
        party = getattr(invoice, "billing_party", None)
        if party is None or getattr(party, "id", None) != bp_id:
            party = BillingPartyModel.query.get(bp_id)
        if party is None:
            logger.warning(
                "[Facturé à] BillingParty introuvable (invoice_id=%s, billing_party_id=%s).",
                invoice_id,
                bp_id,
            )
            return BilledToParty(
                mode="unresolved_party", addressee="Payeur", address=""
            )

        party_type = _party_type_value(party)
        strategy = _strategy_value(invoice)
        party_address = (getattr(party, "billing_address", None) or "").strip()

        # Payeur = patient : identité et domicile courants font foi (jamais le snapshot).
        if party_type == "patient":
            return _patient_self_party(
                invoice, bookings_by_id=bookings_by_id, billing_party=party
            )

        party_name = (getattr(party, "display_name", None) or "Payeur").strip()
        link = None
        if client_id is not None:
            link = ClientBillingParty.query.filter_by(
                client_id=client_id, billing_party_id=bp_id
            ).first()

        # Rôle déclaré : le lien client↔payeur surcharge le payeur (surcharge
        # par patient). ``auto`` / absent ⇒ inférence par type.
        explicit = explicit_recipient_mode(link, party)
        mode_origin: Literal["explicit", "inferred"] = (
            "explicit" if explicit else "inferred"
        )

        if explicit == "organization_debtor" or (
            explicit is None and is_organization_debtor_party(party, strategy=strategy)
        ):
            # Organisme débiteur en nom propre : jamais de « c/o ».
            # - inféré (clinique/EMS/hôpital/assurance, S2) : multi-patients possible,
            #   aucun contact ni référence déduits d'un lien patient (comportement 16.09) ;
            # - déclaré « debtor » sur une facture mono-patient (S1) : le contact
            #   (« À l'att. de ») et la référence du lien s'appliquent.
            attention = None
            reference = reference_label = None
            patient_name = None
            if (
                explicit is not None
                and link is not None
                and strategy != "s2_clinic_monthly"
            ):
                attention = (getattr(link, "contact_name", None) or "").strip() or None
                reference, reference_label = _client_reference_for(party, link)
                patient_name, _address, _patient = _resolve_patient_identity(
                    invoice, bookings_by_id=bookings_by_id
                )
            return BilledToParty(
                mode="organization_debtor",
                addressee=party_name,
                address=party_address,
                attention=attention,
                address_owner=party_name,
                client_reference=reference,
                client_reference_label=reference_label,
                billing_party=party,
                patient_name=patient_name,
                mode_origin=mode_origin,
                **_payer_contact_fields(party),
            )

        # Tiers de correspondance : le lien client↔payeur doit exister.
        if link is None:
            logger.info(
                "[Facturé à] Lien client↔tiers payeur absent (invoice_id=%s) : "
                "facturé au domicile du patient.",
                invoice_id,
            )
            return _patient_self_party(invoice, bookings_by_id=bookings_by_id)

        patient_name, patient_address, patient = _resolve_patient_identity(
            invoice, bookings_by_id=bookings_by_id
        )
        contact = (getattr(link, "contact_name", None) or "").strip() or None
        reference, reference_label = _client_reference_for(party, link)

        if not patient_name or patient_name.casefold() == party_name.casefold():
            # Pas d'identité patient distincte : le payeur est le destinataire.
            return BilledToParty(
                mode="organization_debtor",
                addressee=party_name,
                address=party_address,
                attention=contact,
                address_owner=party_name,
                client_reference=reference,
                client_reference_label=reference_label,
                billing_party=party,
                institution_patient=patient,
                patient_name=patient_name,
                mode_origin=mode_origin,
                **_payer_contact_fields(party),
            )

        if explicit is None and party_address_belongs_to_patient(
            invoice, party_address, live_address=patient_address
        ):
            # L'adresse de facturation appartient au patient : pas de « c/o » artificiel.
            logger.info(
                "[Facturé à] Tiers payeur domicilié chez le patient (invoice_id=%s) : "
                "aucune ligne c/o.",
                invoice_id,
            )
            return BilledToParty(
                mode="patient_self",
                addressee=patient_name,
                address=patient_address or party_address,
                residence=invoice_residence_label(
                    patient=patient, client=getattr(invoice, "client", None)
                )
                or None,
                address_owner=patient_name,
                client_reference=reference,
                client_reference_label=reference_label,
                billing_party=party,
                institution_patient=patient,
                patient_name=patient_name,
                mode_origin=mode_origin,
            )

        return BilledToParty(
            mode="patient_care_of",
            addressee=patient_name,
            address=party_address,
            care_of=party_name,
            attention=contact,
            address_owner=party_name,
            client_reference=reference,
            client_reference_label=reference_label,
            billing_party=party,
            institution_patient=patient,
            patient_name=patient_name,
            mode_origin=mode_origin,
            **_payer_contact_fields(party),
        )

    # Ancien modèle : facturation à un client institution.
    bill_to_client_id = getattr(invoice, "bill_to_client_id", None)
    if bill_to_client_id and bill_to_client_id != client_id:
        from models import Client as ClientModel

        institution = getattr(invoice, "bill_to_client", None)
        if institution is None or getattr(institution, "id", None) != bill_to_client_id:
            institution = ClientModel.query.get(bill_to_client_id)
        if institution is not None and bool(
            getattr(institution, "is_institution", False)
        ):
            logger.info(
                "[Facturé à] Fallback legacy bill_to_client_id (invoice_id=%s, bill_to_client_id=%s).",
                invoice_id,
                bill_to_client_id,
            )
            inst_name = (
                getattr(institution, "institution_name", None) or ""
            ).strip() or "Institution"
            return BilledToParty(
                mode="legacy_institution",
                addressee=inst_name,
                address=(getattr(institution, "billing_address", None) or "").strip(),
                address_owner=inst_name,
            )
        logger.warning(
            "[Facturé à] bill_to_client_id=%s défini mais institution introuvable (invoice_id=%s).",
            bill_to_client_id,
            invoice_id,
        )
        return BilledToParty(
            mode="legacy_institution", addressee="Institution", address=""
        )

    logger.info(
        "[Facturé à] Fallback client bénéficiaire (invoice_id=%s, client_id=%s).",
        invoice_id,
        client_id,
    )
    return _patient_self_party(invoice, bookings_by_id=bookings_by_id)


def billed_to_name_lines(
    party: BilledToParty,
    *,
    name_formatter: Any = None,
) -> list[str]:
    """Lignes « nom » du bloc, dans l'ordre d'impression.

    ``name_formatter`` (ex. nom de famille en majuscules) s'applique au patient /
    organisme et au tiers « c/o », jamais au préfixe ni au contact.
    """
    fmt = name_formatter or (lambda value: value)
    addressee = (party.addressee or "").strip() or (
        "Payeur"
        if party.mode in {"organization_debtor", "unresolved_party"}
        else "Client"
    )
    lines: list[str] = [fmt(addressee)]

    care_of = (party.care_of or "").strip()
    if care_of and care_of.casefold() != addressee.casefold():
        prefix_match = _CARE_OF_PREFIX_RE.match(care_of)
        if prefix_match:
            # « c/o » déjà explicite dans les données : ne pas le dupliquer.
            rest = care_of[prefix_match.end() :].strip()
            lines.append(f"c/o {fmt(rest)}" if rest else care_of)
        else:
            lines.append(f"c/o {fmt(care_of)}")

    attention = (party.attention or "").strip()
    if attention and attention.casefold() not in {
        addressee.casefold(),
        care_of.casefold(),
    }:
        lines.append(f"À l'att. de {attention}")

    residence = (party.residence or "").strip()
    if residence and all(residence.casefold() not in ln.casefold() for ln in lines):
        lines.append(residence)
    return lines
