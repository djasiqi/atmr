"""Génération PDF Bon de transport et Rapport de mission institution (ReportLab platypus)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from services.institutions.mission_report_context import MissionReportContext

MISSING = "—"
_EMPTY_VALUES = frozenset({MISSING, "-", "—", "N/A", "Non renseigné", ""})

# Palette institutionnelle (lisible N&B : hiérarchie par graisse/taille/filets)
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#555555")
BORDER = colors.HexColor("#d9dde1")
ACCENT = colors.HexColor("#00796b")

_LOGO_PATH = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "lirie" / "logo-lirie.png"
)
_LOGO_TARGET_WIDTH = 2.2 * cm
_LOGO_MAX_HEIGHT = 1.4 * cm
_VOUCHER_LOGO_TARGET_WIDTH = 3.4 * cm
_VOUCHER_LOGO_MAX_HEIGHT = 2.0 * cm

_FOOTER_TEXT = (
    "Document généré par LIRIE — plateforme de coordination des transports. "
    "www.lirie.ch · info@lirie.ch"
)
_IDENTITY_VALUE_MAX_LEN = 65  # ~2 lignes dans la colonne valeur (13,2 cm, police 9)

# PDF-LONGCONTENT-05 : limites par champ (résilience layout, ellipse au-delà)
_MAX_PATIENT = 80
_MAX_INSTITUTION = 80
_MAX_CARRIER = 60
_MAX_ADDRESS = 160
_MAX_DESTINATION = 90
_MAX_MEDICAL_NOTES = 2408
_MAX_VOUCHER_NOTES = 120  # bon = document terrain : remarque courte
_MAX_VOUCHER_PATIENT = 40  # bon : nom patient sur une seule ligne
_MAX_VOUCHER_DESTINATION = 55  # bon : destination reconnaissable sur une ligne

# Bon de transport : libellé du type de transport (visible dès le bloc TRANSPORT)
_VOUCHER_TRANSPORT_TYPE_LABELS = {
    "stretcher": "Brancard",
    "wheelchair": "Fauteuil roulant",
    "assisted": "Assis (accompagné)",
    "ambulatory": "Assis",
}

VoucherLayoutName = Literal["legacy", "operational", "ultra_compact", "medical"]
_REVIEW_ONLY_LAYOUTS = frozenset({"ultra_compact", "medical"})


@dataclass(frozen=True)
class RouteStop:
    """Étape trajet normalisée pour les layouts chauffeur (sans « Étape N »)."""

    kind: str
    label: str
    address: str
    planned_time: str | None


@dataclass(frozen=True)
class VoucherLayoutOptions:
    """Micro-design Operational — hero + confirmation.

    Défaut production : hero inline + confirmation inline ultra compacte.
    `stack` / `row` conservés pour comparaison de revue uniquement.
    """

    hero_style: Literal["inline", "split"] = "inline"
    signature_style: Literal["confirmation_inline", "stack", "row"] = "confirmation_inline"


@dataclass(frozen=True)
class VoucherPresentation:
    """Couche présentation unique — les layouts ne recalculent pas la donnée."""

    patient_name: str
    patient_dob: str | None
    route_stops: tuple[RouteStop, ...] = ()
    primary_time: str | None = None
    needs_bullets: tuple[str, ...] = ()
    needs_remark: str | None = None
    needs_floor_info: str | None = None
    institution_name: str | None = None
    contact_line: str | None = None
    transport_type_label: str | None = None
    reference: str = ""
    mission_date: str | None = None
    carrier_name: str | None = None
    carrier_is_external: bool = False
    driver_name: str | None = None
    patient_address: str | None = None
    billing_label: str | None = None
    time_context_label: str | None = None
    has_needs: bool = False
    verify_url: str = "https://www.lirie.ch"
    logo_url: str | None = None


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "docTitle": ParagraphStyle(
            "DocTitle",
            parent=base["Heading1"],
            fontSize=13,
            leading=16,
            spaceAfter=2,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "refLarge": ParagraphStyle(
            "RefLarge",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            spaceAfter=2,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "MissionBody",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "MissionSmall",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "noteItalic": ParagraphStyle(
            "MissionNoteItalic",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=MUTED,
            fontName="Helvetica-Oblique",
        ),
        "cardValueBold": ParagraphStyle(
            "CardValueBold",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=INK,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        ),
        "sectionTitle": ParagraphStyle(
            "SectionTitle",
            parent=base["Heading2"],
            fontSize=9,
            leading=11,
            spaceBefore=6,
            spaceAfter=3,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "identityLabel": ParagraphStyle(
            "IdentityLabel",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
        ),
        "identityValue": ParagraphStyle(
            "IdentityValue",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=INK,
        ),
        "railTitle": ParagraphStyle(
            "RailTitle",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "railBody": ParagraphStyle(
            "RailBody",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "railBullet": ParagraphStyle(
            "RailBullet",
            parent=base["Normal"],
            fontSize=10,
            leading=12,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
            alignment=TA_RIGHT,
        ),
        "heroName": ParagraphStyle(
            "HeroName",
            parent=base["Normal"],
            fontSize=16,
            leading=19,
            textColor=INK,
            fontName="Helvetica-Bold",
            spaceAfter=2,
        ),
        "heroDob": ParagraphStyle(
            "HeroDob",
            parent=base["Normal"],
            fontSize=10,
            leading=12,
            textColor=MUTED,
            spaceAfter=2,
        ),
        "heroInstitution": ParagraphStyle(
            "HeroInstitution",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceAfter=4,
        ),
        "timeProminent": ParagraphStyle(
            "TimeProminent",
            parent=base["Normal"],
            fontSize=12,
            leading=14,
            textColor=INK,
            fontName="Helvetica-Bold",
            spaceAfter=2,
        ),
        "metaMuted": ParagraphStyle(
            "MetaMuted",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "routeArrow": ParagraphStyle(
            "RouteArrow",
            parent=base["Normal"],
            fontSize=10,
            leading=12,
            textColor=MUTED,
            alignment=1,
            spaceBefore=2,
            spaceAfter=2,
        ),
        "sectionLabel": ParagraphStyle(
            "SectionLabel",
            parent=base["Normal"],
            fontSize=9,
            leading=11,
            textColor=INK,
            fontName="Helvetica-Bold",
            spaceBefore=4,
            spaceAfter=2,
        ),
    }


def _has_value(val: Any) -> bool:
    if val is None:
        return False
    return str(val).strip() not in _EMPTY_VALUES


def _resolve_logo() -> str | None:
    """Chemin logo LIRIE ou None (fallback texte — jamais d'exception)."""
    try:
        if _LOGO_PATH.is_file():
            return str(_LOGO_PATH)
    except Exception:
        pass
    return None


def _resolve_upload_logo_path(logo_url: str | None) -> str | None:
    """Chemin local depuis logo_url (/uploads/...), None si absent ou externe."""
    if not _has_value(logo_url):
        return None
    raw = str(logo_url).strip()
    if raw.startswith(("http://", "https://")):
        return None
    clean = raw.lstrip("/")
    if clean.startswith("uploads/"):
        clean = clean[8:]
    try:
        from flask import current_app

        uploads_root = Path(
            current_app.config.get(
                "UPLOADS_DIR",
                str(Path(current_app.root_path) / "uploads"),
            )
        ).resolve()
    except Exception:
        uploads_root = Path(__file__).resolve().parent.parent.parent / "uploads"
    try:
        candidate = (uploads_root / clean).resolve()
        candidate.relative_to(uploads_root.resolve())
    except (ValueError, OSError):
        return None
    return str(candidate) if candidate.is_file() else None


def _build_logo_image(
    logo_path: str,
    *,
    h_align: str = "RIGHT",
    target_width: float | None = None,
    max_height: float | None = None,
) -> Image | None:
    """Logo avec ratio préservé (pas d'upscale au-delà de la largeur native)."""
    try:
        reader = ImageReader(logo_path)
        native_w, native_h = reader.getSize()
        if not native_w or not native_h:
            return None
        aspect = native_h / native_w
        max_w = target_width if target_width is not None else _LOGO_TARGET_WIDTH
        max_h = max_height if max_height is not None else _LOGO_MAX_HEIGHT
        width = min(max_w, native_w)
        height = width * aspect
        if height > max_h:
            height = max_h
            width = height / aspect
        logo = Image(logo_path, width=width, height=height)
        logo.hAlign = h_align
        return logo
    except Exception:
        return None


def _rule(width: float = 17 * cm) -> Table:
    """Filet horizontal discret."""
    table = Table([[""]], colWidths=[width], rowHeights=[1])
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _section_title(title: str, *, with_rule: bool = True) -> list[Any]:
    """Ancre visuelle de section : ■ LIBELLÉ + filet inférieur optionnel."""
    st = _styles()
    clean = title.upper().strip()
    if not clean.startswith("■"):
        clean = f"■ {clean}"
    flow: list[Any] = [Paragraph(clean, st["sectionTitle"])]
    if with_rule:
        flow.append(_rule())
    return flow


def _publisher_block(logo_url: str | None = None) -> Any:
    """Logo institution (prioritaire) ou LIRIE — colonne droite (legacy)."""
    logo_path = _resolve_upload_logo_path(logo_url) or _resolve_logo()
    if logo_path:
        logo = _build_logo_image(logo_path)
        if logo is not None:
            table = Table([[logo]], colWidths=[6 * cm])
            table.setStyle(
                TableStyle(
                    [("ALIGN", (0, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP")]
                )
            )
            return table
    return Spacer(1, 0.01 * cm)


_VOUCHER_QR_SIZE = 1.2 * cm


def _voucher_logo_block(logo_url: str | None = None) -> Any:
    """Logo institution (prioritaire) ou LIRIE, aligné à gauche."""
    logo_path = _resolve_upload_logo_path(logo_url) or _resolve_logo()
    if logo_path:
        logo = _build_logo_image(
            logo_path,
            h_align="LEFT",
            target_width=_VOUCHER_LOGO_TARGET_WIDTH,
            max_height=_VOUCHER_LOGO_MAX_HEIGHT,
        )
        if logo is not None:
            table = Table([[logo]], colWidths=[11 * cm])
            table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            return table
    return Spacer(1, 0.01 * cm)


def _voucher_qr_block(verify_url: str) -> Any:
    """QR LIRIE aligné à droite (même bandeau que le logo)."""
    st = _styles()
    qr = _build_qr_image(verify_url, size=_VOUCHER_QR_SIZE)
    if qr is not None:
        table = Table([[qr]], colWidths=[6 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return table
    return Paragraph(verify_url, st["small"])


def _voucher_header_table(
    title_lines: list[Any], verify_url: str, logo_url: str | None = None
) -> Table:
    """Bandeau bon : logo institution (gauche) + QR (droite), puis titre/référence."""
    header_table = Table(
        [
            [_voucher_logo_block(logo_url), _voucher_qr_block(verify_url)],
            [title_lines, Spacer(1, 0.01 * cm)],
        ],
        colWidths=[11 * cm, 6 * cm],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ]
        )
    )
    return header_table


def _report_header_table(
    title_lines: list[Any], logo_url: str | None = None
) -> Table:
    """Bandeau rapport : logo institution (gauche), puis titre/référence."""
    header_table = Table(
        [
            [_voucher_logo_block(logo_url), Spacer(1, 0.01 * cm)],
            [title_lines, Spacer(1, 0.01 * cm)],
        ],
        colWidths=[11 * cm, 6 * cm],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ]
        )
    )
    return header_table


def _document_header(ctx: MissionReportContext, doc_title: str) -> list[Any]:
    """En-tête rapport : logo institution + titre, référence mission, statut/dates."""
    st = _styles()
    left_lines: list[Any] = [
        Paragraph(doc_title.upper(), st["docTitle"]),
        Paragraph(f"Référence mission : {ctx.reference}", st["refLarge"]),
    ]
    inst_logo = ctx.institution_snapshot.get("logo_url")

    flow: list[Any] = [
        _report_header_table(left_lines, logo_url=inst_logo),
        Spacer(1, 0.12 * cm),
        _rule(),
        Spacer(1, 0.06 * cm),
    ]
    mission = ctx.mission_info
    edition = ctx.traceability.get("edition_date") or ctx.traceability.get(
        "generated_at_label", MISSING
    )
    meta_lines = [
        f"Statut : {ctx.status_label} · Date mission : {mission.get('mission_date', MISSING)}",
        f"Date d'édition : {edition}",
    ]
    for line in meta_lines:
        flow.append(Paragraph(line, st["body"]))
    flow.append(Spacer(1, 0.12 * cm))
    return flow


def _voucher_header(ctx: MissionReportContext) -> list[Any]:
    """En-tête bon de transport : titre + référence · date mission + logo."""
    st = _styles()
    mission_date = ctx.mission_info.get("mission_date", MISSING)
    ref_line = ctx.reference
    if _has_value(mission_date):
        ref_line = f"{ctx.reference} · {mission_date}"
    left_lines: list[Any] = [
        Paragraph("BON DE TRANSPORT", st["docTitle"]),
        Paragraph(ref_line, st["refLarge"]),
    ]
    verify_url = ctx.traceability.get("verify_url") or "https://www.lirie.ch"
    inst_logo = ctx.institution_snapshot.get("logo_url")
    return [
        _voucher_header_table(left_lines, verify_url, logo_url=inst_logo),
        Spacer(1, 0.3 * cm),
    ]


def _compact_identity_table(rows: list[tuple[str, str | None]]) -> Table:
    """Tableau clé/valeur compact ; lignes omises si valeur absente."""
    st = _styles()
    data: list[list[Any]] = []
    for label, value in rows:
        if not _has_value(value):
            continue
        data.append(
            [
                Paragraph(label, st["identityLabel"]),
                Paragraph(str(value), st["identityValue"]),
            ]
        )
    if not data:
        data = [[Paragraph("—", st["identityValue"]), Paragraph(MISSING, st["identityValue"])]]
    table = Table(data, colWidths=[3.8 * cm, 13.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _identity_group_tables(groups: list[list[tuple[str, str | None]]]) -> list[Any]:
    """Rend plusieurs groupes identité avec espacement inter-groupes."""
    flow: list[Any] = []
    for group in groups:
        filtered = [(label, val) for label, val in group if _has_value(val)]
        if not filtered:
            continue
        if flow:
            flow.append(Spacer(1, 0.08 * cm))
        flow.append(_compact_identity_table(filtered))
    if not flow:
        flow.append(_compact_identity_table([]))
    flow.append(Spacer(1, 0.12 * cm))
    return flow


def _is_mission_cancelled(ctx: MissionReportContext) -> bool:
    """Mission annulée (libellé FR ou statut TR)."""
    return "annul" in ctx.status_label.lower()


def _truncate_field(value: str | None, *, max_len: int = _IDENTITY_VALUE_MAX_LEN) -> str | None:
    """Limite un champ identité à ~2 lignes (ellipse au-delà)."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _truncate_medical_notes(notes: str, *, max_len: int = _MAX_MEDICAL_NOTES) -> str:
    """Commentaires médicaux libres tronqués (les besoins critiques restent entiers)."""
    text = notes.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " […]"


def _build_report_identity_table(ctx: MissionReportContext) -> list[Any]:
    """Identité compacte groupée pour le rapport de mission."""
    patient = ctx.patient_block
    inst = ctx.institution_snapshot
    carrier = ctx.carrier_block

    full_name = patient.get("full_name")
    dob = patient.get("dob")
    if _has_value(full_name) and _has_value(dob):
        patient_line = f"{full_name} ({dob})"
    else:
        patient_line = full_name

    service = inst.get("service", MISSING)
    requester = inst.get("requester_name", MISSING)
    if _has_value(service) and _has_value(requester):
        contact_line = f"{service} · {requester}"
    elif _has_value(service):
        contact_line = str(service)
    elif _has_value(requester):
        contact_line = str(requester)
    else:
        contact_line = None

    patient_group = [
        ("Patient", _truncate_field(patient_line, max_len=_MAX_PATIENT)),
        ("Dossier", patient.get("dossier_number")),
        ("Unité", _truncate_field(patient.get("room"))),
    ]
    institution_group = [
        ("Institution", _truncate_field(inst.get("name"), max_len=_MAX_INSTITUTION)),
        ("Contact", _truncate_field(contact_line)),
    ]
    carrier_group: list[tuple[str, str | None]] = [
        ("Transporteur", _truncate_field(carrier.get("name"), max_len=_MAX_CARRIER)),
    ]
    if _has_value(carrier.get("driver_name")):
        carrier_group.append(("Chauffeur", carrier.get("driver_name")))
    if _has_value(carrier.get("vehicle")):
        carrier_group.append(("Véhicule", carrier.get("vehicle")))

    return _identity_group_tables([patient_group, institution_group, carrier_group])


def _build_voucher_contact_line(inst: dict[str, Any]) -> str | None:
    """Contact terrain : service · nom référent · téléphone (conditionnel)."""
    parts: list[str] = []
    service = inst.get("service")
    name = inst.get("requester_name")
    phone = inst.get("requester_phone")
    if _has_value(service):
        parts.append(str(service))
    if _has_value(name):
        parts.append(str(name))
    if _has_value(phone):
        parts.append(str(phone))
    return " · ".join(parts) if parts else None


def _build_voucher_identity_table(ctx: MissionReportContext) -> list[Any]:
    """Identité terrain : patient/institution puis transporteur/chauffeur/contact."""
    patient = ctx.patient_block
    inst = ctx.institution_snapshot
    carrier = ctx.carrier_block

    # PDF-VOUCHER-04 : si le patient est le débiteur, l'adresse et la mention de
    # facturation deviennent utiles (rapprochement bon ↔ facture sans ouvrir LIRIE).
    bills_patient = ctx.request_classification.get("billing_target") == "patient"

    patient_group: list[tuple[str, str | None]] = [
        ("Patient", _truncate_field(patient.get("full_name"), max_len=_MAX_VOUCHER_PATIENT)),
        ("Naissance", patient.get("dob")),
    ]
    if bills_patient and _has_value(patient.get("address")):
        patient_group.append(
            ("Adresse patient", _truncate_field(patient.get("address"), max_len=_MAX_ADDRESS))
        )
    patient_group.append(
        ("Institution", _truncate_field(inst.get("name"), max_len=_MAX_INSTITUTION))
    )
    if bills_patient:
        patient_group.append(("Facturation", "Patient"))
    # Type transport = 1re info opérationnelle → placé juste après l'identité patient
    type_group: list[tuple[str, str | None]] = []
    type_label = _VOUCHER_TRANSPORT_TYPE_LABELS.get(
        ctx.request_classification.get("mobility_level")
    )
    if type_label:
        type_group.append(("Type transport", type_label))

    carrier_group: list[tuple[str, str | None]] = [
        (
            "Transporteur externe" if carrier.get("is_external") else "Transporteur",
            _truncate_field(carrier.get("name"), max_len=_MAX_CARRIER),
        ),
    ]
    if _has_value(carrier.get("driver_name")):
        carrier_group.append(("Chauffeur", carrier.get("driver_name")))
    contact_line = _build_voucher_contact_line(inst)
    if contact_line:
        carrier_group.append(("Contact", _truncate_field(contact_line)))

    return _identity_group_tables([patient_group, type_group, carrier_group])


def _short_step_title(title: str, *, max_len: int = 42) -> str:
    """Titre d'étape court — jamais une adresse longue en titre."""
    raw = (title or "").strip()
    if not _has_value(raw):
        return "Destination"
    if "," in raw and len(raw) > 20:
        raw = raw.split(",")[0].strip()
    if len(raw) > max_len:
        return raw[: max_len - 1].rstrip() + "…"
    return raw


def _step_time_line(step: dict[str, Any], *, cancelled: bool = False) -> str | None:
    """Ligne horaires séparée : « Prévu : 11:30 · Réel : 11:34 » (réel masqué si annulé)."""
    planned = step.get("planned_time", MISSING)
    actual = step.get("actual_time", MISSING)
    bits: list[str] = []
    if _has_value(planned):
        bits.append(f"Prévu : {planned}")
    if _has_value(actual) and not cancelled:
        bits.append(f"Réel : {actual}")
    return " · ".join(bits) if bits else None


def _step_value_cell(
    step: dict[str, Any], *, cancelled: bool = False, max_addr: int = _MAX_ADDRESS
) -> list[Any]:
    """Cellule valeur d'une étape : adresse (tronquée + wrapping) + horaires sur ligne séparée."""
    st = _styles()
    addr = _truncate_field(str(step.get("address", MISSING)), max_len=max_addr)
    cell: list[Any] = [Paragraph(addr or MISSING, st["body"])]
    time_line = _step_time_line(step, cancelled=cancelled)
    if time_line:
        cell.append(Paragraph(time_line, st["small"]))
    return cell


def _route_table(rows: list[tuple[str, list[Any]]]) -> Table:
    """Tableau trajet : label court (gras) + cellule valeur multi-lignes (adresse + horaires)."""
    st = _styles()
    data: list[list[Any]] = [
        [Paragraph(f"<b>{label}</b>", st["body"]), value] for label, value in rows
    ]
    table = Table(data, colWidths=[3.5 * cm, 13.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _build_route_simple(ctx: MissionReportContext) -> list[Any]:
    """Trajet simple : Départ / Destination (adresse wrappée + horaires séparés)."""
    cancelled = _is_mission_cancelled(ctx)
    steps = ctx.route_steps
    departure = next((s for s in steps if s.get("kind") == "departure"), None)
    destination = next((s for s in steps if s.get("kind") == "destination"), None)
    if destination is None:
        destination = next((s for s in steps if s.get("kind") == "dropoff"), None)
    rows: list[tuple[str, list[Any]]] = []
    if departure and _has_value(departure.get("address")):
        rows.append(("Départ", _step_value_cell(departure, cancelled=cancelled)))
    if destination and _has_value(destination.get("address")):
        rows.append(
            ("Destination", _step_value_cell(destination, cancelled=cancelled, max_addr=_MAX_DESTINATION))
        )
    elif len(steps) >= 2 and _has_value(steps[-1].get("address")):
        rows.append(
            ("Destination", _step_value_cell(steps[-1], cancelled=cancelled, max_addr=_MAX_DESTINATION))
        )
    if not rows:
        return [Paragraph(MISSING, _styles()["body"])]
    return [_route_table(rows)]


def _build_route_round_trip(ctx: MissionReportContext) -> list[Any]:
    """Trajet aller-retour : Départ / Destination / Retour."""
    cancelled = _is_mission_cancelled(ctx)
    steps = ctx.route_steps
    departure = next((s for s in steps if s.get("kind") == "departure"), None)
    destination = next((s for s in steps if s.get("kind") == "destination"), None)
    ret = next((s for s in steps if s.get("kind") == "return"), None)
    rows: list[tuple[str, list[Any]]] = []
    if departure and _has_value(departure.get("address")):
        rows.append(("Départ", _step_value_cell(departure, cancelled=cancelled)))
    if destination and _has_value(destination.get("address")):
        rows.append(
            ("Destination", _step_value_cell(destination, cancelled=cancelled, max_addr=_MAX_DESTINATION))
        )
    if ret and _has_value(ret.get("address")):
        rows.append(("Retour", _step_value_cell(ret, cancelled=cancelled)))
    if not rows:
        return _build_route_simple(ctx)
    return [_route_table(rows)]


def _build_route_multistop(ctx: MissionReportContext) -> list[Any]:
    """Timeline verticale numérotée Étape N pour multi-destination."""
    cancelled = _is_mission_cancelled(ctx)
    steps = ctx.route_steps
    if not steps:
        return [Paragraph(MISSING, _styles()["body"])]

    st = _styles()
    rows: list[list[Any]] = []
    for idx, step in enumerate(steps):
        step_num = idx + 1
        kind = step.get("kind", "")
        title = step.get("title", "")
        if kind == "departure":
            label = f"Étape {step_num} — Départ"
        elif kind == "return":
            label = f"Étape {step_num} — Retour institution"
        else:
            label = f"Étape {step_num} — {_short_step_title(str(title))}"

        addr_max = _MAX_ADDRESS if kind in {"departure", "return"} else _MAX_DESTINATION
        addr = _truncate_field(str(step.get("address", MISSING)), max_len=addr_max)
        left = [Paragraph("●", st["railBullet"])]
        right: list[Any] = [
            Paragraph(label, st["railTitle"]),
            Paragraph(addr or MISSING, st["railBody"]),
        ]
        time_line = _step_time_line(step, cancelled=cancelled)
        if time_line:
            right.append(Paragraph(time_line, st["railBody"]))
        rows.append([left, right])

    table = Table(rows, colWidths=[0.8 * cm, 16.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [table]


def _select_route_builder(ctx: MissionReportContext) -> Callable[[MissionReportContext], list[Any]]:
    trip_type = ctx.request_classification.get("trip_type", "one_way")
    if trip_type == "multi_stop":
        return _build_route_multistop
    if trip_type == "round_trip":
        return _build_route_round_trip
    return _build_route_simple


def _voucher_transport_table(rows: list[tuple[str, list[Any]]]) -> Table:
    """Tableau TRANSPORT du bon : colonne label élargie (libellés sur une ligne)."""
    st = _styles()
    data: list[list[Any]] = [
        [Paragraph(f"<b>{label}</b>", st["body"]), value] for label, value in rows
    ]
    table = Table(data, colWidths=[4.6 * cm, 12.4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _voucher_address_cell(
    step: dict[str, Any], *, max_addr: int = _MAX_ADDRESS, show_planned: bool = False
) -> list[Any]:
    """Cellule adresse bon de transport (sans horaires réels)."""
    st = _styles()
    addr = _truncate_field(str(step.get("address", MISSING)), max_len=max_addr)
    cell: list[Any] = [Paragraph(addr or MISSING, st["body"])]
    if show_planned:
        planned = step.get("planned_time", MISSING)
        if _has_value(planned):
            cell.append(Paragraph(f"Prévu : {planned}", st["small"]))
    return cell


def _build_voucher_transport(ctx: MissionReportContext) -> list[Any]:
    """Section TRANSPORT : type, horaire, trajet adaptatif (simple / A-R / multi)."""
    st = _styles()
    steps = ctx.route_steps
    classification = ctx.request_classification
    trip_type = classification.get("trip_type", "one_way")
    departure = next((s for s in steps if s.get("kind") == "departure"), None)
    destination = next((s for s in steps if s.get("kind") == "destination"), None)
    if destination is None:
        destination = next((s for s in steps if s.get("kind") == "dropoff"), None)
    ret = next((s for s in steps if s.get("kind") == "return"), None)

    rows: list[tuple[str, list[Any]]] = []

    if trip_type != "multi_stop" and departure and _has_value(departure.get("planned_time")):
        # scheduled_time_type = "arrival" → l'heure est un rendez-vous, pas une prise en charge
        is_appointment = classification.get("scheduled_time_type") == "arrival"
        time_label = "Rendez-vous" if is_appointment else "Prise en charge"
        rows.append(
            (time_label, [Paragraph(str(departure.get("planned_time")), st["body"])])
        )

    if trip_type == "multi_stop":
        for idx, step in enumerate(steps):
            step_num = idx + 1
            kind = step.get("kind", "")
            title = step.get("title", "")
            if kind == "departure":
                label = f"Étape {step_num} — Départ"
            elif kind == "return":
                label = f"Étape {step_num} — Retour institution"
            else:
                label = f"Étape {step_num} — {_short_step_title(str(title))}"
            addr_max = (
                _MAX_ADDRESS if kind in {"departure", "return"} else _MAX_VOUCHER_DESTINATION
            )
            rows.append(
                (label, _voucher_address_cell(step, max_addr=addr_max, show_planned=True))
            )
    elif trip_type == "round_trip":
        if departure and _has_value(departure.get("address")):
            rows.append(("Départ", _voucher_address_cell(departure)))
        if destination and _has_value(destination.get("address")):
            rows.append(
                ("Destination", _voucher_address_cell(destination, max_addr=_MAX_VOUCHER_DESTINATION))
            )
        if ret and _has_value(ret.get("address")):
            rows.append(("Retour", _voucher_address_cell(ret)))
    else:
        if departure and _has_value(departure.get("address")):
            rows.append(("Départ", _voucher_address_cell(departure)))
        if destination and _has_value(destination.get("address")):
            rows.append(
                ("Destination", _voucher_address_cell(destination, max_addr=_MAX_VOUCHER_DESTINATION))
            )
        elif len(steps) >= 2 and _has_value(steps[-1].get("address")):
            rows.append(
                ("Destination", _voucher_address_cell(steps[-1], max_addr=_MAX_VOUCHER_DESTINATION))
            )

    if not rows:
        return [Paragraph(MISSING, st["body"])]
    return [_voucher_transport_table(rows)]


_MEDICAL_LINE_EMPTY = frozenset({MISSING, "Non", "non", "False", "false", "0", ""})


def _medical_line_has_value(val: Any) -> bool:
    """Valeur structurée affichable (ignore Non, tirets, vides)."""
    s = str(val or "").strip()
    return bool(s) and s not in _MEDICAL_LINE_EMPTY


def _format_voucher_need_bullet(label: str, value: str) -> str:
    """Puce besoin : libellé seul si Oui, sinon « label : valeur »."""
    val = str(value).strip()
    if val == "Oui":
        return f"• {label}"
    return f"• {label} : {val}"


def _build_voucher_needs_alert(ctx: MissionReportContext) -> list[Any]:
    """Bloc encadré « BESOINS PARTICULIERS » (bon de transport, avant le trajet)."""
    medical = ctx.medical_block
    if not _has_medical_content(medical):
        return []
    st = _styles()
    inner: list[Any] = [Paragraph("■ BESOINS PARTICULIERS", st["sectionTitle"])]
    for line in medical.get("lines", []):
        val = str(line.get("value", "")).strip()
        if _medical_line_has_value(val):
            label = str(line.get("label", "")).strip()
            if label:
                inner.append(
                    Paragraph(_format_voucher_need_bullet(label, val), st["body"])
                )
    if _has_value(medical.get("floor_elevator_info")):
        inner.append(
            Paragraph(
                f"• Accès / étage : {medical.get('floor_elevator_info')}",
                st["body"],
            )
        )
    if _has_value(medical.get("notes")):
        notes = _truncate_medical_notes(str(medical.get("notes")), max_len=_MAX_VOUCHER_NOTES)
        inner.append(Paragraph("Remarque :", st["body"]))
        inner.append(Paragraph(notes, st["noteItalic"]))
    box = Table([[inner]], colWidths=[17 * cm])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return [box, Spacer(1, 0.2 * cm)]


def _apply_history_time_format(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heure affichée seulement si multi-jours ou écart > 60 min."""
    dts = [row["at"] for row in rows if isinstance(row.get("at"), datetime)]
    if not dts:
        return rows
    dts_sorted = sorted(dts)
    multi_day = dts_sorted[0].date() != dts_sorted[-1].date()
    span_minutes = (dts_sorted[-1] - dts_sorted[0]).total_seconds() / 60
    show_time = multi_day or span_minutes > 60
    fmt = "%d.%m.%Y %H:%M" if show_time else "%d.%m.%Y"
    for row in rows:
        dt = row.get("at")
        if isinstance(dt, datetime):
            row["date"] = dt.strftime(fmt)
    return rows


def _synthetic_history_table(rows: list[dict[str, Any]]) -> Table:
    """Historique synthétique compact (date | libellé)."""
    st = _styles()
    formatted = _apply_history_time_format(list(rows))
    data: list[list[Any]] = []
    for row in formatted:
        data.append(
            [
                Paragraph(row.get("date", MISSING), st["small"]),
                Paragraph(row.get("label", MISSING), st["body"]),
            ]
        )
    if not data:
        data = [[Paragraph(MISSING, st["small"]), Paragraph("Aucun événement", st["body"])]]
    table = Table(data, colWidths=[3.5 * cm, 13.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _signature_cell(title: str, *, time_field: bool = False) -> list[Any]:
    """Contenu d'une signature : titre + heure réelle/date + ligne de signature."""
    st = _styles()
    first_line = (
        "Heure réelle : ______________________"
        if time_field
        else "Date : ______________________"
    )
    return [
        Paragraph(title, st["cardValueBold"]),
        Spacer(1, 0.35 * cm),
        Paragraph(first_line, st["body"]),
        Spacer(1, 0.2 * cm),
        Paragraph("Signature : ______________________", st["body"]),
    ]


def _signatures_row(
    left_title: str, right_title: str, *, left_time_field: bool = False
) -> Table:
    """Deux signatures côte à côte (gain vertical sur le bon de transport)."""
    block = Table(
        [
            [
                _signature_cell(left_title, time_field=left_time_field),
                _signature_cell(right_title),
            ]
        ],
        colWidths=[8 * cm, 8 * cm],
    )
    block.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, 0), 0.5, BORDER),
                ("BOX", (1, 0), (1, 0), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return block


def _build_qr_image(url: str, size: float = 1.5 * cm) -> Image | None:
    """QR code vers verify_url ; None si échec (fallback texte)."""
    try:
        import qrcode

        qr = qrcode.QRCode(box_size=4, border=1)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Image(buf, width=size, height=size)
    except Exception:
        return None


def _has_medical_content(medical: dict[str, Any]) -> bool:
    """Section besoins médicaux visible uniquement si contenu réel."""
    for line in medical.get("lines", []):
        if _medical_line_has_value(line.get("value")):
            return True
    if _has_value(medical.get("notes")):
        return True
    if _has_value(medical.get("floor_elevator_info")):
        return True
    return False


def _billing_summary_line(billing: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in ("billed_to", "amount", "invoice_status"):
        val = billing.get(key)
        if _has_value(val):
            parts.append(str(val))
    return " · ".join(parts) if parts else None


def _audit_execution_block(ctx: MissionReportContext) -> list[Any]:
    """Bloc compact mode d'exécution (rapport audit uniquement)."""
    st = _styles()
    carrier = ctx.carrier_block
    lines: list[Any] = []
    mode = carrier.get("execution_mode_label")
    if _has_value(mode):
        lines.append(Paragraph(f"<b>Exécution :</b> {mode}", st["body"]))
    if _has_value(carrier.get("name")):
        lines.append(Paragraph(f"<b>Transporteur :</b> {carrier.get('name')}", st["body"]))
    if _has_value(carrier.get("reference")):
        lines.append(
            Paragraph(f"<b>Référence externe :</b> {carrier.get('reference')}", st["body"])
        )
    if _has_value(carrier.get("externalization_reason")):
        lines.append(
            Paragraph(
                f"<b>Raison d'externalisation :</b> {carrier.get('externalization_reason')}",
                st["body"],
            )
        )
    if carrier.get("is_external") and _has_value(carrier.get("declared_at")):
        lines.append(
            Paragraph(
                "<b>Déclarée réalisée par l'institution</b>",
                st["body"],
            )
        )
        if _has_value(carrier.get("declared_by")):
            lines.append(
                Paragraph(f"<b>Déclarée par :</b> {carrier.get('declared_by')}", st["body"])
            )
        lines.append(Paragraph(f"<b>Date :</b> {carrier.get('declared_at')}", st["body"]))
    if lines:
        lines.append(Spacer(1, 0.08 * cm))
    return lines


def _administrative_block(ctx: MissionReportContext) -> list[Any]:
    """Bloc final : colonne dense facturation/traçabilité + QR à droite."""
    st = _styles()
    billing = ctx.billing_block
    trace = ctx.traceability
    verify_url = trace.get("verify_url") or "https://www.lirie.ch"

    left_lines: list[Any] = []
    billing_line = _billing_summary_line(billing)
    if billing_line:
        left_lines.append(Paragraph(f"<b>Facturation :</b> {billing_line}", st["body"]))

    left_lines.append(Paragraph(f"<b>Réf. archivage :</b> {ctx.reference}", st["body"]))

    doc_hash = trace.get("document_hash")
    if _has_value(doc_hash):
        left_lines.append(Paragraph(f"<b>Empreinte :</b> {doc_hash}", st["body"]))

    public_id = trace.get("public_id")
    if _has_value(public_id):
        left_lines.append(Paragraph(f"<b>Identifiant :</b> {public_id}", st["body"]))

    ref_parts: list[str] = []
    if _has_value(ctx.request_number):
        ref_parts.append(f"Demande {ctx.request_number}")
    if _has_value(ctx.booking_number):
        ref_parts.append(f"Réservation {ctx.booking_number}")
    if ref_parts:
        left_lines.append(Paragraph(" · ".join(ref_parts), st["small"]))

    qr = _build_qr_image(verify_url)
    qr_col: list[Any] = []
    if qr is not None:
        qr_col.append(qr)
    else:
        qr_col.append(Paragraph(verify_url, st["small"]))

    inner = Table([[left_lines, qr_col]], colWidths=[12.5 * cm, 4 * cm])
    inner.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    box = Table([[inner]], colWidths=[17 * cm])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return [box]


def _page_footer(canvas: Any, doc: Any, ctx: MissionReportContext) -> None:
    """Pied de page émetteur LIRIE discret (une ligne à gauche, numéro à droite)."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.5 * cm, 1 * cm, _FOOTER_TEXT)
    canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _build_pdf(ctx: MissionReportContext, story: list[Any]) -> bytes:
    buffer = BytesIO()

    def on_page(canvas: Any, doc: Any) -> None:
        _page_footer(canvas, doc, ctx)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()


def _voucher_driver_step_label(kind: str, title: str) -> str:
    """Libellé terrain — jamais « Étape N » ni « Destination 2 »."""
    if kind == "departure":
        return "Départ"
    if kind == "return":
        return "Retour institution"
    raw = str(title).strip()
    if raw.startswith("Destination ") and raw != "Destination":
        return "Destination"
    short = _short_step_title(raw)
    if short and not short.startswith("Destination "):
        return short
    return "Destination"


def _build_voucher_needs_bullets(medical: dict[str, Any]) -> tuple[str, ...]:
    bullets: list[str] = []
    for line in medical.get("lines", []):
        val = str(line.get("value", "")).strip()
        if _medical_line_has_value(val):
            label = str(line.get("label", "")).strip()
            if label:
                bullets.append(_format_voucher_need_bullet(label, val))
    if _has_value(medical.get("floor_elevator_info")):
        bullets.append(f"• Accès / étage : {medical.get('floor_elevator_info')}")
    return tuple(bullets)


def _build_voucher_presentation(ctx: MissionReportContext) -> VoucherPresentation:
    """Normalise le contexte métier en présentation chauffeur (unique)."""
    patient = ctx.patient_block
    inst = ctx.institution_snapshot
    carrier = ctx.carrier_block
    medical = ctx.medical_block
    classification = ctx.request_classification

    patient_name = _truncate_field(
        str(patient.get("full_name") or MISSING), max_len=_MAX_PATIENT
    ) or MISSING
    dob = patient.get("dob")
    patient_dob = str(dob) if _has_value(dob) else None

    bills_patient = classification.get("billing_target") == "patient"
    patient_address: str | None = None
    billing_label: str | None = None
    if bills_patient:
        billing_label = "Patient"
        if _has_value(patient.get("address")):
            patient_address = _truncate_field(str(patient.get("address")), max_len=_MAX_ADDRESS)

    type_label = _VOUCHER_TRANSPORT_TYPE_LABELS.get(classification.get("mobility_level"))

    route_stops: list[RouteStop] = []
    for step in ctx.route_steps:
        kind = str(step.get("kind", ""))
        title = str(step.get("title", ""))
        addr = _truncate_field(str(step.get("address", MISSING)), max_len=_MAX_ADDRESS) or MISSING
        planned = step.get("planned_time")
        planned_time = str(planned) if _has_value(planned) else None
        route_stops.append(
            RouteStop(
                kind=kind,
                label=_voucher_driver_step_label(kind, title),
                address=addr,
                planned_time=planned_time,
            )
        )

    primary_time: str | None = None
    time_context_label: str | None = None
    departure = next((s for s in ctx.route_steps if s.get("kind") == "departure"), None)
    if departure and _has_value(departure.get("planned_time")):
        primary_time = str(departure.get("planned_time"))
        is_appointment = classification.get("scheduled_time_type") == "arrival"
        time_context_label = "Rendez-vous" if is_appointment else "Prise en charge"

    needs_remark: str | None = None
    if _has_value(medical.get("notes")):
        needs_remark = _truncate_medical_notes(str(medical.get("notes")), max_len=_MAX_VOUCHER_NOTES)

    carrier_name = _truncate_field(carrier.get("name"), max_len=_MAX_CARRIER)

    return VoucherPresentation(
        patient_name=patient_name,
        patient_dob=patient_dob,
        route_stops=tuple(route_stops),
        primary_time=primary_time,
        needs_bullets=_build_voucher_needs_bullets(medical),
        needs_remark=needs_remark,
        needs_floor_info=str(medical.get("floor_elevator_info"))
        if _has_value(medical.get("floor_elevator_info"))
        else None,
        institution_name=_truncate_field(inst.get("name"), max_len=_MAX_INSTITUTION),
        contact_line=_truncate_field(_build_voucher_contact_line(inst)),
        transport_type_label=type_label,
        reference=ctx.reference,
        mission_date=str(ctx.mission_info.get("mission_date"))
        if _has_value(ctx.mission_info.get("mission_date"))
        else None,
        carrier_name=carrier_name,
        carrier_is_external=bool(carrier.get("is_external")),
        driver_name=str(carrier.get("driver_name"))
        if _has_value(carrier.get("driver_name"))
        else None,
        patient_address=patient_address,
        billing_label=billing_label,
        time_context_label=time_context_label,
        has_needs=_has_medical_content(medical),
        verify_url=str(ctx.traceability.get("verify_url") or "https://www.lirie.ch"),
        logo_url=ctx.institution_snapshot.get("logo_url"),
    )


def _voucher_operational_header(pres: VoucherPresentation) -> list[Any]:
    """En-tête bon : titre + référence L3 + logo (inchangé entre maquettes)."""
    st = _styles()
    ref_line = pres.reference
    if _has_value(pres.mission_date):
        ref_line = f"{pres.reference} · {pres.mission_date}"
    left_lines: list[Any] = [
        Paragraph("BON DE TRANSPORT", st["docTitle"]),
        Paragraph(ref_line, st["metaMuted"]),
    ]
    return [
        _voucher_header_table(left_lines, pres.verify_url, logo_url=pres.logo_url),
        Spacer(1, 0.25 * cm),
    ]


def _voucher_hero_block(
    pres: VoucherPresentation, *, hero_style: Literal["inline", "split"]
) -> list[Any]:
    st = _styles()
    flow: list[Any] = []
    if hero_style == "inline":
        if pres.patient_dob:
            line = f"{pres.patient_name} ({pres.patient_dob})"
        else:
            line = pres.patient_name
        flow.append(Paragraph(line, st["heroName"]))
    else:
        flow.append(Paragraph(pres.patient_name, st["heroName"]))
        if pres.patient_dob:
            flow.append(Paragraph(pres.patient_dob, st["heroDob"]))
    if _has_value(pres.institution_name):
        flow.append(Paragraph(str(pres.institution_name), st["heroInstitution"]))
    if _has_value(pres.transport_type_label):
        flow.append(Paragraph(str(pres.transport_type_label), st["metaMuted"]))
    if _has_value(pres.patient_address):
        flow.append(
            Paragraph(f"Adresse patient : {pres.patient_address}", st["small"])
        )
    if _has_value(pres.billing_label):
        flow.append(Paragraph(f"Facturation : {pres.billing_label}", st["small"]))
    flow.append(Spacer(1, 0.15 * cm))
    return flow


def _voucher_needs_box(
    pres: VoucherPresentation,
    *,
    title: str = "ATTENTION — BESOINS PARTICULIERS",
    accent: bool = False,
    include_transport_type: bool = False,
) -> list[Any]:
    if not pres.has_needs:
        return []
    st = _styles()
    inner: list[Any] = [Paragraph(title, st["sectionTitle"])]
    if include_transport_type and _has_value(pres.transport_type_label):
        inner.append(Paragraph(f"• Type : {pres.transport_type_label}", st["body"]))
    for bullet in pres.needs_bullets:
        inner.append(Paragraph(bullet, st["body"]))
    if _has_value(pres.needs_remark):
        inner.append(Paragraph("Remarque :", st["body"]))
        inner.append(Paragraph(str(pres.needs_remark), st["noteItalic"]))
    box_style = [
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if accent:
        box_style.extend(
            [
                ("LINELEFT", (0, 0), (0, -1), 3, ACCENT),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f7f6")),
            ]
        )
    box = Table([[inner]], colWidths=[17 * cm])
    box.setStyle(TableStyle(box_style))
    return [box, Spacer(1, 0.15 * cm)]


def _voucher_route_vertical(pres: VoucherPresentation) -> list[Any]:
    """Trajet Operational : heure L1 avant adresse, connecteur ↓ entre étapes."""
    st = _styles()
    flow: list[Any] = [Paragraph("TRAJET", st["sectionLabel"])]
    if _has_value(pres.time_context_label) and _has_value(pres.primary_time):
        flow.append(Paragraph(str(pres.time_context_label), st["small"]))

    stops = pres.route_stops
    if not stops:
        flow.append(Paragraph(MISSING, st["body"]))
        return flow

    for idx, stop in enumerate(stops):
        if idx > 0:
            flow.append(Paragraph("↓", st["routeArrow"]))
        if _has_value(stop.planned_time):
            flow.append(Paragraph(str(stop.planned_time), st["timeProminent"]))
        if stop.kind == "return" and not _has_value(stop.address):
            flow.append(Paragraph(stop.label, st["body"]))
            continue
        max_len = (
            _MAX_ADDRESS if stop.kind in {"departure", "return"} else _MAX_VOUCHER_DESTINATION
        )
        addr = _truncate_field(stop.address, max_len=max_len) or MISSING
        if stop.kind == "return":
            flow.append(Paragraph(addr, st["body"]))
            flow.append(Paragraph(stop.label, st["small"]))
        else:
            if stop.label not in {"Départ", "Destination"}:
                flow.append(Paragraph(f"<b>{stop.label}</b>", st["body"]))
            flow.append(Paragraph(addr, st["body"]))
    flow.append(Spacer(1, 0.12 * cm))
    return flow


def _voucher_route_inline(pres: VoucherPresentation) -> list[Any]:
    """Trajet Ultra compact : « heure — adresse » par ligne."""
    st = _styles()
    flow: list[Any] = [Paragraph("TRAJET", st["sectionLabel"])]
    if not pres.route_stops:
        flow.append(Paragraph(MISSING, st["body"]))
        return flow
    for stop in pres.route_stops:
        if stop.kind == "return" and not _has_value(stop.address):
            flow.append(Paragraph(stop.label, st["body"]))
            continue
        addr = _truncate_field(stop.address, max_len=_MAX_ADDRESS) or MISSING
        if _has_value(stop.planned_time):
            line = f"{stop.planned_time} — {addr}"
        else:
            line = addr if stop.kind != "return" else f"{stop.label} — {addr}"
        flow.append(Paragraph(line, st["body"]))
    flow.append(Spacer(1, 0.1 * cm))
    return flow


# Au-delà de cette longueur, la ligne contact | transporteur passe sur deux lignes.
_VOUCHER_META_ONELINE_MAX = 90


def _voucher_meta_footer(pres: VoucherPresentation) -> list[Any]:
    """Pied L3 : contact | transporteur · chauffeur (1 ligne, sinon 2)."""
    st = _styles()
    left = str(pres.contact_line) if _has_value(pres.contact_line) else ""
    right_parts: list[str] = []
    if _has_value(pres.carrier_name):
        prefix = "Transporteur externe" if pres.carrier_is_external else "Transporteur"
        right_parts.append(f"{prefix} : {pres.carrier_name}")
    elif pres.carrier_is_external:
        right_parts.append("Transporteur externe")
    if _has_value(pres.driver_name):
        right_parts.append(f"Chauffeur : {pres.driver_name}")
    right = " · ".join(right_parts)

    if not left and not right:
        return []
    if left and right:
        one_line = f"{left}    |    {right}"
        if len(left) + len(right) + 5 <= _VOUCHER_META_ONELINE_MAX:
            return [Paragraph(one_line, st["metaMuted"]), Spacer(1, 0.12 * cm)]
        return [
            Paragraph(left, st["metaMuted"]),
            Paragraph(right, st["metaMuted"]),
            Spacer(1, 0.12 * cm),
        ]
    return [Paragraph(left or right, st["metaMuted"]), Spacer(1, 0.12 * cm)]


def _signature_cell_compact(
    title: str, *, time_field: bool = False, minimal: bool = False
) -> list[Any]:
    st = _styles()
    first_line = (
        "Heure réelle : ______________________"
        if time_field
        else "Date : ______________________"
    )
    sp_top = 0.15 * cm if minimal else 0.35 * cm
    sp_mid = 0.1 * cm if minimal else 0.2 * cm
    return [
        Paragraph(title, st["cardValueBold"]),
        Spacer(1, sp_top),
        Paragraph(first_line, st["body"]),
        Spacer(1, sp_mid),
        Paragraph("Signature : ______________________", st["body"]),
    ]


def _signatures_stack(*, minimal: bool = False) -> list[Any]:
    """Signatures en colonne unique, sans encadré (PDF-UX-01)."""
    st = _styles()
    inner: list[Any] = [
        Paragraph("Signatures", st["sectionLabel"]),
        *_signature_cell_compact("Chauffeur", time_field=True, minimal=minimal),
        Spacer(1, 0.08 * cm),
        *_signature_cell_compact("Patient ou représentant", minimal=minimal),
    ]
    return [KeepTogether(inner)]


def _signatures_row_compact(*, minimal: bool = False) -> Table:
    """Signatures côte à côte, sans encadré (PDF-UX-01)."""
    pad = 6 if minimal else 10
    block = Table(
        [
            [
                _signature_cell_compact("Chauffeur", time_field=True, minimal=minimal),
                _signature_cell_compact("Patient ou représentant", minimal=minimal),
            ]
        ],
        colWidths=[8 * cm, 8 * cm],
    )
    block.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), pad),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return block


def _voucher_confirmation_inline() -> list[Any]:
    """Confirmation finale ultra compacte (une ligne, hauteur minimale)."""
    st = _styles()
    line = (
        "Confirmation : Chauffeur __________________  "
        "Patient/représentant __________________"
    )
    return [Spacer(1, 0.45 * cm), Paragraph(line, st["body"])]


def _voucher_signatures(options: VoucherLayoutOptions, *, minimal: bool = False) -> list[Any]:
    flow: list[Any] = []
    if options.signature_style == "confirmation_inline":
        flow.extend(_voucher_confirmation_inline())
    elif options.signature_style == "stack":
        flow.extend(_signatures_stack(minimal=minimal))
    else:
        flow.extend(_section_title("Signatures"))
        flow.append(_signatures_row_compact(minimal=minimal))
    return flow


def _layout_voucher_legacy(ctx: MissionReportContext) -> list[Any]:
    """Baseline actuelle — table clé/valeur (production API jusqu'à bascule)."""
    story = _voucher_header(ctx)
    story.extend(_build_voucher_identity_table(ctx))
    story.extend(_build_voucher_needs_alert(ctx))
    story.extend(_section_title("Transport"))
    story.extend(_build_voucher_transport(ctx))
    story.append(Spacer(1, 0.3 * cm))
    story.extend(_section_title("Signatures"))
    story.append(
        _signatures_row("Chauffeur", "Patient ou représentant", left_time_field=True)
    )
    return story


def _layout_voucher_operational(
    pres: VoucherPresentation, options: VoucherLayoutOptions
) -> list[Any]:
    """Document terrain chauffeur — favori production (~80%)."""
    story = _voucher_operational_header(pres)
    story.extend(_voucher_hero_block(pres, hero_style=options.hero_style))
    story.extend(_voucher_needs_box(pres))
    story.extend(_voucher_route_vertical(pres))
    story.extend(_voucher_meta_footer(pres))
    story.extend(_voucher_signatures(options))
    return story


def _layout_voucher_ultra_compact(pres: VoucherPresentation) -> list[Any]:
    """Variante A — densité maximale (REVIEW_ONLY)."""
    st = _styles()
    ref_line = f"{pres.reference} · {pres.mission_date or MISSING}"
    left_lines: list[Any] = [
        Paragraph("BON DE TRANSPORT", st["docTitle"]),
        Paragraph(ref_line, st["metaMuted"]),
    ]
    story: list[Any] = [
        _voucher_header_table(left_lines, pres.verify_url, logo_url=pres.logo_url),
        Spacer(1, 0.15 * cm),
        Paragraph("PATIENT", st["sectionLabel"]),
    ]
    if pres.patient_dob:
        story.append(
            Paragraph(f"{pres.patient_name} ({pres.patient_dob})", st["heroName"])
        )
    else:
        story.append(Paragraph(pres.patient_name, st["heroName"]))
    if pres.has_needs:
        story.append(Spacer(1, 0.08 * cm))
        story.append(Paragraph("BESOINS", st["sectionLabel"]))
        for bullet in pres.needs_bullets:
            story.append(Paragraph(bullet, st["body"]))
        if _has_value(pres.needs_remark):
            story.append(Paragraph(f"Remarque : {pres.needs_remark}", st["noteItalic"]))
    story.extend(_voucher_route_inline(pres))
    if _has_value(pres.contact_line):
        story.append(Paragraph("CONTACT", st["sectionLabel"]))
        story.append(Paragraph(str(pres.contact_line), st["body"]))
    meta_bits = [pres.reference]
    if _has_value(pres.carrier_name):
        meta_bits.append(str(pres.carrier_name))
    story.append(Paragraph(" · ".join(meta_bits), st["metaMuted"]))
    story.append(Spacer(1, 0.1 * cm))
    story.extend(
        _voucher_signatures(VoucherLayoutOptions(signature_style="row"), minimal=True)
    )
    return story


def _layout_voucher_medical(
    pres: VoucherPresentation, options: VoucherLayoutOptions
) -> list[Any]:
    """Variante C — besoins renforcés (REVIEW_ONLY)."""
    story = _voucher_operational_header(pres)
    story.extend(_voucher_hero_block(pres, hero_style=options.hero_style))
    story.extend(
        _voucher_needs_box(
            pres,
            title="BESOINS PARTICULIERS — ATTENTION",
            accent=True,
            include_transport_type=True,
        )
    )
    story.extend(_voucher_route_vertical(pres))
    story.extend(_voucher_meta_footer(pres))
    story.extend(_voucher_signatures(options))
    return story


def build_operational_voucher_pdf(
    ctx: MissionReportContext,
    layout: VoucherLayoutName = "legacy",
    options: VoucherLayoutOptions | None = None,
) -> bytes:
    """Bon de transport — layouts legacy (API) ou maquettes UX chauffeur."""
    opts = options or VoucherLayoutOptions()
    if layout == "legacy":
        story = _layout_voucher_legacy(ctx)
    elif layout == "operational":
        pres = _build_voucher_presentation(ctx)
        story = _layout_voucher_operational(pres, opts)
    elif layout == "ultra_compact":
        pres = _build_voucher_presentation(ctx)
        story = _layout_voucher_ultra_compact(pres)
    elif layout == "medical":
        pres = _build_voucher_presentation(ctx)
        story = _layout_voucher_medical(pres, opts)
    else:
        raise ValueError(f"layout invalide : {layout!r}")
    return _build_pdf(ctx, story)


def build_mission_audit_report_pdf(ctx: MissionReportContext) -> bytes:
    """Rapport de mission — document de preuve d'exécution compact."""
    medical = ctx.medical_block

    story = _document_header(ctx, "Rapport de mission")
    story.extend(_build_report_identity_table(ctx))

    story.extend(_section_title("Déroulement du transport", with_rule=False))
    route_builder = _select_route_builder(ctx)
    story.extend(route_builder(ctx))
    if ctx.route_legs_truncated:
        story.append(Paragraph("Parcours tronqué (20 étapes max.)", _styles()["small"]))
    story.append(Spacer(1, 0.12 * cm))

    story.extend(_section_title("Historique", with_rule=False))
    story.append(_synthetic_history_table(ctx.synthetic_history))
    story.append(Spacer(1, 0.12 * cm))

    story.extend(_audit_execution_block(ctx))

    if _has_medical_content(medical):
        story.extend(_section_title("Besoins médicaux", with_rule=False))
        med_lines: list[str] = []
        for line in medical.get("lines", []):
            val = str(line.get("value", "")).strip()
            if val and val not in {MISSING, "Non"}:
                med_lines.append(f"{line.get('label', '')} : {val}")
        if _has_value(medical.get("floor_elevator_info")):
            med_lines.append(f"Accès / étage : {medical.get('floor_elevator_info')}")
        for line in med_lines:
            story.append(Paragraph(line, _styles()["body"]))
        # Commentaires libres : tronqués (les infos critiques ci-dessus restent entières)
        if _has_value(medical.get("notes")):
            notes = _truncate_medical_notes(str(medical.get("notes")))
            story.append(Paragraph(f"Remarques : {notes}", _styles()["body"]))
        story.append(Spacer(1, 0.1 * cm))

    if ctx.attachments:
        story.extend(_section_title("Pièces jointes"))
        for att in ctx.attachments:
            story.append(Paragraph(str(att.get("label", "")), _styles()["body"]))
        story.append(Spacer(1, 0.1 * cm))

    story.append(
        KeepTogether(
            _section_title("Informations administratives") + _administrative_block(ctx)
        )
    )

    return _build_pdf(ctx, story)
