import contextlib
import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from html import escape as _html_escape_minimal
from typing_extensions import override

from flask import current_app
from reportlab.platypus import Flowable
from sqlalchemy.orm import joinedload, selectinload

from infrastructure.invoices.invoice_calculator import round_to_5_cents
from models import Client, CompanyBillingSettings, Invoice, InvoiceLine, InvoiceLineType
from services.documents.invoice_template_builder import InvoiceTemplateBuilder
from services.documents.qrbill import QRBillService

LEVEL_ONE = 1
LEVEL_THRESHOLD = 2
MAX_PATIENT_NAME_LENGTH = 18
MONTHS_PER_YEAR = 12

app_logger = logging.getLogger("pdf_service")

# Polices DejaVu : enregistrement une seule fois par processus (évite coût répété à chaque PDF)
_DEJAVU_PDF_FONTS_READY: bool = False

# ✅ Monitoring performance PDF
TEMPLATE_VERSION = "unified_v1"
PERF_WARNING_ROWS_THRESHOLD = 40
PERF_WARNING_MS_THRESHOLD = 1500

# --- Zone destinataire compatible enveloppe C5 à fenêtre ---
# Standard pratique: bloc adresse ~45mm haut, ~20mm du haut, ~20mm du bord gauche.
# Ajuster X/Y si besoin (ex: DEST_ADDR_X_MM +6 à +12 pour décalage droite).
# 85mm par défaut (fenêtres C5 parfois étroites) — augmenter si validation physique.
# 18mm = décalage -7mm vs 25mm pour visibilité complète dans fenêtre C5 (ne pas toucher au reste).
DEST_ADDR_X_MM = 18.0  # Position X depuis bord gauche (mm) — zone fenêtre C5
DEST_ADDR_Y_MM = (
    20.0  # Position Y depuis bord haut (mm) — converti en canvas (origine bas)
)
DEST_ADDR_MAX_WIDTH_MM = 85.0  # Largeur max wrapping (zone fenêtre C5, safe par défaut)
DEST_ADDR_LINE_HEIGHT_MM = 4.0  # Interligne (mm)
DEST_ADDR_ZONE_HEIGHT_MM = 45.0  # Hauteur zone fenêtre C5

# Espacement pour pousser le QR-Bill en bas de sa page (A4 - margins - QR height)
# Formule: spacer_max = usable_height - QR_height - overhead
#   marge bas 0.5cm: usable=771pt, QR+overhead~292pt → spacer_safe≈475
# Source unique : utilisé par pdf.py et partnerships/invoices_pdf.py
QR_BILL_SPACER_PT = 478

# Dimensions du QR-Bill (12x6 cm standard suisse) — source unique pour tous les PDF
QR_BILL_WIDTH_CM = 12.0
QR_BILL_HEIGHT_CM = 6.0
QR_BILL_TABLE_COL_WIDTHS_CM = (6.0, 12.0)  # (colonne QR, colonne vide)

# Zoom : facteur d'échelle (1.0 = 100%, 1.1 = agrandir 10%, 0.9 = rétrécir 10%)
# 1.57 = compromis pour spacer 435 (1.6 débordait à 435)
QR_BILL_SCALE_FACTOR = 1.6

# Positionnement horizontal : décalage gauche/droite (mm)
# > 0 = décaler vers la droite, < 0 = décaler vers la gauche
QR_BILL_LEFT_PADDING_MM = -5.0

# Marge bas page QR-Bill (pas de pied de page légal) — QR-Bill au maximum en bas
# 0.5 cm = spacer 475 avec scale 1.6 (maximum descente sans débordement)
QR_BILL_PAGE_BOTTOM_MARGIN_CM = 0.5

# Pied « totaux » : même gabarit que InvoiceLivePreview `.totals` { max-width: 280px } (~74 mm).
INVOICE_PREVIEW_TOTALS_LABEL_CM = 4.25
INVOICE_PREVIEW_TOTALS_AMOUNT_CM = 3.15

# --- Grille typo facture PDF (miroir InvoiceLivePreview.module.css, valeurs pt) ---
FONT_HEADER_COMPANY = 14
FONT_CLIENT_NAME = 12
FONT_BODY = 10
FONT_META_NUMBER = 11
FONT_META_DATES = 10
FONT_TABLE_HEADER = 10
FONT_SECONDARY = 8
FONT_TOTAL = 12
FONT_COMPANY_CONTACT = 9
COLOR_TEXT_PDF = "#000000"
COLOR_MUTED_PDF = "#64748b"

# Marges horizontales page facture A4 (contenu = frames ReportLab)
INVOICE_PAGE_LEFT_MARGIN_CM = 1.9
INVOICE_PAGE_RIGHT_MARGIN_CM = 1.9

# Pied de page : mention plateforme (sous le trait, au bas de la marge)
FOOTER_PLATFORM_TAGLINE = (
    "Facturation et gestion des prestations via LIRIE — "
    "solution digitale dédiée au transport médical · www.lirie.ch"
)


def _make_invoice_doc_with_qrbill_page(
    buffer: Any,
    top_margin_cm: float,
    bottom_margin_cm: float,
    left_margin_cm: float,
    right_margin_cm: float,
    on_first_page: Any,
    on_later_pages: Any = None,
) -> Any:
    """Crée un DocTemplate avec une page QR-Bill dédiée (marge bas 0.5 cm, pas de pied légal).

    Les pages de contenu gardent bottom_margin_cm (ex: 2.5 cm pour pied de page).
    La page QR-Bill utilise QR_BILL_PAGE_BOTTOM_MARGIN_CM (0.5 cm).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    from reportlab.platypus.doctemplate import _doNothing

    if on_later_pages is None:
        on_later_pages = _doNothing

    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=top_margin_cm * cm,
        bottomMargin=bottom_margin_cm * cm,
        leftMargin=left_margin_cm * cm,
        rightMargin=right_margin_cm * cm,
    )
    doc._calc()

    # Frame pages contenu (marge bas standard)
    frame_content = Frame(
        doc.leftMargin,
        bottom_margin_cm * cm,
        doc.width,
        A4[1] - doc.topMargin - bottom_margin_cm * cm,
        id="normal",
    )
    # Frame page QR-Bill (marge bas 0.5 cm, pas de pied légal)
    qrbill_bottom = QR_BILL_PAGE_BOTTOM_MARGIN_CM * cm
    frame_qrbill = Frame(
        doc.leftMargin,
        qrbill_bottom,
        doc.width,
        A4[1] - doc.topMargin - qrbill_bottom,
        id="qrbill",
    )

    doc.addPageTemplates(
        [
            PageTemplate(id="First", frames=frame_content, onPage=on_first_page),
            PageTemplate(id="Later", frames=frame_content, onPage=on_later_pages),
            PageTemplate(id="QRBill", frames=frame_qrbill),
        ]
    )

    # Basculer vers Later après la première page (comportement SimpleDocTemplate)
    def _handle_page_begin():
        doc._handle_pageBegin()
        if doc.page == 1:
            doc._handle_nextPageTemplate("Later")

    doc.handle_pageBegin = _handle_page_begin
    return doc


def _make_qr_bill_table(drawing: Any) -> Any:
    """Crée un tableau ReportLab pour afficher le QR-Bill (dimensions et style unifiés).

    Utilisé par pdf.py et partnerships/invoices_pdf.py.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import Table, TableStyle

    w_pt = QR_BILL_WIDTH_CM * cm * QR_BILL_SCALE_FACTOR
    h_pt = QR_BILL_HEIGHT_CM * cm * QR_BILL_SCALE_FACTOR
    orig_w, orig_h = drawing.width, drawing.height
    drawing.width = w_pt
    drawing.height = h_pt
    if orig_w > 0 and orig_h > 0:
        drawing.scale(w_pt / orig_w, h_pt / orig_h)

    col_widths = [
        QR_BILL_TABLE_COL_WIDTHS_CM[0] * cm,
        QR_BILL_TABLE_COL_WIDTHS_CM[1] * cm,
    ]
    qr_table = Table([[drawing, ""]], colWidths=col_widths)
    left_pad_pt = QR_BILL_LEFT_PADDING_MM * mm
    qr_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("LEFTPADDING", (0, 0), (0, 0), left_pad_pt),
                ("LEFTPADDING", (1, 0), (1, 0), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return qr_table


def _compute_c5_zone_canvas_coords(
    _page_w: float, page_h: float
) -> tuple[float, float, float, float]:
    """Calcule les coordonnées canvas pour la zone fenêtre C5.

    ReportLab : origine en bas gauche. DEST_ADDR_Y_MM = « depuis le haut ».
    Conversion : rect_bottom = page_h - y_from_top - zone_height.

    Returns:
        (x_pt, rect_bottom_pt, zone_w_pt, zone_h_pt)
    """
    from reportlab.lib.units import mm

    x_pt = DEST_ADDR_X_MM * mm
    zone_w_pt = DEST_ADDR_MAX_WIDTH_MM * mm
    zone_h_pt = DEST_ADDR_ZONE_HEIGHT_MM * mm
    y_from_top_pt = DEST_ADDR_Y_MM * mm
    rect_bottom_pt = page_h - y_from_top_pt - zone_h_pt
    return (x_pt, rect_bottom_pt, zone_w_pt, zone_h_pt)


def _on_first_page_debug_envelope(canvas: Any, doc: Any) -> None:
    """Dessine un rectangle guide « fenêtre enveloppe C5 » si PDF_DEBUG_ENVELOPE=1.

    DEV uniquement : rectangle rouge + repères mm pour vérifier le positionnement.
    Jamais activé par défaut en prod (opt-in explicite via env).
    """
    import os

    if os.environ.get("PDF_DEBUG_ENVELOPE") != "1":
        return
    from reportlab.lib import colors

    page_w, page_h = doc.pagesize
    x_pt, rect_bottom, zone_w_pt, zone_h_pt = _compute_c5_zone_canvas_coords(
        page_w, page_h
    )

    canvas.saveState()
    canvas.setStrokeColor(colors.red)
    canvas.setLineWidth(0.5)
    canvas.rect(x_pt, rect_bottom, zone_w_pt, zone_h_pt, stroke=1, fill=0)
    canvas.setFont("Helvetica", 6)
    canvas.setFillColor(colors.red)
    canvas.drawString(
        x_pt, rect_bottom - 8, f"C5 window {DEST_ADDR_X_MM:.0f}x{DEST_ADDR_Y_MM:.0f}mm"
    )
    canvas.restoreState()


def _format_company_contact_footer_bar(
    company_name: str,
    email: str,
    phone: str,
    uid: str,
) -> str:
    """Ligne de rappel d’identité en pied de page : « Société | email | tél | IDE/UID : … »."""
    parts: list[str] = []
    n = (company_name or "").strip()
    if n:
        parts.append(n)
    e = (email or "").strip()
    if e:
        parts.append(e)
    p = (phone or "").strip()
    if p:
        parts.append(p)
    u = (uid or "").strip()
    if u:
        parts.append(f"IDE/UID : {u}")
    return " | ".join(parts)


def _make_legal_footer_page_callback(
    footer_message: str,
    mention: str | None,
    centered_style: Any,
    contact_bar: str | None = None,
    platform_tagline: str | None = None,
) -> Any:
    """Crée un callback pour dessiner le pied de page légal en bas de page (zone fixe).

    Le pied de page est dessiné dans la marge inférieure, pas dans le flux du contenu.
    Texte légal + IBAN et barre identité : **centrés** ; barre identité **à la ligne** sous le bloc légal ; mention LIRIE **centrée**.
    ``platform_tagline`` : ``None`` = ``FOOTER_PLATFORM_TAGLINE`` ; ``\"\"`` = masquer.
    """

    def _draw_footer(canvas: Any, doc: Any) -> None:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm, mm

        from reportlab.platypus import Paragraph

        canvas.saveState()
        page_w = doc.pagesize[0]
        left_x = doc.leftMargin
        right_x = page_w - doc.rightMargin
        avail_width = right_x - left_x
        # Même typo (8 pt, gris) : bloc légal + barre identité + LIRIE, tout **centré** sur la largeur utile.
        # Pas de spaceAfter du parent (centered_style).
        muted_footer_tagline_style = ParagraphStyle(
            "FooterTaglineMuted",
            parent=centered_style,
            fontSize=FONT_SECONDARY,
            leading=int(round(FONT_SECONDARY * 1.28)),
            textColor=colors.HexColor(COLOR_MUTED_PDF),
            spaceBefore=0,
            spaceAfter=0,
            alignment=TA_CENTER,
        )
        # y augmente vers le haut. Bas de page : mention LIRIE, puis trait, puis un bloc unique
        # (légal + IBAN + barre identité), puis mention rappel éventuelle.
        y_pos = 1.2 * cm

        tag_src = (
            FOOTER_PLATFORM_TAGLINE if platform_tagline is None else platform_tagline
        )
        tag = (tag_src or "").strip()

        upper_after_tag = bool(
            (contact_bar or "").strip()
            or (footer_message or "").strip()
            or (mention or "").strip()
        )

        if tag:
            p_tag = Paragraph(
                _xml_escape_for_paragraph(tag),
                muted_footer_tagline_style,
            )
            w_t, h_t = p_tag.wrap(avail_width, 100)
            p_tag.drawOn(canvas, left_x, y_pos)
            y_pos += h_t + 1.5 * mm

        if tag and upper_after_tag:
            canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
            canvas.setLineWidth(0.35)
            canvas.line(left_x, y_pos, right_x, y_pos)
            y_pos += 2 * mm

        fm = (footer_message or "").strip()
        bar = (contact_bar or "").strip()
        combined_legal_identity = ""
        if fm and bar:
            combined_legal_identity = (
                _reportlab_safe_footer_html(fm)
                + "<br/>"
                + _xml_escape_for_paragraph(bar)
            )
        elif fm:
            combined_legal_identity = _reportlab_safe_footer_html(fm)
        elif bar:
            combined_legal_identity = _xml_escape_for_paragraph(bar)

        if combined_legal_identity:
            p_body = Paragraph(combined_legal_identity, muted_footer_tagline_style)
            w_b, h_b = p_body.wrap(avail_width, 260)
            p_body.drawOn(canvas, left_x, y_pos)
            y_pos += h_b + 4

        if mention:
            p2 = Paragraph(
                f'<font size="8" color="grey">'
                f"{_xml_escape_for_paragraph(mention)}</font>",
                centered_style,
            )
            w2, _ = p2.wrap(avail_width, 50)
            p2.drawOn(canvas, (page_w - w2) / 2, y_pos)

        canvas.restoreState()

    return _draw_footer


def _normalize_address_for_comparison(address: str) -> str:
    """Normalise une adresse pour la comparaison (détection aller-retour).

    Normalisation robuste :
    - Minuscules
    - Suppression accents (approximative)
    - Suppression ponctuation
    - Suppression espaces multiples
    - Suppression virgules doubles
    - Trim

    Args:
        address: Adresse brute

    Returns:
        Adresse normalisée pour comparaison
    """
    if not address:
        return ""
    import re
    import unicodedata

    # Minuscules
    normalized = address.lower().strip()

    # Supprimer accents (normalisation Unicode NFD puis suppression des diacritiques)
    try:
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    except Exception:
        # Fallback si unicodedata échoue
        pass

    # Supprimer ponctuation (garder seulement lettres, chiffres, espaces)
    normalized = re.sub(r"[^\w\s]", "", normalized)

    # Supprimer virgules doubles et espaces multiples
    normalized = re.sub(r",+", "", normalized)  # Supprimer toutes les virgules
    normalized = re.sub(r"\s+", " ", normalized)  # Espaces multiples -> 1 espace

    return normalized.strip()


def _is_booking_cancelled(booking: Any) -> bool:
    """Vérifie si un booking est annulé (status CANCELED / CANCELLED).

    Gère enum (BookingStatus.CANCELED) et str pour robustesse.
    Accepte les deux orthographes (US: CANCELED, UK: CANCELLED).
    """
    if not booking:
        return False
    status_raw = getattr(booking, "status", None)
    if status_raw is None:
        return False
    # Enum: utiliser .value si disponible, sinon str()
    status_str = getattr(status_raw, "value", None) or str(status_raw) or ""
    return status_str.upper().strip() in {"CANCELED", "CANCELLED"}


def _get_cancellation_transport_display(booking: Any) -> str:
    """Libellé transport pour booking annulé (règle prioritaire S2).

    Si booking.status == CANCELED, le PDF ne doit jamais afficher "pickup → dropoff".
    Utilise cancellation_display_label si présent (annulation standardisée),
    sinon fallback "Annulation (historique)" (backfill / legacy).
    """
    if not booking:
        return "Annulation (historique)"
    label = getattr(booking, "cancellation_display_label", None)
    if label and str(label).strip():
        return str(label).strip()
    return "Annulation (historique)"


def _short_label_for_transport(address: str) -> str:
    """Produit un libellé pour 'A' ou 'B' dans 'A ↔ B' / 'A → B'.

    - Prend la première partie avant virgule si présente, sinon tout.
    - Pas de troncature : le Paragraph wrap automatiquement sur plusieurs lignes.
    """
    if not address:
        return ""
    s = address.strip()
    if "," in s:
        s = s.split(",")[0].strip()
    return s


def _short_detail_label(address: str) -> str:
    """Libellé pour la ligne détail A/R (ex. 'Courbes', 'HUG').

    - Avant virgule si présent.
    - Si " des ", " de ", " du " : prend la partie après (ex. "Chemin des Courbes" → "Courbes").
    - Pas de troncature : le Paragraph wrap automatiquement sur plusieurs lignes.
    """
    if not address or not address.strip():
        return ""
    s = address.strip()
    if "," in s:
        s = s.split(",")[0].strip()
    for sep in (" des ", " de ", " du "):
        if sep in s:
            part = s.split(sep)[-1].strip()
            if part:
                s = part
            break
    return s


def _svg_content_to_drawing(svg_content: str | bytes) -> Any:
    """Convertit du contenu SVG (str ou bytes) en drawing ReportLab via un fichier temporaire.

    svg2rlg() n'accepte que str | PathLike[str], pas BytesIO.
    """
    import os
    import tempfile

    from svglib.svglib import svg2rlg

    content = (
        svg_content.encode("utf-8") if isinstance(svg_content, str) else svg_content
    )
    fd, path = tempfile.mkstemp(suffix=".svg")
    try:
        try:
            os.write(fd, content)
        finally:
            os.close(fd)
        return svg2rlg(path)
    finally:
        with contextlib.suppress(OSError):
            Path(path).unlink()


# Constantes pour la détection d'aller-retour
_MIN_ITEMS_FOR_ROUND_TRIP = 2
# Colonne Transport : pas de limite de caractères — le Paragraph wrap automatiquement
# sur plusieurs lignes selon transport_w (11.5 cm).
# Tolérance pour les montants (delta acceptable en CHF)
_AMOUNT_TOLERANCE_CHF = Decimal("5.00")
# Fenêtre temporelle maximale pour un aller-retour (en heures)
_MAX_ROUND_TRIP_TIME_WINDOW_HOURS = 12
_FLOAT_EQ_EPS = 1e-9
_CHF_CENTS_IN_FRANC = 100
_SWISS_GROUP_DIGITS = 3
_ISO_DATE_LEN = 10
_DISPLAY_MAG_MAX = 1e12


def _detect_and_group_round_trips(
    invoice_lines_with_bookings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Détecte et regroupe les aller-retour dans les lignes de facture.

    Priorité de détection:
    1. Champs explicites (parent_booking_id, is_return) si disponibles
    2. Heuristique (adresses inversées + même jour + ≤12h + montants similaires)

    Args:
        invoice_lines_with_bookings: Liste de dict avec:
            - 'line': InvoiceLine
            - 'booking': Booking (ou None)
            - 'patient_id': int (depuis line.meta)
            - 'patient_name': str (depuis line.meta)
            - 'date': datetime (depuis booking.scheduled_time)
            - 'pickup': str (depuis booking.pickup_location)
            - 'dropoff': str (depuis booking.dropoff_location)
            - 'amount': Decimal (depuis line.line_total)

    Returns:
        Liste de lignes consolidées avec:
            - 'is_round_trip': bool
            - 'transport_display': str (format "A ↔ B" pour A/R, "A → B" pour aller simple)
            - 'transport_type': str ("A/R" ou "Aller")
            - 'date', 'patient_name', 'amount', etc.
    """
    from datetime import datetime

    # ✅ ÉTAPE 1: Détection par champs explicites (parent_booking_id, is_return)
    explicit_pairs: dict[int, list[dict[str, Any]]] = {}  # parent_id -> [items]
    items_by_booking_id: dict[int, dict[str, Any]] = {}  # booking.id -> item
    used_by_explicit = set()  # Indices utilisés par la détection explicite

    for idx, item in enumerate(invoice_lines_with_bookings):
        booking = item.get("booking")
        if not booking:
            continue
        booking_id = getattr(booking, "id", None)
        if booking_id:
            items_by_booking_id[booking_id] = item

        # Détecter les paires explicites via parent_booking_id
        parent_id = getattr(booking, "parent_booking_id", None)
        is_return = getattr(booking, "is_return", False)

        if parent_id:
            # Ce booking est un retour lié à un parent
            if parent_id not in explicit_pairs:
                explicit_pairs[parent_id] = []
            explicit_pairs[parent_id].append(
                {"idx": idx, "item": item, "type": "return"}
            )
        elif is_return and booking_id:
            # Ce booking est marqué comme retour (chercher le parent)
            if parent_id:
                if parent_id not in explicit_pairs:
                    explicit_pairs[parent_id] = []
                explicit_pairs[parent_id].append(
                    {"idx": idx, "item": item, "type": "return"}
                )

    # Regrouper les paires explicites trouvées
    consolidated_explicit = []
    for parent_id, return_items in explicit_pairs.items():
        if len(return_items) != 1:
            # Plusieurs retours pour un même parent = ambiguïté, ignorer
            continue
        parent_item = items_by_booking_id.get(parent_id)
        if not parent_item:
            # Parent non trouvé dans les lignes de facture, ignorer
            continue

        return_data = return_items[0]
        return_idx = return_data["idx"]
        return_item = return_data["item"]
        used_by_explicit.add(return_idx)

        # Trouver l'index du parent dans la liste originale
        parent_idx = None
        for idx, item in enumerate(invoice_lines_with_bookings):
            if (
                item.get("booking")
                and getattr(item["booking"], "id", None) == parent_id
            ):
                parent_idx = idx
                used_by_explicit.add(parent_idx)
                break

        if parent_idx is None:
            continue

        # Créer ligne consolidée A/R
        parent_booking = parent_item.get("booking")
        return_booking = return_item.get("booking")
        if not parent_booking or not return_booking:
            continue

        # ✅ Annulés : ne pas regrouper (chaque ligne reste standalone avec son libellé)
        if _is_booking_cancelled(parent_booking) or _is_booking_cancelled(
            return_booking
        ):
            used_by_explicit.discard(return_idx)
            used_by_explicit.discard(parent_idx)  # parent_idx déjà trouvé au-dessus
            continue

        pickup_aller = getattr(parent_booking, "pickup_location", "") or ""
        dropoff_aller = getattr(parent_booking, "dropoff_location", "") or ""

        amount_aller = parent_item.get("amount", Decimal("0"))
        amount_retour = return_item.get("amount", Decimal("0"))
        raw_sum = amount_aller + amount_retour
        amount_rounded = round_to_5_cents(Decimal(str(raw_sum)))

        short_a = _short_label_for_transport(pickup_aller)
        short_b = _short_label_for_transport(dropoff_aller)
        detail_a = _short_detail_label(pickup_aller)
        detail_b = _short_detail_label(dropoff_aller)

        date_aller = parent_item.get("date")
        date_retour = return_item.get("date")
        earliest = (
            date_aller
            if (date_aller and date_retour and date_aller <= date_retour)
            else date_retour
        )

        transport_display = f"{short_a} ↔ {short_b}"
        consolidated_explicit.append(
            {
                "is_round_trip": True,
                "transport_type": "A/R",
                "date": date_aller or date_retour,
                "earliest_scheduled": earliest,
                "patient_id": parent_item.get("patient_id"),
                "patient_name": parent_item.get("patient_name", "Patient"),
                "pickup": pickup_aller,
                "dropoff": dropoff_aller,
                "transport_display": transport_display,
                "aller_detail": f"{short_a} → {short_b}",
                "retour_detail": f"{short_b} → {short_a}",
                "aller_detail_short": f"{detail_a} → {detail_b}",
                "retour_detail_short": f"{detail_b} → {detail_a}",
                "amount": amount_rounded,
                "line1": parent_item.get("line"),
                "line2": return_item.get("line"),
                "booking1": parent_booking,
                "booking2": return_booking,
            }
        )

    # ✅ ÉTAPE 2: Heuristique pour les items non appariés explicitement
    # Grouper par (patient_id, date_jour) pour les items restants
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    # Items sans patient_id/date (ex: livraison matériel) : ne pas les perdre
    standalone_items: list[dict[str, Any]] = []

    for idx, item in enumerate(invoice_lines_with_bookings):
        if idx in used_by_explicit:
            # Déjà traité par détection explicite, ignorer
            continue
        patient_id = item.get("patient_id")
        date = item.get("date")
        booking = item.get("booking")

        # ✅ Annulés : ne pas regrouper, transport_display = libellé uniquement
        if _is_booking_cancelled(booking):
            item["is_round_trip"] = False
            item["transport_type"] = "Aller"
            item["transport_display"] = _get_cancellation_transport_display(booking)
            item["earliest_scheduled"] = item.get("date")
            standalone_items.append(item)
            continue

        if not patient_id or not date:
            # Pas de regroupement possible, garder tel quel (ne pas perdre la ligne)
            item["is_round_trip"] = False
            item["transport_type"] = "Aller"
            line = item.get("line")
            if line and line.type == InvoiceLineType.MATERIAL_DELIVERY:
                item["transport_display"] = (
                    line.description[:80] if line.description else "Livraison"
                )
            else:
                pickup = item.get("pickup", "")
                dropoff = item.get("dropoff", "")
                if pickup and dropoff:
                    short_a = _short_label_for_transport(pickup)
                    short_b = _short_label_for_transport(dropoff)
                    item["transport_display"] = f"{short_a} → {short_b}"
                else:
                    item["transport_display"] = (
                        f"{pickup} → {dropoff}" if pickup or dropoff else ""
                    )
            item["earliest_scheduled"] = item.get("date")
            standalone_items.append(item)
            continue

        # Date à la journée (sans heure)
        date_key = (
            date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)[:10]
        )
        groups[(patient_id, date_key)].append(item)

    consolidated_lines: list[dict[str, Any]] = []

    for (_patient_id, _date_key), items in groups.items():
        if len(items) < _MIN_ITEMS_FOR_ROUND_TRIP:
            # Pas assez d'items pour un A/R, garder tel quel
            for item in items:
                item["is_round_trip"] = False
                item["transport_type"] = "Aller"
                line = item.get("line")
                booking = item.get("booking")
                if _is_booking_cancelled(booking):
                    item["transport_display"] = _get_cancellation_transport_display(
                        booking
                    )
                elif line and line.type == InvoiceLineType.MATERIAL_DELIVERY:
                    item["transport_display"] = (
                        line.description[:80] if line.description else "Livraison"
                    )
                else:
                    pickup = item.get("pickup", "")
                    dropoff = item.get("dropoff", "")
                    if pickup and dropoff:
                        short_a = _short_label_for_transport(pickup)
                        short_b = _short_label_for_transport(dropoff)
                        item["transport_display"] = f"{short_a} → {short_b}"
                    else:
                        item["transport_display"] = (
                            f"{pickup} → {dropoff}" if pickup or dropoff else ""
                        )
                item["earliest_scheduled"] = item.get("date")
                consolidated_lines.append(item)
            continue

        # Chercher des paires A→B et B→A
        normalized_pairs = []
        for item in items:
            pickup = item.get("pickup", "")
            dropoff = item.get("dropoff", "")
            if not pickup or not dropoff:
                continue
            normalized_pairs.append(
                {
                    "item": item,
                    "pickup_norm": _normalize_address_for_comparison(pickup),
                    "dropoff_norm": _normalize_address_for_comparison(dropoff),
                    "pickup_orig": pickup,
                    "dropoff_orig": dropoff,
                }
            )

        # Chercher les paires aller-retour avec validations strictes (index par trajet pour éviter O(k²))
        matched_pairs = []
        used_indices = set()
        candidate_pairs = []  # Stocker toutes les paires candidates avant validation

        by_route: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx_route, pr in enumerate(normalized_pairs):
            by_route[(pr["pickup_norm"], pr["dropoff_norm"])].append(idx_route)

        for i, pair1 in enumerate(normalized_pairs):
            if i in used_indices:
                continue
            rev_key = (pair1["dropoff_norm"], pair1["pickup_norm"])
            for j in by_route.get(rev_key, []):
                if j <= i or j in used_indices:
                    continue
                pair2 = normalized_pairs[j]
                # pickup1 == dropoff2 et dropoff1 == pickup2 (impliqué par rev_key)
                item1 = pair1["item"]
                item2 = pair2["item"]
                date1 = item1.get("date")
                date2 = item2.get("date")
                delta_seconds = float("inf")
                if (
                    date1
                    and date2
                    and isinstance(date1, datetime)
                    and isinstance(date2, datetime)
                ):
                    delta_seconds = abs((date2 - date1).total_seconds())
                candidate_pairs.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "pair1": pair1,
                        "pair2": pair2,
                        "delta_seconds": delta_seconds,
                    }
                )

        # ✅ Préférer les paires les plus proches temporellement (tri par delta croissant)
        candidate_pairs.sort(key=lambda c: c["delta_seconds"])

        # ✅ Valider chaque paire candidate avec critères stricts
        for candidate in candidate_pairs:
            idx1 = candidate["idx1"]
            idx2 = candidate["idx2"]
            pair1 = candidate["pair1"]
            pair2 = candidate["pair2"]
            item1 = pair1["item"]
            item2 = pair2["item"]

            # ✅ Validation 1: Montants identiques (ou delta ≤ tolérance)
            amount1 = Decimal(str(item1.get("amount", 0)))
            amount2 = Decimal(str(item2.get("amount", 0)))
            amount_diff = abs(amount1 - amount2)
            if amount_diff > _AMOUNT_TOLERANCE_CHF:
                # Montants trop différents, ne pas regrouper
                continue

            # ✅ Validation 2: Fenêtre temporelle (si horaire disponible)
            date1 = item1.get("date")
            date2 = item2.get("date")
            if (
                date1
                and date2
                and isinstance(date1, datetime)
                and isinstance(date2, datetime)
            ):
                time_diff = abs(
                    (date2 - date1).total_seconds() / 3600
                )  # Différence en heures
                if time_diff > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS:
                    # Fenêtre temporelle trop large, ne pas regrouper
                    continue

            # ✅ Validation 3: Éviter ambiguïté (si plusieurs retours possibles pour le même aller)
            # Compter combien de retours possibles existent pour cet aller
            pickup1_norm = pair1["pickup_norm"]
            dropoff1_norm = pair1["dropoff_norm"]
            possible_returns = 0
            for other_pair in normalized_pairs:
                if (
                    other_pair["pickup_norm"] == dropoff1_norm
                    and other_pair["dropoff_norm"] == pickup1_norm
                ):
                    possible_returns += 1

            # Si plus d'un retour possible, ne pas regrouper (ambiguïté)
            if possible_returns > 1:
                continue

            # ✅ Toutes les validations passées : regrouper
            matched_pairs.append((idx1, idx2))
            used_indices.add(idx1)
            used_indices.add(idx2)

        # Retour au « hub » : ex. Foyer→activité le matin, Clinique→Foyer l'apres-midi (pas B→A strict).
        unmatched_for_hub = [
            i for i in range(len(normalized_pairs)) if i not in used_indices
        ]
        hub_candidates: list[dict[str, Any]] = []
        for ii in range(len(unmatched_for_hub)):
            for jj in range(ii + 1, len(unmatched_for_hub)):
                ia = unmatched_for_hub[ii]
                ib = unmatched_for_hub[jj]
                pa = normalized_pairs[ia]
                pb = normalized_pairs[ib]
                for a_idx, b_idx, par_a, par_b in (
                    (ia, ib, pa, pb),
                    (ib, ia, pb, pa),
                ):
                    if (
                        par_b["dropoff_norm"] == par_a["pickup_norm"]
                        and par_b["pickup_norm"] != par_a["dropoff_norm"]
                    ):
                        it_a = par_a["item"]
                        it_b = par_b["item"]
                        amount1 = Decimal(str(it_a.get("amount", 0)))
                        amount2 = Decimal(str(it_b.get("amount", 0)))
                        if abs(amount1 - amount2) > _AMOUNT_TOLERANCE_CHF:
                            continue
                        date_a = it_a.get("date")
                        date_b = it_b.get("date")
                        if (
                            date_a
                            and date_b
                            and isinstance(date_a, datetime)
                            and isinstance(date_b, datetime)
                        ) and (
                            abs((date_b - date_a).total_seconds() / 3600)
                            > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS
                        ):
                            continue
                        d1 = par_a.get("date")
                        d2 = par_b.get("date")
                        delta_seconds = float("inf")
                        if (
                            d1
                            and d2
                            and isinstance(d1, datetime)
                            and isinstance(d2, datetime)
                        ):
                            delta_seconds = abs((d2 - d1).total_seconds())
                        hub_candidates.append(
                            {
                                "a_idx": a_idx,
                                "b_idx": b_idx,
                                "delta_seconds": delta_seconds,
                            }
                        )
        hub_candidates.sort(key=lambda c: c["delta_seconds"])
        for cand in hub_candidates:
            ai, bi = cand["a_idx"], cand["b_idx"]
            if ai in used_indices or bi in used_indices:
                continue
            matched_pairs.append((ai, bi))
            used_indices.add(ai)
            used_indices.add(bi)

        # Chaîne : fin du 1er trajet = début du 2e (ex. clinique→foyer + foyer→domicile)
        chain_unmatched = [
            i for i in range(len(normalized_pairs)) if i not in used_indices
        ]
        if len(chain_unmatched) >= _MIN_ITEMS_FOR_ROUND_TRIP:
            chain_candidates: list[dict[str, Any]] = []
            for ii in range(len(chain_unmatched)):
                for jj in range(ii + 1, len(chain_unmatched)):
                    ia = chain_unmatched[ii]
                    ib = chain_unmatched[jj]
                    pa = normalized_pairs[ia]
                    pb = normalized_pairs[ib]
                    if pa["dropoff_norm"] != pb["pickup_norm"]:
                        continue
                    if (
                        pa["pickup_norm"] == pb["dropoff_norm"]
                        and pa["dropoff_norm"] == pb["pickup_norm"]
                    ):
                        continue
                    ita = pa["item"]
                    itb = pb["item"]
                    amt_a = Decimal(str(ita.get("amount", 0)))
                    amt_b = Decimal(str(itb.get("amount", 0)))
                    if abs(amt_a - amt_b) > _AMOUNT_TOLERANCE_CHF:
                        continue
                    da = ita.get("date")
                    db = itb.get("date")
                    if (
                        da
                        and db
                        and isinstance(da, datetime)
                        and isinstance(db, datetime)
                    ) and (
                        abs((db - da).total_seconds() / 3600)
                        > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS
                    ):
                        continue
                    delta_seconds = float("inf")
                    if (
                        da
                        and db
                        and isinstance(da, datetime)
                        and isinstance(db, datetime)
                    ):
                        delta_seconds = abs((db - da).total_seconds())
                    chain_candidates.append(
                        {
                            "ia": ia,
                            "ib": ib,
                            "delta_seconds": delta_seconds,
                        }
                    )
            chain_candidates.sort(key=lambda c: c["delta_seconds"])
            for cand in chain_candidates:
                ia, ib = cand["ia"], cand["ib"]
                if ia in used_indices or ib in used_indices:
                    continue
                matched_pairs.append((ia, ib))
                used_indices.add(ia)
                used_indices.add(ib)

        # Créer les lignes consolidées
        for idx1, idx2 in matched_pairs:
            pair1 = normalized_pairs[idx1]
            pair2 = normalized_pairs[idx2]
            item1 = pair1["item"]
            item2 = pair2["item"]

            # Déterminer l'ordre (aller puis retour)
            # L'aller est celui avec la date/heure la plus tôt
            date1 = item1.get("date")
            date2 = item2.get("date")
            if date2 and date1 and date2 < date1:
                # Inverser si nécessaire
                item1, item2 = item2, item1
                pair1, pair2 = pair2, pair1

            # Après tri : pair1 / item1 = segment le plus tôt.
            # Inverse strict : A→B et B→A. Chaîne : A→B puis B→C (dépose B = prise B).
            is_inverse = (
                pair1["pickup_norm"] == pair2["dropoff_norm"]
                and pair1["dropoff_norm"] == pair2["pickup_norm"]
            )
            is_chain_seg = (
                pair1["dropoff_norm"] == pair2["pickup_norm"] and not is_inverse
            )

            raw_sum = item1.get("amount", Decimal("0")) + item2.get(
                "amount", Decimal("0")
            )
            amount_rounded = round_to_5_cents(Decimal(str(raw_sum)))

            if is_chain_seg:
                orig_from = pair1["pickup_orig"]
                orig_to = pair2["dropoff_orig"]
                short_a = _short_label_for_transport(orig_from)
                short_b = _short_label_for_transport(orig_to)
                detail_a = _short_detail_label(orig_from)
                detail_b = _short_detail_label(orig_to)
                leg1_a = _short_label_for_transport(pair1["pickup_orig"])
                leg1_b = _short_label_for_transport(pair1["dropoff_orig"])
                leg2_a = _short_label_for_transport(pair2["pickup_orig"])
                leg2_b = _short_label_for_transport(pair2["dropoff_orig"])
                aller_detail_fmt = f"{leg1_a} → {leg1_b}"
                retour_detail_fmt = f"{leg2_a} → {leg2_b}"
                aller_detail_short_fmt = f"{_short_detail_label(pair1['pickup_orig'])} → {_short_detail_label(pair1['dropoff_orig'])}"
                retour_detail_short_fmt = f"{_short_detail_label(pair2['pickup_orig'])} → {_short_detail_label(pair2['dropoff_orig'])}"
                pickup_aller = orig_from
                dropoff_aller = orig_to
            else:
                pickup_aller = pair1["pickup_orig"]
                dropoff_aller = pair1["dropoff_orig"]
                short_a = _short_label_for_transport(pickup_aller)
                short_b = _short_label_for_transport(dropoff_aller)
                detail_a = _short_detail_label(pickup_aller)
                detail_b = _short_detail_label(dropoff_aller)
                aller_detail_fmt = f"{short_a} → {short_b}"
                retour_detail_fmt = f"{short_b} → {short_a}"
                aller_detail_short_fmt = f"{detail_a} → {detail_b}"
                retour_detail_short_fmt = f"{detail_b} → {detail_a}"

            d1 = item1.get("date")
            d2 = item2.get("date")
            earliest = d1 if (d1 and d2 and d1 <= d2) else d2
            b1 = item1.get("booking")
            b2 = item2.get("booking")
            is_cancelled = _is_booking_cancelled(b1) or _is_booking_cancelled(b2)
            transport_display = (
                _get_cancellation_transport_display(b1 or b2)
                if is_cancelled
                else f"{short_a} ↔ {short_b}"
            )
            consolidated = {
                "is_round_trip": True,
                "transport_type": "A/R",
                "date": item1.get("date"),
                "earliest_scheduled": earliest,
                "patient_id": _patient_id,
                "patient_name": item1.get("patient_name", "Patient"),
                "pickup": pickup_aller,
                "dropoff": dropoff_aller,
                "transport_display": transport_display,
                "aller_detail": aller_detail_fmt,
                "retour_detail": retour_detail_fmt,
                "aller_detail_short": aller_detail_short_fmt,
                "retour_detail_short": retour_detail_short_fmt,
                "amount": amount_rounded,
                "line1": item1.get("line"),
                "line2": item2.get("line"),
                "booking1": item1.get("booking"),
                "booking2": item2.get("booking"),
            }
            consolidated_lines.append(consolidated)

        # Ajouter les items non appariés (allers simples)
        for i, pair in enumerate(normalized_pairs):
            if i not in used_indices:
                item = pair["item"]
                item["is_round_trip"] = False
                item["transport_type"] = "Aller"
                line = item.get("line")
                booking = item.get("booking")
                if _is_booking_cancelled(booking):
                    item["transport_display"] = _get_cancellation_transport_display(
                        booking
                    )
                elif line and line.type == InvoiceLineType.MATERIAL_DELIVERY:
                    item["transport_display"] = (
                        line.description[:80] if line.description else "Livraison"
                    )
                else:
                    short_a = _short_label_for_transport(pair["pickup_orig"])
                    short_b = _short_label_for_transport(pair["dropoff_orig"])
                    item["transport_display"] = f"{short_a} → {short_b}"
                item["earliest_scheduled"] = item.get("date")
                consolidated_lines.append(item)

    # ✅ Combiner les résultats explicites, heuristiques et items standalone (sans patient/date)
    # L'ordre final est assuré par _sort_consolidated_lines_for_s2 : tri par (date, patient, heure)
    # → les standalone avec date sont intercalés, ceux sans date en fin de facture
    return consolidated_explicit + consolidated_lines + standalone_items


def _sort_consolidated_lines_for_s2(
    consolidated_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tri stable pour les lignes S2: date puis patient puis earliest_scheduled.

    Les lignes sans date (ex: standalone edge case) sont mises en fin de facture
    pour éviter des lignes "sans date" en tête qui perturberaient la lecture.
    """
    from datetime import datetime as dt

    _SENTINEL_NO_DATE = "9999-12-31"  # Trie après toutes les dates réelles

    def sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
        d = row.get("date")
        date_key = (
            d.strftime("%Y-%m-%d")
            if d and isinstance(d, dt)
            else (str(d)[:10] if d else _SENTINEL_NO_DATE)
        )
        patient = (row.get("patient_name") or "").strip()
        earliest = row.get("earliest_scheduled")
        earliest_key = (
            earliest.strftime("%H:%M:%S")
            if earliest and isinstance(earliest, dt)
            else (str(earliest) if earliest else "")
        )
        return (date_key, patient, earliest_key)

    return sorted(consolidated_lines, key=sort_key)


# Note: Cette fonction n'est actuellement pas utilisée mais peut être utile pour
# des formats d'adresse spécifiques dans le futur. Conservée pour référence.
def _format_address_for_display(address: str) -> str:  # pyright: ignore[reportUnusedFunction]
    """Formate une adresse pour l'affichage dans le PDF.

    Format de sortie :
    - Ligne 1 : Rue et numéro
    - Ligne 2 : Code postal et ville

    Args:
        address: Adresse complète (peut être au format "Rue, Numéro, Code Postal, Ville"
                 ou "Rue Numéro, Code Postal Ville" ou autres formats)

    Returns:
        Adresse formatée avec <br/> pour les retours à la ligne
    """
    if not address or address == "Adresse non renseignée":
        return "Adresse non renseignée"

    # Nettoyer l'adresse
    clean_address = address.strip()

    # Essayer différents formats d'adresse
    # Format 1: "Rue, Numéro, Code Postal, Ville" (avec virgules)
    parts = [p.strip() for p in clean_address.split(",")]

    MIN_ADDRESS_PARTS = 2
    MIN_ADDRESS_PARTS_POSTAL = 3
    MIN_ADDRESS_PARTS_CITY = 4

    if len(parts) >= MIN_ADDRESS_PARTS_CITY:
        # Format: "Rue, Numéro, Code Postal, Ville"
        street_and_number = f"{parts[0]}, {parts[1]}"
        postal_code = parts[2]
        city = parts[3]
        return f"{street_and_number}<br/>{postal_code} {city}"
    if len(parts) >= MIN_ADDRESS_PARTS_POSTAL:
        # Format: "Rue Numéro, Code Postal, Ville" ou "Rue, Code Postal, Ville"
        street = parts[0]
        postal_code = parts[1]
        city = parts[2]
        return f"{street}<br/>{postal_code} {city}"
    if len(parts) >= MIN_ADDRESS_PARTS:
        # Format: "Rue Numéro, Code Postal Ville"
        street = parts[0]
        # Essayer d'extraire code postal et ville de la dernière partie
        last_part = parts[-1].strip()
        parts_space = last_part.split()
        if len(parts_space) >= MIN_ADDRESS_PARTS:
            postal_code = parts_space[0]
            city = " ".join(parts_space[1:])
            return f"{street}<br/>{postal_code} {city}"
        # Si on ne peut pas parser, retourner tel quel avec un <br/> au milieu
        return f"{street}<br/>{last_part}"

    # Si le format n'est pas reconnu, essayer de trouver un code postal (4 chiffres)
    import re

    postal_match = re.search(r"\b(\d{4})\b", clean_address)
    if postal_match:
        postal_code = postal_match.group(1)
        postal_pos = clean_address.find(postal_code)
        street = clean_address[:postal_pos].strip().rstrip(",")
        city = clean_address[postal_pos + len(postal_code) :].strip()
        if street and city:
            return f"{street}<br/>{postal_code} {city}"

    # Fallback : retourner l'adresse telle quelle
    return clean_address


def _sanitize_billed_to_address(name: str, address: str) -> str:
    """Nettoie l'adresse « Facturé à » avant formatage.

    - Normalise les espaces.
    - Retire le nom (ex. clinique) si déjà présent dans l'adresse (évite doublons).
    - Supprime la répétition « CP Ville » si dupliquée (ex. 1247 Anières 9 1247).
    """
    import re

    if not address or not str(address).strip():
        return ""
    s = " ".join(str(address).strip().split())

    if name:
        name_norm = re.sub(r"\s+", " ", name.strip().lower())
        s_norm = s.lower()
        if name_norm and name_norm in s_norm:
            s = re.sub(re.escape(name), "", s, count=1, flags=re.IGNORECASE).strip(
                " ,\n"
            )

    m = re.search(r"\b(\d{4}\s+[A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)\b", s)
    if m:
        block = m.group(1).strip()
        _min_dup = 2  # garder 1ère occurrence si bloc dupliqué
        if s.lower().count(block.lower()) >= _min_dup:
            first = re.search(re.escape(block), s, flags=re.IGNORECASE)
            if first:
                end = first.end()
                s = s[:end].strip()
    return s


# Détection du pays dans l'adresse : (pattern regex, code ISO 2, libellé affiché)
_BILLED_TO_COUNTRY_PATTERNS = (
    (r"suisse|switzerland", "CH", "Suisse"),
    (r"france", "FR", "France"),
    (r"deutschland|germany", "DE", "Allemagne"),
    (r"italy|italia", "IT", "Italie"),
)


def _dedupe_postal_and_city_line(cp_city: str) -> str:
    """Supprime la répétition « NPA Ville NPA Ville » (ex. 1247 Anières 1247 Anières, Suisse)."""
    import re

    if not (cp_city and str(cp_city).strip()):
        return cp_city
    t = re.sub(r"\s+", " ", str(cp_city).strip())
    t = re.sub(r"(\b\d{4})\s*,\s+", r"\1 ", t)
    if "," in t:
        head, _sep, tail = t.partition(",")
        head = head.strip()
        tail = tail.strip().lstrip(",")
        m = re.match(r"^((\d{4}\s+.+?))(?:\s+\1)+$", head, re.IGNORECASE)
        if m:
            head = m.group(1).strip()
        return f"{head}, {tail}".replace(" ,", " ")
    m2 = re.match(r"^((\d{4}\s+.+?))(?:\s+\1)+$", t, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return t


def _format_billed_to_three_lines(raw: str, company_country: str | None = None) -> str:
    """Formate l'adresse « Facturé à » en exactement 2 lignes (rue+numéro, CP ville).

    Utilisé avec le nom (ligne 1) pour un bloc 3 lignes propre :
    - Ligne 1 : nom (billed_to_name)
    - Ligne 2 : rue + numéro
    - Ligne 3 : CP Ville. Le pays n'est ajouté que s'il est différent du domicile
      de la compagnie (company_country).

    Retourne "ligne2<br/>ligne3" (sans espaces superflus, pas de répétition).
    """
    import re

    if not raw or not str(raw).strip():
        return "Adresse non renseignée"
    s = " ".join(str(raw).strip().split())

    postal_match = re.search(r"\b(\d{4})\b", s)
    if not postal_match:
        return s
    postcode = postal_match.group(1)
    pos = postal_match.start()
    street = s[:pos].strip().rstrip(" ,")
    rest = s[pos + 4 :].strip()
    # Détecter le pays dans la partie après le CP (pour décider si on l'affiche)
    address_country_code = None
    address_country_display = None
    for pattern, code, display in _BILLED_TO_COUNTRY_PATTERNS:
        if re.search(pattern, rest, re.IGNORECASE):
            address_country_code = code
            address_country_display = display
            break
    # Enlever pays (Suisse, etc.) de la ville
    city = (
        re.sub(
            r"\s*(?:,?\s*)(?:Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$",
            "",
            rest,
            flags=re.IGNORECASE,
        )
        .strip()
        .rstrip(",")
        .strip()
    )
    if not street:
        street = "Adresse non renseignée"
    if not city:
        cp_city = postcode
    else:
        cp_city = f"{postcode} {city}"
        # N'afficher le pays que s'il est différent du domicile de la compagnie
        if address_country_code and address_country_display:
            show_country = not company_country or (
                address_country_code != (company_country or "").strip().upper()
            )
            if show_country:
                cp_city = f"{cp_city}, {address_country_display}"
    cp_city = _dedupe_postal_and_city_line(cp_city)
    return f"{street}<br/>{cp_city}"


def _sanitize_legal_footer_for_iban(footer: str) -> str:
    """Retire toute mention « [IBAN non configuré] » du legal_footer.

    En prod, on ne doit jamais afficher cette phrase dans le PDF.
    """
    if not footer:
        return ""
    import re

    s = footer
    s = re.sub(
        r"(<br\s*/?>\s*)?\s*IBAN\s*:\s*\[IBAN non configuré\]\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\[IBAN non configuré\]\s*", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _resolve_legal_footer_placeholders(
    footer: str,
    payment_terms_days: int,
    overdue_fee: str | float | Decimal,
    jours_text: str,
) -> str:
    """Remplace les placeholders dans legal_footer par les valeurs des paramètres de paiement.

    Placeholders supportés :
    - {payment_terms_days} : délai en jours (ex: 10)
    - {overdue_fee} : frais de retard formatés (ex: 15.00)
    - {jours} : "jours" ou "jour" selon payment_terms_days
    """
    if not footer:
        return ""
    fee_str = f"{float(overdue_fee):.2f}" if overdue_fee else "15.00"
    return (
        footer.replace("{payment_terms_days}", str(payment_terms_days))
        .replace("{overdue_fee}", fee_str)
        .replace("{jours}", jours_text)
    )


def _format_iban_for_footer_display(iban: str) -> str:
    """Groupe l’IBAN par blocs de 4 si compact ; espaces insécables entre groupes (évite coupure milieu PDF)."""
    raw = (iban or "").strip()
    if not raw:
        return ""
    compact = raw.replace(" ", "")
    if len(compact) >= 15:
        # U+00A0 : même rendu visuel qu’un espace, moins de césure au milieu de l’IBAN dans Paragraph
        nbsp = "\u00a0"
        return nbsp.join(compact[i : i + 4] for i in range(0, len(compact), 4))
    return raw


def _footer_chf_amount_no_break(fee_str: str) -> str:
    """« CHF » + espace insécable + montant : évite une coupure entre CHF et le nombre au pied de page."""
    return f"CHF\u00a0{(fee_str or '').strip()}"


def _build_default_legal_footer_html(
    payment_terms_days: int,
    overdue_fee: Any,
    iban_value: str | None,
) -> str:
    """Pied de page par défaut si ``legal_footer`` est vide ; texte et IBAN sur un flux unique (pas de ``<br/>``)."""
    try:
        fee_str = f"{float(overdue_fee):.2f}"
    except (TypeError, ValueError):
        fee_str = "15.00"
    try:
        n_days = int(payment_terms_days)
    except (TypeError, ValueError):
        try:
            n_days = int(float(payment_terms_days))
        except (TypeError, ValueError):
            n_days = 30

    if n_days < 1:
        n_days = 30

    if n_days <= 1:
        delai_phrase = (
            "Merci de régler cette facture dans un délai d'un jour "
            "suivant la date d'émission."
        )
    else:
        delai_phrase = (
            f"Merci de régler cette facture dans les {n_days} jours "
            "suivant la date d'émission."
        )

    core = (
        f"{delai_phrase} En cas de retard, des frais de rappel de "
        f"{_footer_chf_amount_no_break(fee_str)} "
        "peuvent s'appliquer (cf. conditions générales)."
    )
    if (iban_value or "").strip():
        return (
            f"{core} IBAN : "
            f"{_format_iban_for_footer_display((iban_value or '').strip())}"
        )
    return core


def _get_reminder_footer_message(level: int) -> str:
    """Retourne le texte de pied de page adapté au niveau de rappel."""
    if level == 1:
        return (
            "Sauf erreur ou croisement de nos courriers, le règlement de cette facture "
            "ne nous est pas parvenu. Nous vous remercions de bien vouloir procéder à "
            "son règlement sous 10 jours."
        )
    if level == LEVEL_THRESHOLD:
        return (
            "Malgré notre précédent rappel, le règlement de cette facture ne nous est "
            "pas parvenu. Nous vous prions de bien vouloir régulariser cette situation "
            "dans les meilleurs délais."
        )
    return (
        "Malgré nos précédents rappels, cette facture reste impayée. À défaut de "
        "règlement sous 5 jours, nous nous réservons le droit d'entreprendre des "
        "démarches de recouvrement."
    )


def _load_logo_ratio_safe(
    logo_path: Path | None, max_width_pt: float
) -> tuple[Any, float, float]:
    """Charge un logo (SVG/PNG/JPG) en préservant le ratio.

    - max_width_pt : largeur max en points; hauteur calculée depuis le ratio du fichier.
    - SVG : scale uniforme (min des rapports) pour éviter toute distorsion.
    - Raster : ImageReader getSize, height = max_width * (ih/iw).

    Returns:
        (logo_img, width_pt, height_pt) ou (None, 0.0, 0.0) en cas d'erreur.
    """
    if not logo_path or not Path(logo_path).exists():
        return (None, 0.0, 0.0)
    try:
        if Path(logo_path).suffix.lower() == ".svg":
            from svglib.svglib import (
                svg2rlg,
            )

            drawing = svg2rlg(str(logo_path))
            if not drawing:
                return (None, 0.0, 0.0)
            ow = float(drawing.width)
            oh = float(drawing.height)
            if ow <= 0 or oh <= 0:
                return (None, 0.0, 0.0)
            scale = max_width_pt / ow
            drawing.scale(scale, scale)
            return (drawing, max_width_pt, oh * scale)
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image

        ir = ImageReader(str(logo_path))
        iw, ih = ir.getSize()
        if iw <= 0:
            return (None, 0.0, 0.0)
        w_pt = max_width_pt
        h_pt = max_width_pt * (float(ih) / float(iw))
        img = Image(str(logo_path), width=w_pt, height=h_pt)
        return (img, w_pt, h_pt)
    except Exception as e:
        app_logger.warning("Logo ratio-safe load failed: %s", e)
        return (None, 0.0, 0.0)


def _ensure_dejavu_pdf_fonts() -> tuple[str, str]:
    """Enregistre les polices DejaVu au plus une fois par worker ; sinon Helvetica."""
    global _DEJAVU_PDF_FONTS_READY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if _DEJAVU_PDF_FONTS_READY:
        try:
            pdfmetrics.getFont("DejaVuSans")
            pdfmetrics.getFont("DejaVuSans-Bold")
            return ("DejaVuSans", "DejaVuSans-Bold")
        except Exception:
            return ("Helvetica", "Helvetica-Bold")

    try:
        pdfmetrics.registerFont(
            TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        )
        pdfmetrics.registerFont(
            TTFont(
                "DejaVuSans-Bold",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            )
        )
        _DEJAVU_PDF_FONTS_READY = True
        return ("DejaVuSans", "DejaVuSans-Bold")
    except Exception:
        _DEJAVU_PDF_FONTS_READY = True
        return ("Helvetica", "Helvetica-Bold")


def _bookings_by_reservation_ids_for_pdf(invoice: "Invoice") -> dict[int, Any]:
    """Une requête SQL pour tous les bookings référencés par les lignes de la facture."""
    from models import Booking

    ids: set[int] = set()
    lines = getattr(invoice, "lines", None)
    if lines:
        for line in lines:
            rid = getattr(line, "reservation_id", None)
            if rid is not None:
                ids.add(int(rid))
    if not ids:
        return {}
    rows = (
        Booking.query.filter(Booking.id.in_(ids))
        .options(joinedload(Booking.client).joinedload(Client.user))
        .all()
    )
    return {cast(int, b.id): b for b in rows}


def _name_with_uppercase_last_name(name: str) -> str:
    """Met le nom de famille (dernier mot) en majuscules pour le bloc « Facturé à »."""
    if not name or not str(name).strip():
        return name
    parts = name.strip().split()
    if not parts:
        return name
    parts[-1] = parts[-1].upper()
    return " ".join(parts)


def _get_billed_to(
    invoice: "Invoice",
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> tuple[str, str]:
    """Retourne (nom, adresse formatée) pour le bloc « Facturé à ».

    ``bookings_by_id`` : chargé par le pipeline PDF (évite N+1). Si None, requête
    ponctuelle pour S1_PATIENT / appels isolés.
    """
    company_country = None
    if getattr(invoice, "company", None) and getattr(
        invoice.company, "domicile_country", None
    ):
        company_country = (invoice.company.domicile_country or "CH").strip().upper()
    if not company_country:
        company_country = "CH"
    if getattr(invoice, "billing_party_id", None):
        from models import BillingParty as BillingPartyModel
        from models.billing_party import ClientBillingParty
        from models.enums import BillingPartyType, InvoiceBillingStrategy

        _client_id = getattr(invoice, "client_id", None)
        _bp_id = getattr(invoice, "billing_party_id", None)
        use_billing_party = True
        _link = None

        # Charger le billing_party en amont pour le bypass S2/clinic
        bp = BillingPartyModel.query.get(invoice.billing_party_id)

        # ════════════════════════════════════════════════════════════════════
        # BYPASS pour factures cliniques mensuelles (S2) ou établissements
        # ════════════════════════════════════════════════════════════════════
        # Pour ces factures multi-patients, on ne vérifie PAS le lien
        # ClientBillingParty car le client_id est arbitraire (premier patient).
        # On utilise directement le billing_party (clinique/EMS/hôpital).
        # ════════════════════════════════════════════════════════════════════
        _billing_strategy = getattr(invoice, "billing_strategy", None)
        _bp_type = getattr(bp, "type", None) if bp else None
        _is_clinic_invoice = (
            _billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY
            or _bp_type
            in (
                BillingPartyType.CLINIC,
                BillingPartyType.EMS,
                BillingPartyType.HOSPITAL,
            )
        )

        if _is_clinic_invoice:
            app_logger.info(
                "[PDF] Facture clinique/établissement (strategy=%s, bp_type=%s) → bypass lien ClientBillingParty (invoice_id=%s).",
                _billing_strategy.value if _billing_strategy else None,
                _bp_type.value if _bp_type else None,
                getattr(invoice, "id", None),
            )
            # On force l'utilisation du billing_party, pas de vérification du lien
            _link = None  # Pas de référence SPC pour les établissements
        elif _client_id is not None and _bp_id is not None:
            # ════════════════════════════════════════════════════════════════════
            # Logique standard : vérifier le lien ClientBillingParty
            # ════════════════════════════════════════════════════════════════════
            # Si le client n'a plus de lien avec ce tiers payeur (lien supprimé),
            # facturer au domicile du client (fallback).
            _link = ClientBillingParty.query.filter_by(
                client_id=_client_id, billing_party_id=_bp_id
            ).first()
            if _link is None:
                app_logger.info(
                    "[PDF] Lien client↔tiers payeur supprimé (invoice_id=%s, client_id=%s, billing_party_id=%s). Facturé à = domicile du client.",
                    getattr(invoice, "id", None),
                    _client_id,
                    _bp_id,
                )
                use_billing_party = False

        if use_billing_party and bp:
            raw = bp.billing_address or "Adresse non renseignée"
            raw = _sanitize_billed_to_address(bp.display_name or "Payeur", raw)
            addr = _format_billed_to_three_lines(
                raw or "Adresse non renseignée", company_country=company_country
            )
            if getattr(invoice, "client_id", None) and getattr(bp, "type", None) in (
                BillingPartyType.FAMILY,
                BillingPartyType.CURATORSHIP,
                BillingPartyType.OPAD,
                BillingPartyType.LAWYER,
                BillingPartyType.INSURANCE,
                BillingPartyType.OTHER,
            ):
                client = getattr(invoice, "client", None)
                if client and getattr(client, "user", None):
                    client_name = (
                        f"{client.user.first_name or ''} {(client.user.last_name or '').upper()}".strip()
                        or getattr(client.user, "username", None)
                        or "Client"
                    )
                    client_name = _name_with_uppercase_last_name(
                        client_name or "Client"
                    )
                    bp_name = _name_with_uppercase_last_name(
                        bp.display_name or "Payeur"
                    )
                    name = f"{client_name}\nc/o {bp_name}"
                else:
                    name = _name_with_uppercase_last_name(bp.display_name or "Payeur")
            else:
                name = _name_with_uppercase_last_name(bp.display_name or "Payeur")
            if (
                _client_id is not None
                and _bp_id is not None
                and (bp.display_name or "").upper().strip().find("SPC") >= 0
                and _link is not None
                and getattr(_link, "client_reference", None)
                and (_link.client_reference or "").strip()
            ):
                addr = f"{addr}<br/><br/><br/>No. SPC : {(_link.client_reference or '').strip()}"
            return (name, addr)
        if getattr(invoice, "billing_party_id", None) and not bp:
            app_logger.warning(
                "[PDF] billing_party_id=%s défini mais BillingParty introuvable (invoice_id=%s). Fallback.",
                getattr(invoice, "billing_party_id", None),
                getattr(invoice, "id", None),
            )
            return (_name_with_uppercase_last_name("Payeur"), "Adresse non renseignée")
    if invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id:
        from models import Client as ClientModel

        institution = ClientModel.query.get(invoice.bill_to_client_id)
        if institution and institution.is_institution:
            app_logger.info(
                "[PDF] Fallback legacy bill_to_client_id utilisé (invoice_id=%s, bill_to_client_id=%s).",
                getattr(invoice, "id", None),
                getattr(invoice, "bill_to_client_id", None),
            )
            name = _name_with_uppercase_last_name(
                institution.institution_name or "Institution"
            )
            raw = institution.billing_address or "Adresse non renseignée"
            raw = _sanitize_billed_to_address(name, raw)
            return (
                name,
                _format_billed_to_three_lines(
                    raw or "Adresse non renseignée", company_country=company_country
                ),
            )
        app_logger.warning(
            "[PDF] bill_to_client_id=%s défini mais institution introuvable/invalide (invoice_id=%s). Fallback.",
            getattr(invoice, "bill_to_client_id", None),
            getattr(invoice, "id", None),
        )
        return (
            _name_with_uppercase_last_name("Institution"),
            "Adresse non renseignée",
        )
    app_logger.info(
        "[PDF] Fallback client bénéficiaire utilisé (invoice_id=%s, client_id=%s).",
        getattr(invoice, "id", None),
        getattr(invoice, "client_id", None),
    )
    client = invoice.client

    # ── Institution client avec facturation patient : utiliser le nom/adresse du patient ──
    from models.enums import InvoiceBillingStrategy as IBS

    _billing_strat = getattr(invoice, "billing_strategy", None)
    if (
        client
        and getattr(client, "is_institution", False)
        and _billing_strat == IBS.S1_PATIENT
    ):
        app_logger.info(
            "[PDF] Client institution + S1_PATIENT → recherche patient réel (invoice_id=%s).",
            getattr(invoice, "id", None),
        )
        # Chercher le nom du patient depuis le premier booking de la facture
        _patient_name = None
        _patient_address = None
        if hasattr(invoice, "lines") and invoice.lines:
            from models import Booking

            for line in invoice.lines:
                # billed_booking est un backref InstrumentedList, pas un objet unique
                _bk_rel = getattr(line, "billed_booking", None)
                if isinstance(_bk_rel, list) and _bk_rel:
                    _bk = _bk_rel[0]
                elif _bk_rel and not isinstance(_bk_rel, list):
                    _bk = _bk_rel
                elif line.reservation_id and bookings_by_id is not None:
                    _bk = bookings_by_id.get(line.reservation_id)
                else:
                    # Hors pipeline PDF : requête unique de repli
                    _bk = (
                        Booking.query.get(line.reservation_id)
                        if line.reservation_id
                        else None
                    )
                if _bk and getattr(_bk, "customer_name", None):
                    _patient_name = _bk.customer_name
                    break
        # Chercher l'adresse du patient via InstitutionPatient
        if client.linked_institution_id:
            try:
                from models.institution_patient import InstitutionPatient
                from models.transport_request import TransportRequest

                _tr = (
                    TransportRequest.query.filter_by(
                        institution_id=client.linked_institution_id,
                    )
                    .order_by(TransportRequest.id.desc())
                    .first()
                )
                if _tr and _tr.patient_id:
                    _ip = InstitutionPatient.query.get(_tr.patient_id)
                    if _ip:
                        if not _patient_name:
                            _patient_name = (
                                f"{_ip.first_name or ''} {_ip.last_name or ''}".strip()
                            )
                        parts = [
                            _ip.address or "",
                            _ip.postal_code or "",
                            _ip.city or "",
                        ]
                        _patient_address = ", ".join(p for p in parts if p)
            except Exception as e:
                app_logger.warning("[PDF] Patient lookup error: %s", e)

        if _patient_name:
            p_name = _name_with_uppercase_last_name(_patient_name)
            p_raw = _patient_address or "Adresse non renseignée"
            p_raw = _sanitize_billed_to_address(p_name, p_raw)
            return (
                p_name,
                _format_billed_to_three_lines(
                    p_raw or "Adresse non renseignée", company_country=company_country
                ),
            )

    client_name = (
        f"{client.user.first_name or ''} {(client.user.last_name or '').upper()}".strip()
        or client.user.username
        or "Client"
    )
    client_name = _name_with_uppercase_last_name(client_name)
    # Si établissement de résidence (EMS, fondation, etc.) : Nom client puis nom établissement
    residence_facility = (getattr(client, "residence_facility", None) or "").strip()
    name = f"{client_name}\n{residence_facility}" if residence_facility else client_name
    raw = "Adresse non renseignée"
    if hasattr(client, "domicile_address") and client.domicile_address:
        street = client.domicile_address
        if (
            hasattr(client, "domicile_zip")
            and hasattr(client, "domicile_city")
            and client.domicile_zip
            and client.domicile_city
        ):
            raw = f"{street}, {client.domicile_zip} {client.domicile_city}"
        else:
            raw = street
    elif (
        hasattr(client, "user")
        and client.user
        and hasattr(client.user, "address")
        and client.user.address
    ):
        raw = client.user.address
    raw = _sanitize_billed_to_address(client_name, raw)
    return (
        name,
        _format_billed_to_three_lines(
            raw or "Adresse non renseignée", company_country=company_country
        ),
    )


def _wrap_line_by_words(line: str, max_chars: int = 90) -> str:
    """Wrap une ligne trop longue par mots, sans couper brutalement.

    max_chars: approximation ~3 chars/mm à 10pt pour zone C5 (~90mm).
    Fallback quand font metrics indisponibles.
    """
    if not line or len(line) <= max_chars:
        return line
    words = line.split()
    result: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        need = len(w) + (1 if current else 0)
        if current_len + need > max_chars and current:
            result.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += need
    if current:
        result.append(" ".join(current))
    return "\n".join(result)


def _wrap_line_by_width(
    line: str,
    font_name: str,
    font_size: float,
    max_width_pt: float,
) -> list[str]:
    """Wrap une ligne selon la largeur réelle (stringWidth / simpleSplit).

    Priorité : font metrics ReportLab. Fallback : max_chars si police non mesurable.
    """
    if not line or not line.strip():
        return []
    try:
        from reportlab.lib.utils import simpleSplit

        lines = simpleSplit(line, font_name, font_size, max_width_pt)
        return list(lines) if lines else [line]
    except Exception:
        max_chars = max(30, int(max_width_pt / 2.5))  # ~2.5 pt/char à 10pt
        wrapped = _wrap_line_by_words(line, max_chars=max_chars)
        return [ln for ln in wrapped.split("\n") if ln]


def _build_recipient_block_flowable(
    invoice: "Invoice",
    normal_style: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
    name_font_size: float | None = None,
    addr_font_size: float | None = None,
) -> tuple[Any | None, list[str]]:
    """Construit le flowable pour le bloc destinataire compatible zone C5.

    Zone fenêtre : uniquement nom + adresse (pas de label « Facturé à : »).
    - Filtre les lignes vides (no data => no UI).
    - Wrap via stringWidth/simpleSplit (font metrics ReportLab).
    - Ne dessine rien si aucune ligne utile.
    - 2ᵉ+ lignes du bloc nom (ex. ``c/o OPAD …`` après saut de ligne) : même taille que l’adresse,
      pas le corps du nom en gras.

    ``name_font_size`` / ``addr_font_size`` : hiérarchie type facture pro (ex. 12 / 10 pt) ;
    si omis, réutilise la taille du ``normal_style``.

    Returns:
        (Paragraph ou None, recipient_lines pour tests).
    """
    from reportlab.lib.units import mm

    name, addr = _get_billed_to(invoice, bookings_by_id=bookings_by_id)
    lines: list[str] = []
    if name and str(name).strip():
        for name_line in str(name).strip().split("\n"):
            if name_line.strip():
                lines.append(name_line.strip())
    name_count = len(lines)
    if addr:
        for part in (
            str(addr).replace("<br/>", "\n").replace("<br />", "\n").split("\n")
        ):
            p = part.strip()
            # Conserver les lignes vides (ex. 2 sauts avant "No. SPC")
            lines.append(p)
    # No data => no UI : ne rien afficher si aucune ligne utile
    if not lines:
        return (None, [])

    font_name = getattr(normal_style, "fontName", "Helvetica") or "Helvetica"
    base_fs = float(getattr(normal_style, "fontSize", 10) or 10)
    name_fs = float(name_font_size) if name_font_size is not None else base_fs
    addr_fs = float(addr_font_size) if addr_font_size is not None else base_fs
    max_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
    max_lines = max(1, int(DEST_ADDR_ZONE_HEIGHT_MM / DEST_ADDR_LINE_HEIGHT_MM))
    max_chars_fallback = max(30, int(DEST_ADDR_MAX_WIDTH_MM * 3))

    # Rôle par ligne source : nom principal (gras + name_fs) ; lignes suivantes du champ nom
    # (c/o …) comme l’adresse (addr_fs, sans gras).
    visual_rows: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        if i < name_count:
            role = "name_primary" if i == 0 else "name_co"
        else:
            role = "addr"
        fs = name_fs if role == "name_primary" else addr_fs
        if line == "":
            visual_rows.append(("", role))
        else:
            wrapped = _wrap_line_by_width(line, font_name, fs, max_width_pt)
            for wline in wrapped:
                visual_rows.append((wline, role))

    if len(visual_rows) > max_lines:
        visual_rows = visual_rows[:max_lines]
        last_t, last_role = visual_rows[-1]
        if last_t == "":
            pass
        elif len(last_t) + 1 > max_chars_fallback:
            truncated = last_t[: max_chars_fallback - 1]
            last_t = (
                truncated.rsplit(" ", 1)[0] + "…"
                if " " in truncated
                else truncated + "…"
            )
            visual_rows[-1] = (last_t, last_role)
        else:
            visual_rows[-1] = (last_t + "…", last_role)

    parts: list[str] = []
    for vl, role in visual_rows:
        if vl == "":
            parts.append("<br/>")
            continue
        esc = _xml_escape_for_paragraph(vl)
        if role == "name_primary":
            parts.append(f'<font size="{int(name_fs)}"><b>{esc}</b></font><br/>')
        else:
            parts.append(f'<font size="{int(addr_fs)}">{esc}</font><br/>')

    text = "".join(parts)
    # Ne pas utiliser rstrip("<br/>") : il enlève tout caractère dans {<,b,r,/,>} en fin de chaîne
    # et peut corrompre ``</font>`` (par ex. ``…</font><br/>`` → ``…</fon``) puis paraparser.
    while text.endswith("<br/>"):
        text = text[:-5]
    from reportlab.platypus import Paragraph

    para = Paragraph(text, normal_style)
    return (para, lines)


def _xml_escape_for_paragraph(text: str) -> str:
    """Échappe & < > pour du contenu dans un ReportLab Paragraph (mini HTML)."""
    s = _html_escape_minimal(text or "", quote=False)
    return s.replace("'", "&apos;").replace('"', "&quot;")


def _reportlab_safe_footer_html(text: str) -> str:
    """Texte configurable (pied légal, etc.) : conserve uniquement les sauts ``<br/>`` ; échappe le reste.

    Évite les erreurs paraparser (balises ``font`` / ``para`` non fermées) si le texte contient
    des ``<`` ou du pseudo-HTML invalide.
    """
    import re

    if not text:
        return ""
    parts = re.split(r"(?i)(<br\s*/?>)", text)
    out: list[str] = []
    for part in parts:
        if re.fullmatch(r"(?i)<br\s*/?>", part or ""):
            out.append("<br/>")
        else:
            out.append(_xml_escape_for_paragraph(part))
    return "".join(out)


def _reportlab_multiline_plain_to_html(text: str) -> str:
    """Notes / texte multiligne : une ligne = un ``<br/>``, tout le contenu échappé."""
    t = text or ""
    if not t:
        return ""
    return "<br/>".join(_xml_escape_for_paragraph(line) for line in t.splitlines())


def _pdf_note_global_discount_for_totals_table(note: str) -> str:
    """Texte note remise globale (méta) : jamais de HTML brut vers le Table/Paragraph ReportLab."""
    s = (note or "").strip()[:280]
    if not s:
        return ""
    return _xml_escape_for_paragraph(s)


def _pdf_s2_ar_tag_markup() -> str:
    """Tag [A/R] dans le détail — pas d'attribut couleur (évite soucis paraparser)."""
    return f'<font size="{int(FONT_SECONDARY)}">[A/R]</font>'


def _collect_adjustment_notes_from_consolidated_item(
    item: dict[str, Any],
) -> str | None:
    """Notes d'ajustement (ex. remise %) — trajets A/R = union des deux lignes, sans doublon."""
    notes: list[str] = []
    if item.get("is_round_trip"):
        for key in ("line1", "line2"):
            ln = item.get(key)
            raw = getattr(ln, "adjustment_note", None) if ln is not None else None
            if raw is not None:
                s = str(raw).strip()
                if s:
                    notes.append(s)
    else:
        ln = item.get("line")
        raw = getattr(ln, "adjustment_note", None) if ln is not None else None
        if raw is not None:
            s = str(raw).strip()
            if s:
                notes.append(s)
    seen: set[str] = set()
    unique: list[str] = []
    for n in notes:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    if not unique:
        return None
    return " · ".join(unique) if len(unique) > 1 else unique[0]


def _gd_percent_hint_display(raw: Any) -> str:
    """Comme React ``Number(gd.percent) || '—'`` (libellé avant « % » dans la parenthèse)."""
    import math

    if raw is None:
        return "—"
    if isinstance(raw, str) and raw.strip() == "":
        return "—"
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(n):
        return "—"
    if n == 0:
        return "—"
    if abs(n - round(n)) < _FLOAT_EQ_EPS:
        return str(round(n))
    s = f"{n:.10f}".rstrip("0").rstrip(".")
    return s if s else "—"


def _reportlab_paragraph_percent_literals(text: str) -> str:
    """Évite que ReportLab interprète ``%`` tout en affichant un seul signe (``%%`` donnait « %% » à l'écran)."""
    return (text or "").replace("%", "&#37;")


class _GlobalDiscountHintBox(Flowable):
    """Encadré type ``.globalDiscHint`` : fond, bordure 1px, coins arrondis (équiv. 6px)."""

    def __init__(
        self,
        paragraph: Any,
        width_pt: float,
        *,
        pad_v_pt: float = 6.0,
        pad_h_pt: float = 7.5,
        radius_pt: float = 4.5,
    ) -> None:
        super().__init__()
        self._para = paragraph
        self._box_w = float(width_pt)
        self._pad_v = float(pad_v_pt)
        self._pad_h = float(pad_h_pt)
        self._radius = float(radius_pt)
        self.h = 1.0

    @override
    def wrap(self, aW: float, aH: float) -> tuple[float, float]:
        content_w = min(self._box_w, float(aW))
        inner_w = max(content_w - 2 * self._pad_h, 10.0)
        _pw, ph = self._para.wrap(inner_w, aH if aH > 0 else 10_000)
        self.h = float(ph) + 2 * self._pad_v
        return (self._box_w, self.h)

    def draw(self) -> None:
        from reportlab.lib import colors

        c = self.canv
        c.saveState()
        r = self._radius
        r = min(r, self.h / 2.0 - 0.1, self._box_w / 2.0 - 0.1)
        r = max(r, 0.0)
        bg = colors.HexColor("#f8fafc")
        bd = colors.HexColor("#e2e8f0")
        c.setFillColor(bg)
        c.setStrokeColor(bd)
        c.setLineWidth(0.75)
        c.roundRect(0.0, 0.0, self._box_w, self.h, r, stroke=1, fill=1)
        self._para.drawOn(c, self._pad_h, self._pad_v)
        c.restoreState()


def _detail_lines_heading_paragraph(styles: Any, font_name_bold: str) -> Any:
    """Titre « Détail des prestations » — proche de InvoiceLivePreview (.sectionTitle)."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    ps = ParagraphStyle(
        "InvoiceDetailLinesHeading",
        parent=styles["Normal"],
        fontName=font_name_bold,
        fontSize=FONT_BODY,
        leading=int(round(FONT_BODY * 1.3)),
        spaceAfter=10,
        textColor=colors.HexColor("#334155"),
    )
    return Paragraph("DÉTAIL DES PRESTATIONS", ps)


def _global_discount_hint_flowable(
    invoice: "Invoice",
    styles: Any,
    font_name: str,
    *,
    content_width_pt: float | None = None,
    content_width_cm: float | None = None,
) -> Any | None:
    """Encadré « Réduction globale enregistrée (p %) — … ».

    Aligné sur InvoiceLivePreview.module.css `.globalDiscHint` :
    margin-bottom 12px (caller), padding 8px 10px, texte compact (≈ plus petit que le corps),
    couleur #475569, fond #f8fafc, bordure 1px #e2e8f0, border-radius 6px (coins arrondis PDF).
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, Paragraph

    meta = getattr(invoice, "meta", None)
    if isinstance(meta, str) and meta.strip():
        try:
            meta = json.loads(meta)
        except Exception:
            meta = None
    if not isinstance(meta, dict) or not meta.get("global_discount"):
        return None
    gd = meta.get("global_discount")
    if not isinstance(gd, dict):
        return None
    pct_disp = _gd_percent_hint_display(gd.get("percent"))
    note = str(gd.get("note") or "").strip()
    # Même chaîne que InvoiceLivePreview.jsx (une seule phrase, tiret cadratin U+2014).
    # Espace insécable avant « application. » pour éviter une coupure vilaine en fin de ligne.
    body_core = (
        f"Réduction globale enregistrée ({pct_disp} %) \u2014 "
        "montants HT ci-dessus après\u00a0application."
    )
    body = (
        f"{body_core} « {_xml_escape_for_paragraph(note[:200])} »"
        if note
        else body_core
    )
    body = _reportlab_paragraph_percent_literals(body)

    # PDF : texte très compact pour l'aspect « hint » neutre (plus petit que le corps).
    ps = ParagraphStyle(
        "GlobalDiscountHint",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=11,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0,
        spaceBefore=0,
        spaceAfter=0,
        textColor=colors.HexColor("#475569"),
    )
    para = Paragraph(body, ps)
    if content_width_pt is not None and content_width_pt > 0:
        w = float(content_width_pt)
    elif content_width_cm is not None:
        w = float(content_width_cm) * cm
    else:
        w = 17.0 * cm
    # padding 8px 10px ; radius 6px → ~4,5 pt ; bordure ~1 px → 0,75 pt
    box = _GlobalDiscountHintBox(
        para,
        w,
        pad_v_pt=6.0,
        pad_h_pt=7.5,
        radius_pt=4.5,
    )
    return KeepTogether(box)


def _pdf_show_ar_legend(
    invoice: Any,
    consolidated: list[dict[str, Any]],
    bookings_by_id: dict[int, Any] | None = None,
) -> bool:
    """Légende [A/R] : uniquement si une ligne du tableau affiche réellement ``[A/R]``.

    Aligné sur ``_build_s2_table`` : blocs consolidés + lignes orphelines RIDE/matériel
    (sans réservation résolue dans ``bookings_by_id``) avec ``round_trip_merge_partner_*``.
    Pas de légende sur aller simple sans ces indicateurs.
    """
    for item in consolidated:
        if _consolidated_item_shows_ar_tag_pdf(item):
            return True
    bb = bookings_by_id or {}
    for ln in getattr(invoice, "lines", []) or []:
        if getattr(ln, "type", None) not in (
            InvoiceLineType.RIDE,
            InvoiceLineType.MATERIAL_DELIVERY,
        ):
            continue
        rid = getattr(ln, "reservation_id", None)
        if rid is not None and bb.get(int(rid)):
            continue
        lm = getattr(ln, "line_meta", None)
        if not isinstance(lm, dict):
            continue
        if lm.get("preview_hide_merged_round_trip") is True:
            continue
        if lm.get("round_trip_merge_partner_reservation_id") is not None:
            return True
    return False


def _consolidated_item_indicates_round_trip_tag(item: dict[str, Any]) -> bool:
    """[A/R] structurel : regroupement détecté aller+retour avec deux segments dans l'item."""
    return bool(
        item.get("is_round_trip")
        and item.get("aller_detail")
        and item.get("retour_detail")
    )


def _consolidated_item_shows_ar_tag_pdf(item: dict[str, Any]) -> bool:
    """Une ligne PDF doit afficher [A/R] : consolidation réelle OU métier alignée HTML (merge partenaire).

    Ignore les lignes ``preview_hide_merged_round_trip`` (non rendues comme les lignes masquées HTML).
    """
    if _consolidated_item_indicates_round_trip_tag(item):
        return True
    for key in ("line1", "line2", "line"):
        ln = item.get(key)
        if ln is None:
            continue
        lm = getattr(ln, "line_meta", None)
        if not isinstance(lm, dict):
            continue
        if lm.get("preview_hide_merged_round_trip") is True:
            continue
        if lm.get("round_trip_merge_partner_reservation_id") is not None:
            return True
    return False


def _line_description_from_consolidated_item(item: dict[str, Any]) -> str | None:
    """Description facture (``InvoiceLine.description``) si renseignée — même source que InvoiceLivePreview.

    Pour un aller-retour consolidé, prend la première description non vide parmi les deux lignes.
    """

    def _one(ln: Any) -> str | None:
        if not ln:
            return None
        raw = getattr(ln, "description", None)
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None

    if item.get("is_round_trip"):
        for key in ("line1", "line2"):
            t = _one(item.get(key))
            if t:
                return t
    return _one(item.get("line"))


def _consolidated_item_is_ride_transport(item: dict[str, Any]) -> bool:
    """« Trajet : » uniquement pour les lignes RIDE (pas livraison matériel, CUSTOM, frais)."""
    if item.get("is_round_trip"):
        for key in ("line1", "line2"):
            ln = item.get(key)
            if ln is not None and getattr(ln, "type", None) == InvoiceLineType.RIDE:
                return True
        return False
    ln = item.get("line")
    return ln is not None and getattr(ln, "type", None) == InvoiceLineType.RIDE


def _consolidated_item_is_material_delivery(item: dict[str, Any]) -> bool:
    """Livraison matériel : préfixe « Livraison : », pas « Trajet : »."""
    if item.get("is_round_trip"):
        for key in ("line1", "line2"):
            ln = item.get(key)
            if (
                ln is not None
                and getattr(ln, "type", None) == InvoiceLineType.MATERIAL_DELIVERY
            ):
                return True
        return False
    ln = item.get("line")
    return (
        ln is not None
        and getattr(ln, "type", None) == InvoiceLineType.MATERIAL_DELIVERY
    )


def _pdf_escape_wrapped_plain(
    text: str,
    font_name: str,
    desc_inner_pt: float,
) -> str:
    """Texte plain → lignes wrappées selon largeur colonne, puis échappement HTML."""
    if not text or not str(text).strip():
        return ""
    lines = _wrap_line_by_width(
        str(text).strip(),
        font_name,
        float(FONT_BODY),
        desc_inner_pt,
    )
    return "<br/>".join(_xml_escape_for_paragraph(x) for x in lines)


def _pdf_limit_html_br_lines(html: str, max_lines: int | None) -> str:
    """Limite un bloc HTML (séparateur ``<br/>``) à ``max_lines`` lignes."""
    if not html or max_lines is None or max_lines <= 0:
        return html
    parts = html.split("<br/>")
    if len(parts) <= max_lines:
        return html
    kept = parts[:max_lines]
    kept[-1] = kept[-1].rstrip().rstrip(".… ") + "…"
    return "<br/>".join(kept)


def _truncate_text_to_width_with_ellipsis(
    text: str,
    *,
    font_name: str,
    font_size: float,
    max_width_pt: float,
) -> str:
    """Tronque une chaîne pour tenir sur une seule ligne selon la largeur réelle."""
    s = (text or "").strip()
    if not s:
        return ""
    wrapped = _wrap_line_by_width(s, font_name, font_size, max_width_pt)
    if len(wrapped) <= 1:
        return wrapped[0] if wrapped else ""
    ell = "…"
    lo = 1
    hi = len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = s[:mid].rstrip()
        probe = f"{candidate}{ell}" if candidate else ell
        if len(_wrap_line_by_width(probe, font_name, font_size, max_width_pt)) <= 1:
            best = probe
            lo = mid + 1
        else:
            hi = mid - 1
    out = best or ell
    return out.rstrip().rstrip(".… ") + "…"


def _pdf_format_transport_detail_inner_wrapped(
    raw: str,
    *,
    font_name: str,
    desc_inner_pt: float,
    is_ride_line: bool,
    is_material_delivery: bool,
    force_balanced_two_lines: bool = False,
) -> str:
    """Trajet / Livraison / libellé : préfixes métier + wrap largeur colonne (adresse complète, pas ville seule)."""
    s = (raw or "").strip()
    if s.lower().startswith("trajet :"):
        s = s[8:].strip()
    fs = float(FONT_BODY)
    if is_material_delivery and s:
        lines = _wrap_line_by_width(f"Livraison : {s}", font_name, fs, desc_inner_pt)
        return "<br/>".join(_xml_escape_for_paragraph(x) for x in lines)
    if not is_ride_line:
        if not s:
            return ""
        lines = _wrap_line_by_width(s, font_name, fs, desc_inner_pt)
        return "<br/>".join(_xml_escape_for_paragraph(x) for x in lines)
    if not s:
        return ""
    for sep in (" ↔ ", " → "):
        if sep in s:
            a, b = s.split(sep, 1)
            a_clean = a.strip()
            if a_clean.lower().startswith("trajet :"):
                a_clean = a_clean[8:].strip()
            b_clean = b.strip()
            if force_balanced_two_lines:
                line_a = _truncate_text_to_width_with_ellipsis(
                    f"Trajet : {a_clean}",
                    font_name=font_name,
                    font_size=fs,
                    max_width_pt=desc_inner_pt,
                )
                line_b = _truncate_text_to_width_with_ellipsis(
                    f"→ {b_clean}",
                    font_name=font_name,
                    font_size=fs,
                    max_width_pt=desc_inner_pt,
                )
                return "<br/>".join(
                    [
                        _xml_escape_for_paragraph(line_a),
                        _xml_escape_for_paragraph(line_b),
                    ]
                )
            chunk: list[str] = []
            chunk.extend(
                _wrap_line_by_width(
                    f"Trajet : {a_clean}",
                    font_name,
                    fs,
                    desc_inner_pt,
                )
            )
            chunk.extend(
                _wrap_line_by_width(
                    f"→ {b_clean}",
                    font_name,
                    fs,
                    desc_inner_pt,
                )
            )
            return "<br/>".join(_xml_escape_for_paragraph(x) for x in chunk)
    lines = _wrap_line_by_width(f"Trajet : {s}", font_name, fs, desc_inner_pt)
    return "<br/>".join(_xml_escape_for_paragraph(x) for x in lines)


def _build_s2_table(
    invoice: "Invoice",
    font_name: str,
    font_name_bold: str,
    s2_main_style: Any,
    bookings_by_id: dict[int, Any],
    *,
    include_non_ride: bool = False,
    available_width_pt: float | None = None,
    max_simple_description_lines: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Construit le tableau de détail des prestations.

    Aligné sur ``InvoiceLivePreview`` pour tous les cas : **Date | Description | Montant**.
    Le nom du client transporté (clinique S2, tierce partie, institution S1 patient) est rendu dans
    la description (``Client : …``), pas dans une colonne séparée. La colonne Montant reprend
    uniquement le montant (sans « CHF » sur une deuxième ligne).

    Si ``available_width_pt`` est fourni (ex. ``doc.width``), la colonne description
    prend la largeur restante. Sinon, largeurs fixes de repli.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    # ✅ Déterminer si c'est une facture client directe (non tierce partie)
    strategy_value = None
    try:
        bs = getattr(invoice, "billing_strategy", None)
        if bs is None:
            strategy_value = None
        else:
            strategy_value = bs.value if hasattr(bs, "value") else str(bs)
    except Exception:
        strategy_value = None
    is_s2_invoice = strategy_value == "s2_clinic_monthly"
    is_third_party_invoice = bool(
        getattr(invoice, "billing_party_id", None)
        or (
            invoice.bill_to_client_id and invoice.bill_to_client_id != invoice.client_id
        )
        or is_s2_invoice
    )
    # Aligné InvoiceLivePreview : afficher « catalogue → net HT » sur chaque ligne si
    # `original_line_total` est en méta, y compris lorsqu'une remise globale est enregistrée
    # (l'encadré et le bloc totaux restent la synthèse).
    suppress_line_discount_breakdown = False

    lines_with_bookings: list[dict[str, Any]] = []
    for line in invoice.lines:
        if (
            line.type
            not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            )
            or not line.reservation_id
        ):
            continue
        booking = bookings_by_id.get(line.reservation_id)
        if not booking:
            continue

        # ✅ Pour factures client (non tierce partie), utiliser le nom du client
        # Pour factures tierce partie/S2, utiliser le patient depuis meta ou booking
        patient_name = "Patient"
        patient_id = None

        if is_third_party_invoice or is_s2_invoice:
            # Facture tierce partie ou S2 : utiliser le patient depuis meta ou booking
            if hasattr(line, "meta") and isinstance(line.meta, dict):
                patient_name = (
                    line.meta.get("patient_name")
                    or booking.customer_name
                    or (
                        f"{booking.client.user.first_name or ''} "
                        f"{booking.client.user.last_name or ''}"
                    ).strip()
                    or "Patient"
                )
                patient_id = line.meta.get("patient_id") or booking.client_id
            else:
                patient_name = (
                    booking.customer_name
                    or (
                        f"{booking.client.user.first_name or ''} "
                        f"{booking.client.user.last_name or ''}"
                    ).strip()
                    or "Patient"
                )
                patient_id = booking.client_id
        else:
            # Facture client directe : utiliser le nom du client comme "patient"
            # Pour les clients institution avec facturation patient (S1_PATIENT),
            # utiliser booking.customer_name (= nom réel du patient)
            client = invoice.client
            _is_inst_patient = (
                client
                and getattr(client, "is_institution", False)
                and strategy_value == "s1_patient"
            )
            if _is_inst_patient and booking.customer_name:
                patient_name = booking.customer_name
            elif client and hasattr(client, "user") and client.user:
                patient_name = (
                    f"{client.user.first_name or ''} {client.user.last_name or ''}".strip()
                    or client.user.username
                    or "Client"
                )
            else:
                patient_name = "Client"
            patient_id = invoice.client_id if invoice.client_id else None
        if len(patient_name) > MAX_PATIENT_NAME_LENGTH:
            patient_name = patient_name[: MAX_PATIENT_NAME_LENGTH - 1] + "."
        lines_with_bookings.append(
            {
                "line": line,
                "booking": booking,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "date": booking.scheduled_time,
                "pickup": booking.pickup_location or "",
                "dropoff": booking.dropoff_location or "",
                "amount": line.line_total,
            }
        )
    consolidated = _detect_and_group_round_trips(lines_with_bookings)
    consolidated = _sort_consolidated_lines_for_s2(consolidated)
    show_date_column = _pdf_detail_table_show_date_column(invoice, consolidated)

    # Client privé direct : pas tierce / pas S2 (comportement remise globale dans le détail).
    is_compact_private = not is_third_party_invoice and not is_s2_invoice
    _thead_lead = int(round(FONT_TABLE_HEADER * 1.3))
    _thead_ps = ParagraphStyle(
        "InvoiceCompactThead",
        fontName=font_name_bold,
        fontSize=FONT_TABLE_HEADER,
        leading=_thead_lead,
        textColor=colors.black,
        spaceBefore=0,
        spaceAfter=0,
    )
    if show_date_column:
        _header_row = [
            Paragraph(
                "Date", ParagraphStyle("ThDate", parent=_thead_ps, alignment=TA_LEFT)
            ),
            Paragraph(
                "Description",
                ParagraphStyle("ThDesc", parent=_thead_ps, alignment=TA_LEFT),
            ),
            Paragraph(
                "<nobr>Montant</nobr>",
                ParagraphStyle("ThHt", parent=_thead_ps, alignment=TA_RIGHT),
            ),
        ]
    else:
        _header_row = [
            Paragraph(
                "Description",
                ParagraphStyle("ThDesc", parent=_thead_ps, alignment=TA_LEFT),
            ),
            Paragraph(
                "<nobr>Montant</nobr>",
                ParagraphStyle("ThHt", parent=_thead_ps, alignment=TA_RIGHT),
            ),
        ]
    if is_compact_private:
        # Avec remise globale, l'aperçu HTML conserve les sous-lignes catalogue → net par ligne.
        suppress_line_discount_breakdown = False
    # Largeur utile texte colonne Description (retrait padding L/R — aligné TableStyle body).
    _s2_desc_hpad_pt = 7.5 + 3.0
    _date_w_col = 2.95 * cm
    _amt_w_col = 2.75 * cm
    if available_width_pt is not None and float(available_width_pt) > 0:
        if show_date_column:
            _desc_w_for_wrap = float(
                max(available_width_pt - _date_w_col - _amt_w_col, 1 * cm)
            )
        else:
            _desc_w_for_wrap = float(max(available_width_pt - _amt_w_col, 1 * cm))
    else:
        _desc_w_for_wrap = float(12 * cm if show_date_column else 13 * cm)
    desc_inner_pt = max(_desc_w_for_wrap - _s2_desc_hpad_pt, 60.0)

    table_data = [_header_row]
    s2_patient_separator_after_rows: list[int] = []
    for i, item in enumerate(consolidated):
        if not is_compact_private and i > 0:
            prev = consolidated[i - 1]
            pk_prev = (prev.get("patient_id"), prev.get("patient_name", ""))
            pk_cur = (item.get("patient_id"), item.get("patient_name", ""))
            if pk_prev != pk_cur:
                s2_patient_separator_after_rows.append(len(table_data))
        date_str = ""
        if item.get("date"):
            date_str = item["date"].strftime("%d.%m.%Y")
        else:
            _ln_item = item.get("line") or item.get("line1")
            if _ln_item is not None:
                date_str = _pdf_line_detail_date_str(_ln_item, invoice)
        pn_raw = item.get("patient_name", "Patient")
        if len(pn_raw) > MAX_PATIENT_NAME_LENGTH:
            pn_raw = pn_raw[: MAX_PATIENT_NAME_LENGTH - 1] + "."
        cat_disp, net_disp = _consolidated_row_catalog_net(item)
        if suppress_line_discount_breakdown:
            cat_disp = None
        adj_note = _collect_adjustment_notes_from_consolidated_item(item)
        note_suffix = ""
        if adj_note:
            esc_n = _xml_escape_for_paragraph(adj_note)
            note_suffix = (
                f'<br/><font size="{int(FONT_SECONDARY)}" color="#6b7280">'
                f"<i>{esc_n}</i></font>"
            )
        is_ar = _consolidated_item_shows_ar_tag_pdf(item)
        is_ride_td = _consolidated_item_is_ride_transport(item)
        is_material_td = _consolidated_item_is_material_delivery(item)
        disc_suffix = ""
        if (
            not suppress_line_discount_breakdown
            and cat_disp is not None
            and abs(Decimal(cat_disp) - net_disp) > _PDF_CATALOG_NET_EPS
        ):
            disc_suffix = _pdf_s2_per_line_discount_suffix_html(
                cat_disp, net_disp, compact_private_sub=True
            )
        line_desc_opt = _line_description_from_consolidated_item(item)
        if is_ar:
            if line_desc_opt:
                if max_simple_description_lines == 2 and (
                    " → " in line_desc_opt or " ↔ " in line_desc_opt
                ):
                    esc_desc = _pdf_format_transport_detail_inner_wrapped(
                        line_desc_opt,
                        font_name=font_name,
                        desc_inner_pt=desc_inner_pt,
                        is_ride_line=True,
                        is_material_delivery=False,
                        force_balanced_two_lines=True,
                    )
                else:
                    esc_desc = _pdf_escape_wrapped_plain(
                        line_desc_opt, font_name, desc_inner_pt
                    )
                esc_desc = _pdf_limit_html_br_lines(
                    esc_desc, max_simple_description_lines
                )
                inner_html = (
                    f"{esc_desc} {_pdf_s2_ar_tag_markup()}{disc_suffix}{note_suffix}"
                )
            else:
                body_tr = _pdf_format_transport_detail_inner_wrapped(
                    item.get("transport_display", ""),
                    font_name=font_name,
                    desc_inner_pt=desc_inner_pt,
                    is_ride_line=is_ride_td,
                    is_material_delivery=is_material_td,
                    force_balanced_two_lines=max_simple_description_lines == 2,
                )
                body_tr = _pdf_limit_html_br_lines(
                    body_tr, max_simple_description_lines
                )
                inner_html = (
                    f"{body_tr} {_pdf_s2_ar_tag_markup()}{disc_suffix}{note_suffix}"
                )
            amount_cell = _pdf_s2_amount_only_paragraph(
                net_disp,
                s2_main_style,
                is_round_trip=False,
                ht_column_plain=True,
            )
        else:
            if line_desc_opt:
                if max_simple_description_lines == 2 and (
                    " → " in line_desc_opt or " ↔ " in line_desc_opt
                ):
                    line_desc_html = _pdf_format_transport_detail_inner_wrapped(
                        line_desc_opt,
                        font_name=font_name,
                        desc_inner_pt=desc_inner_pt,
                        is_ride_line=True,
                        is_material_delivery=False,
                        force_balanced_two_lines=True,
                    )
                else:
                    line_desc_html = _pdf_escape_wrapped_plain(
                        line_desc_opt, font_name, desc_inner_pt
                    )
                line_desc_html = _pdf_limit_html_br_lines(
                    line_desc_html, max_simple_description_lines
                )
                inner_html = f"{line_desc_html}{disc_suffix}{note_suffix}"
            else:
                transport = item.get("transport_display", "")
                body_tr = _pdf_format_transport_detail_inner_wrapped(
                    transport,
                    font_name=font_name,
                    desc_inner_pt=desc_inner_pt,
                    is_ride_line=is_ride_td,
                    is_material_delivery=is_material_td,
                    force_balanced_two_lines=max_simple_description_lines == 2,
                )
                body_tr = _pdf_limit_html_br_lines(
                    body_tr, max_simple_description_lines
                )
                inner_html = f"{body_tr}{disc_suffix}{note_suffix}"
            amount_cell = _pdf_s2_amount_only_paragraph(
                net_disp,
                s2_main_style,
                is_round_trip=False,
                ht_column_plain=True,
            )
        patient_prefix_html = ""
        if is_compact_private:
            if (
                strategy_value == "s1_patient"
                and invoice.client
                and getattr(invoice.client, "is_institution", False)
                and pn_raw
            ):
                patient_prefix_html = (
                    f'<font size="{int(FONT_SECONDARY)}" color="#475569">Client : '
                    f"{_xml_escape_for_paragraph(pn_raw)}</font><br/>"
                )
        elif pn_raw and str(pn_raw).strip() not in ("Patient",):
            # Clinique S2 / tierce : ``InvoiceLivePreview`` `.lineClinicContext` (#475569, ~11px).
            patient_prefix_html = (
                f'<font size="{int(FONT_SECONDARY)}" color="#475569">Client : '
                f"{_xml_escape_for_paragraph(pn_raw)}</font><br/>"
            )
        desc_cell = Paragraph(f"{patient_prefix_html}{inner_html}", s2_main_style)
        if show_date_column:
            table_data.append(
                [
                    _compact_private_date_paragraph(date_str, font_name),
                    desc_cell,
                    amount_cell,
                ]
            )
        else:
            table_data.append([desc_cell, amount_cell])

    if include_non_ride:
        for line in invoice.lines:
            # Lignes hors transport : ex. accompagnement (CUSTOM > 0), honoraires.
            if line.type not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            ):
                # CUSTOM ≤ 0 hors déduction manuelle : synthèse totaux ; CUSTOM > 0 technique : exclu
                if (
                    line.type == InvoiceLineType.CUSTOM
                    and not _custom_line_include_in_s2_detail_table(line)
                ):
                    continue
                cat_o, net_o = _line_catalog_vs_net_ht(line)
                cat_show = cat_o if abs(cat_o - net_o) > _PDF_CATALOG_NET_EPS else None
                if suppress_line_discount_breakdown:
                    cat_show = None
                disc_o = ""
                if (
                    not suppress_line_discount_breakdown
                    and cat_show is not None
                    and abs(cat_o - net_o) > _PDF_CATALOG_NET_EPS
                ):
                    disc_o = _pdf_s2_per_line_discount_suffix_html(
                        cat_o, net_o, compact_private_sub=True
                    )
                esc_d = _pdf_escape_wrapped_plain(
                    (line.description or "")[:500], font_name, desc_inner_pt
                )
                sub = _custom_prestation_subline_for_pdf(line)
                if sub:
                    esc_s = _xml_escape_for_paragraph(sub)
                    desc_html = f'{esc_d}<br/><font size="{FONT_SECONDARY}" color="#64748b">{esc_s}</font>'
                else:
                    desc_html = esc_d
                desc_html = f"{desc_html}{disc_o}"
                desc_html = _pdf_limit_html_br_lines(
                    desc_html, max_simple_description_lines
                )
                desc_cell = Paragraph(desc_html, s2_main_style)
                amt_cell_o = _pdf_s2_amount_only_paragraph(
                    net_o,
                    s2_main_style,
                    is_round_trip=False,
                    ht_column_plain=True,
                )
                svc_date_o = _pdf_line_detail_date_str(line, invoice)
                _date_cell_o = _compact_private_date_paragraph(
                    svc_date_o or "", font_name
                )
                if show_date_column:
                    table_data.append([_date_cell_o, desc_cell, amt_cell_o])
                else:
                    table_data.append([desc_cell, amt_cell_o])
        # Transport MATÉRIEL / course sans réservation résolue : absent du bloc consolidé mais dans le total
        for line in invoice.lines:
            if line.type not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            ):
                continue
            if line.reservation_id and bookings_by_id.get(line.reservation_id):
                continue
            if line.line_total is None:
                continue
            amt = line.line_total
            if amt == 0:
                continue
            cat_or, net_or = _line_catalog_vs_net_ht(line)
            cat_show_o = cat_or if abs(cat_or - net_or) > _PDF_CATALOG_NET_EPS else None
            if suppress_line_discount_breakdown:
                cat_show_o = None
            disc_or = ""
            if (
                not suppress_line_discount_breakdown
                and cat_show_o is not None
                and abs(cat_or - net_or) > _PDF_CATALOG_NET_EPS
            ):
                disc_or = _pdf_s2_per_line_discount_suffix_html(
                    cat_or, net_or, compact_private_sub=True
                )
            raw_desc_orphan = (line.description or "")[:500]
            if (
                line.type == InvoiceLineType.MATERIAL_DELIVERY
                and raw_desc_orphan.strip()
            ):
                esc_d = _pdf_format_transport_detail_inner_wrapped(
                    raw_desc_orphan,
                    font_name=font_name,
                    desc_inner_pt=desc_inner_pt,
                    is_ride_line=False,
                    is_material_delivery=True,
                )
            elif line.type == InvoiceLineType.RIDE and raw_desc_orphan.strip():
                esc_d = _pdf_format_transport_detail_inner_wrapped(
                    raw_desc_orphan,
                    font_name=font_name,
                    desc_inner_pt=desc_inner_pt,
                    is_ride_line=True,
                    is_material_delivery=False,
                    force_balanced_two_lines=max_simple_description_lines == 2,
                )
            else:
                esc_d = _pdf_escape_wrapped_plain(
                    raw_desc_orphan, font_name, desc_inner_pt
                )
            lm_or = line.line_meta if isinstance(line.line_meta, dict) else {}
            orphan_ar = (
                lm_or.get("round_trip_merge_partner_reservation_id") is not None
                and lm_or.get("preview_hide_merged_round_trip") is not True
            )
            ar_suffix = ""
            if orphan_ar:
                ar_suffix = f" {_pdf_s2_ar_tag_markup()}"
            orphan_pn_prefix = ""
            if is_third_party_invoice or is_s2_invoice:
                raw_pn = lm_or.get("patient_name")
                if raw_pn and str(raw_pn).strip() and str(raw_pn).strip() != "—":
                    pn_disp = str(raw_pn).strip()
                    if len(pn_disp) > MAX_PATIENT_NAME_LENGTH:
                        pn_disp = pn_disp[: MAX_PATIENT_NAME_LENGTH - 1] + "."
                    orphan_pn_prefix = (
                        f'<font size="{int(FONT_SECONDARY)}" color="#475569">Client : '
                        f"{_xml_escape_for_paragraph(pn_disp)}</font><br/>"
                    )
            orphan_inner_html = _pdf_limit_html_br_lines(
                f"{esc_d}{ar_suffix}{disc_or}", max_simple_description_lines
            )
            desc_cell = Paragraph(
                f"{orphan_pn_prefix}{orphan_inner_html}", s2_main_style
            )
            amt_cell_or = _pdf_s2_amount_only_paragraph(
                net_or,
                s2_main_style,
                is_round_trip=False,
                ht_column_plain=True,
            )
            orphan_date = _pdf_line_detail_date_str(line, invoice) or ""
            _date_cell_or = _compact_private_date_paragraph(orphan_date, font_name)
            if show_date_column:
                table_data.append([_date_cell_or, desc_cell, amt_cell_or])
            else:
                table_data.append([desc_cell, amt_cell_or])

    # Largeurs : Date / Montant assez larges pour éviter coupures (nobr + police date réduite).
    date_w = 2.95 * cm
    amount_w = 2.75 * cm
    if available_width_pt is not None and available_width_pt > 0:
        if show_date_column:
            desc_w = max(available_width_pt - date_w - amount_w, 1 * cm)
            col_widths = [date_w, desc_w, amount_w]
        else:
            desc_w = max(available_width_pt - amount_w, 1 * cm)
            col_widths = [desc_w, amount_w]
    else:
        if show_date_column:
            desc_w = 12 * cm
            col_widths = [date_w, desc_w, amount_w]
        else:
            desc_w = 13 * cm
            col_widths = [desc_w, amount_w]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    _hdr_bg = colors.HexColor("#f8fafc")
    _row_sep = colors.HexColor("#f1f5f9")
    # Aligné InvoiceLivePreview : th padding 8px 10px ; date / desc resserrés (.colDate + td).
    _pad_tb_h = 6.0
    _pad_tb_body = 6.0
    _pad_lr = 7.5
    _pad_date_r = 1.0
    _pad_desc_l = 3.0
    if show_date_column:
        style_rules = [
            ("BACKGROUND", (0, 0), (-1, 0), _hdr_bg),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, 0), _pad_tb_h),
            ("BOTTOMPADDING", (0, 0), (-1, 0), _pad_tb_h),
            ("LEFTPADDING", (0, 0), (0, 0), _pad_lr),
            ("RIGHTPADDING", (0, 0), (0, 0), _pad_date_r),
            ("LEFTPADDING", (1, 0), (1, 0), _pad_desc_l),
            ("RIGHTPADDING", (1, 0), (1, 0), _pad_lr),
            ("LEFTPADDING", (2, 0), (2, 0), _pad_lr),
            ("RIGHTPADDING", (2, 0), (2, 0), _pad_lr),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, _row_sep),
            ("FONTNAME", (0, 1), (-1, -1), font_name),
            ("FONTSIZE", (0, 1), (-1, -1), FONT_BODY),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
            ("FONTSIZE", (0, 0), (-1, 0), FONT_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("TOPPADDING", (0, 1), (-1, -1), _pad_tb_body),
            ("BOTTOMPADDING", (0, 1), (-1, -1), _pad_tb_body),
            ("LEFTPADDING", (0, 1), (0, -1), _pad_lr),
            ("RIGHTPADDING", (0, 1), (0, -1), _pad_date_r),
            ("LEFTPADDING", (1, 1), (1, -1), _pad_desc_l),
            ("RIGHTPADDING", (1, 1), (1, -1), _pad_lr),
            ("LEFTPADDING", (2, 1), (2, -1), _pad_lr),
            ("RIGHTPADDING", (2, 1), (2, -1), _pad_lr),
        ]
    else:
        style_rules = [
            ("BACKGROUND", (0, 0), (-1, 0), _hdr_bg),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, 0), _pad_tb_h),
            ("BOTTOMPADDING", (0, 0), (-1, 0), _pad_tb_h),
            ("LEFTPADDING", (0, 0), (0, 0), _pad_desc_l),
            ("RIGHTPADDING", (0, 0), (0, 0), _pad_lr),
            ("LEFTPADDING", (1, 0), (1, 0), _pad_lr),
            ("RIGHTPADDING", (1, 0), (1, 0), _pad_lr),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, _row_sep),
            ("FONTNAME", (0, 1), (-1, -1), font_name),
            ("FONTSIZE", (0, 1), (-1, -1), FONT_BODY),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
            ("FONTSIZE", (0, 0), (-1, 0), FONT_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("TOPPADDING", (0, 1), (-1, -1), _pad_tb_body),
            ("BOTTOMPADDING", (0, 1), (-1, -1), _pad_tb_body),
            ("LEFTPADDING", (0, 1), (0, -1), _pad_desc_l),
            ("RIGHTPADDING", (0, 1), (0, -1), _pad_lr),
            ("LEFTPADDING", (1, 1), (1, -1), _pad_lr),
            ("RIGHTPADDING", (1, 1), (1, -1), _pad_lr),
        ]
    n_rows = len(table_data)
    if n_rows > LEVEL_THRESHOLD:
        for r in range(1, n_rows - 1):
            style_rules.append(("LINEBELOW", (0, r), (-1, r), 0.35, _row_sep))
    for r in s2_patient_separator_after_rows:
        style_rules.append(("LINEBELOW", (0, r), (-1, r), 0.15, colors.lightgrey))
    tbl.setStyle(TableStyle(style_rules))
    return (tbl, consolidated)


def _swiss_group_int_str(whole: int) -> str:
    """Groupe milliers avec apostrophe (usage facture CH)."""
    s = str(abs(int(whole)))
    if s == "0":
        return "0"
    parts: list[str] = []
    while len(s) > _SWISS_GROUP_DIGITS:
        parts.insert(0, s[-_SWISS_GROUP_DIGITS:])
        s = s[:-_SWISS_GROUP_DIGITS]
    if s:
        parts.insert(0, s)
    return "'".join(parts)


def _format_chf_pdf(amount: float) -> str:
    """Montant CHF pour PDF : espace insécable après CHF, apostrophe milliers."""
    a = float(amount)
    neg = a < 0
    a = abs(a)
    whole = int(a + _FLOAT_EQ_EPS)
    cents = round((a - whole) * _CHF_CENTS_IN_FRANC + _FLOAT_EQ_EPS)
    if cents >= _CHF_CENTS_IN_FRANC:
        whole += cents // _CHF_CENTS_IN_FRANC
        cents = cents % _CHF_CENTS_IN_FRANC
    ip = _swiss_group_int_str(whole)
    # Espace ASCII (pas insécable) : certains moteurs PDF fusionnent sinon les colonnes à l'affichage.
    core = f"CHF {ip}.{cents:02d}"
    return f"- {core}" if neg else core


def _format_chf_discount_pdf(disc_ht: float) -> str:
    """Ligne remise : montant positif, préfixe moins (aligné facture pro)."""
    a = abs(float(disc_ht))
    whole = int(a + _FLOAT_EQ_EPS)
    cents = round((a - whole) * _CHF_CENTS_IN_FRANC + _FLOAT_EQ_EPS)
    if cents >= _CHF_CENTS_IN_FRANC:
        whole += cents // _CHF_CENTS_IN_FRANC
        cents = cents % _CHF_CENTS_IN_FRANC
    ip = _swiss_group_int_str(whole)
    return f"- CHF {ip}.{cents:02d}"


# Largeur cible colonne montants (Courier) : aligne les decimales a l'affichage.
_PDF_CHF_MONO_CELL_WIDTH = 22
# Tolérance comparaison montants float (ex. net HT vs total sans TVA).
_PDF_CHF_AMOUNT_EQ_EPS = 0.005


def _format_chf_pdf_mono(
    amount: float, *, discount: bool = False, width: int = _PDF_CHF_MONO_CELL_WIDTH
) -> str:
    """Montant CHF pour colonne tableau en Courier : largeur fixe, décimales alignées."""
    a = float(amount)
    if discount:
        a = abs(a)
        neg = False
    else:
        neg = a < 0
        a = abs(a)
    whole = int(a + _FLOAT_EQ_EPS)
    cents = round((a - whole) * _CHF_CENTS_IN_FRANC + _FLOAT_EQ_EPS)
    if cents >= _CHF_CENTS_IN_FRANC:
        whole += cents // _CHF_CENTS_IN_FRANC
        cents = cents % _CHF_CENTS_IN_FRANC
    ip = _swiss_group_int_str(whole)
    inner = f"CHF -{ip}.{cents:02d}" if discount or neg else f"CHF {ip}.{cents:02d}"
    return inner.rjust(width)


def _format_qty_unit_for_custom_pdf(q: Any) -> str:
    """Affichage compacité d'une quantité / durée (évite 2,00 h si entier)."""
    try:
        f = float(q)
    except (TypeError, ValueError):
        return "?"
    if abs(f - round(f)) < _FLOAT_EQ_EPS and abs(f) < _DISPLAY_MAG_MAX:
        return str(round(f))
    s = f"{f:.2f}"
    if s.endswith("0") and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _format_money_two_decimals(d: Any) -> str:
    try:
        return f"{float(d):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _custom_prestation_subline_for_pdf(line: InvoiceLine) -> str | None:
    """Sous-ligne factu : détail « durée / quantité » pour prestations CUSTOM > 0.

    Aligné sur ``InvoiceLivePreview.jsx`` ``customPrestationSubline`` : même formule
    « tarif × durée = HT » (mode temps) et pas de sous-ligne sans ``custom_prestation``.
    """
    if line.type != InvoiceLineType.CUSTOM or line.line_total <= 0:
        return None
    meta = getattr(line, "line_meta", None)
    cp = meta.get("custom_prestation") if isinstance(meta, dict) else None
    if not isinstance(cp, dict):
        return None
    qv = _format_qty_unit_for_custom_pdf(line.qty)
    up = _format_money_two_decimals(line.unit_price)
    tot = _format_money_two_decimals(line.line_total)
    try:
        qty_f = float(line.qty)
        unit_f = float(line.unit_price)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(qty_f) and math.isfinite(unit_f)):
        return None
    mode = cp.get("mode")
    if mode == "time":
        tu = cp.get("time_unit")
        if not tu or tu not in ("min", "h", "d", "mois"):
            return None
        qsym = {"min": "min", "h": "h", "d": "j", "mois": "mois"}
        psym = {"min": "min", "h": "h", "d": "j", "mois": "mois"}
        # Aligné InvoiceLivePreview : « tarif × durée → HT »
        return f"{up} CHF/{psym[tu]} × {qv} {qsym[tu]} → {tot} CHF HT"
    if mode == "quantity":
        return f"{qv} × {up} CHF = {tot} CHF HT"
    return None


def _line_meta_service_date_display_fr(inv_line: Any) -> str:
    """JJ.MM.AAAA ou MM.YYYY si prestation CUSTOM au mois.

    A/R consolidé : plage ``service_date`` / ``service_date_end`` (ou *_iso_*).
    """
    lm = getattr(inv_line, "line_meta", None)
    if not isinstance(lm, dict):
        return ""
    raw = lm.get("service_date") or lm.get("service_date_iso")
    if raw is None:
        return ""
    s = str(raw).strip()[:_ISO_DATE_LEN]
    if len(s) != _ISO_DATE_LEN:
        return ""
    try:
        dp = date.fromisoformat(s)
    except ValueError:
        return ""
    cp = lm.get("custom_prestation")
    if (
        isinstance(cp, dict)
        and cp.get("mode") == "time"
        and cp.get("time_unit") == "mois"
    ):
        return f"{dp.month:02d}.{dp.year}"
    raw_end = lm.get("service_date_end") or lm.get("service_date_iso_end")
    if raw_end is not None:
        s_end = str(raw_end).strip()[:_ISO_DATE_LEN]
        if len(s_end) == _ISO_DATE_LEN:
            try:
                dp_end = date.fromisoformat(s_end)
            except ValueError:
                dp_end = None
            if dp_end is not None and dp_end != dp:
                return (
                    f"{dp.day:02d}.{dp.month:02d}.{dp.year} – "
                    f"{dp_end.day:02d}.{dp_end.month:02d}.{dp_end.year}"
                )
    return f"{dp.day:02d}.{dp.month:02d}.{dp.year}"


def _pdf_billing_period_label_fr(invoice: Any) -> str | None:
    """Même libellé que ``InvoiceLivePreview.billingPeriodLabel`` (« avril 2026 »)."""
    y = getattr(invoice, "period_year", None)
    m = getattr(invoice, "period_month", None)
    if y is None or m is None:
        return None
    try:
        yi, mi = int(y), int(m)
    except (TypeError, ValueError):
        return None
    if not (1 <= mi <= 12):
        return None
    _names = (
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    )
    return f"{_names[mi - 1]} {yi}"


def _pdf_line_detail_date_str(line: Any, invoice: Any) -> str:
    """Texte colonne Date — aligné sur ``InvoiceLivePreview.lineDetailDateLabel``.

    - Trajet / livraison : méta ou date réelle de réservation (``scheduled_time``).
    - CUSTOM : méta ; libellé période « mois année » uniquement si ``custom_prestation.time_unit == mois``.
    """
    if not line:
        return ""
    lm = getattr(line, "line_meta", None)
    if not isinstance(lm, dict):
        lm = {}
    if lm.get("global_discount_line") or lm.get("per_line_discount_line"):
        return ""
    t = getattr(line.type, "value", line.type)
    kind = str(t or "").strip().upper()
    if kind not in ("RIDE", "CUSTOM", "MATERIAL_DELIVERY"):
        return ""
    base = _line_meta_service_date_display_fr(line)
    if base:
        return base
    rid = getattr(line, "reservation_id", None)
    if rid and kind in ("RIDE", "MATERIAL_DELIVERY"):
        try:
            from models.booking import Booking

            bk = Booking.query.get(int(rid))
            if bk is not None:
                st = getattr(bk, "scheduled_time", None)
                if st is not None and hasattr(st, "strftime"):
                    return st.strftime("%d.%m.%Y")
        except Exception:
            pass
    if kind == "CUSTOM":
        cp = lm.get("custom_prestation")
        if (
            isinstance(cp, dict)
            and cp.get("mode") == "time"
            and cp.get("time_unit") == "mois"
        ):
            pl = _pdf_billing_period_label_fr(invoice)
            if pl:
                return pl
    return ""


def _pdf_detail_table_show_date_column(
    invoice: Any,
    consolidated: list[dict[str, Any]],
) -> bool:
    """True si l’aperçu HTML afficherait la colonne Date (≥ une date ou repli période)."""
    for item in consolidated:
        if item.get("date"):
            return True
        for key in ("line", "line1"):
            ln = item.get(key)
            if ln is not None and _pdf_line_detail_date_str(ln, invoice):
                return True
    for line in getattr(invoice, "lines", []) or []:
        if _pdf_line_detail_date_str(line, invoice):
            return True
    return False


def _invoice_preview_chf_amount(amount: float) -> str:
    """Comme le frontend ``formatCurrencyCHF`` : « 67.50 CHF »."""
    return f"{float(amount):.2f} CHF"


def _line_meta_skips_custom_pdf_detail(line_meta: Any) -> bool:
    """Lignes CUSTOM techniques (remise globale / par ligne) : pas dans le détail tableau."""
    if not isinstance(line_meta, dict):
        return False
    return bool(
        line_meta.get("global_discount_line") or line_meta.get("per_line_discount_line")
    )


# Aligné sur edit_draft_invoice : déduction fixe HT saisie dans l'éditeur brouillon.
_META_MANUAL_DISCOUNT_PDF = "manual_discount"


def _custom_line_include_in_s2_detail_table(line: Any) -> bool:
    """True si la ligne CUSTOM doit apparaître dans le tableau date/patient/transport/montant."""
    if line.type != InvoiceLineType.CUSTOM:
        return True
    lt = getattr(line, "line_total", None)
    if lt is None:
        return False
    if lt == 0:
        return False
    meta = line.line_meta if isinstance(line.line_meta, dict) else {}
    if lt < 0:
        return bool(meta.get(_META_MANUAL_DISCOUNT_PDF))
    return not _line_meta_skips_custom_pdf_detail(meta)


# Aligné sur application/invoices/edit_draft_invoice._META_ORIGINAL_LINE_TOTAL
_META_ORIGINAL_LINE_TOTAL_HT = "original_line_total"
_PDF_CATALOG_NET_EPS = Decimal("0.02")


def _decimal_from_meta_original(raw: Any) -> Decimal | None:
    if raw is None or raw == "":
        return None
    try:
        return Decimal(str(raw).replace(",", ".").replace(" ", ""))
    except Exception:
        return None


def _line_catalog_vs_net_ht(line: Any) -> tuple[Decimal, Decimal]:
    """Montant HT avant réduction (méta snapshot) et montant HT facturé pour une ligne."""
    net = (
        line.line_total
        if getattr(line, "line_total", None) is not None
        else Decimal("0")
    )
    meta = getattr(line, "line_meta", None)
    if not isinstance(meta, dict):
        return (net, net)
    cat = _decimal_from_meta_original(meta.get(_META_ORIGINAL_LINE_TOTAL_HT))
    if cat is None:
        return (net, net)
    return (cat, net)


def _consolidated_row_catalog_net(
    item: dict[str, Any],
) -> tuple[Decimal | None, Decimal]:
    """Sommes prix initial vs prix après réduction (remise par ligne) pour une ligne consolidée S2."""
    net_total = Decimal(item.get("amount") or 0)
    lines: list[Any] = []
    if item.get("is_round_trip"):
        for key in ("line1", "line2"):
            ln = item.get(key)
            if ln is not None:
                lines.append(ln)
    else:
        ln = item.get("line")
        if ln is not None:
            lines.append(ln)
    if not lines:
        return (None, net_total)
    cat_sum = Decimal("0")
    net_sum = Decimal("0")
    has_catalog_snap = False
    for ln in lines:
        cat_i, net_i = _line_catalog_vs_net_ht(ln)
        cat_sum += cat_i
        net_sum += net_i
        lm = getattr(ln, "line_meta", None)
        if isinstance(lm, dict) and lm.get(_META_ORIGINAL_LINE_TOTAL_HT) is not None:
            has_catalog_snap = True
    if not has_catalog_snap:
        return (None, net_total)
    out_net = net_total if abs(net_sum - net_total) > _PDF_CATALOG_NET_EPS else net_sum
    if abs(cat_sum - out_net) <= _PDF_CATALOG_NET_EPS:
        return (None, out_net)
    return (cat_sum, out_net)


def _pdf_s2_per_line_discount_suffix_html(
    catalog: Decimal,
    net: Decimal,
    *,
    compact_private_sub: bool = False,
) -> str:
    """Sous le libellé transport / prestation : catalogue → net HT (colonne Montant = montant final seul).

    ``compact_private_sub`` : style ``InvoiceLivePreview`` `.lineSub` (≈12px, ``#64748b``).
    """
    esc_cat = _xml_escape_for_paragraph(f"{Decimal(catalog):.2f}")
    esc_net = _xml_escape_for_paragraph(f"{Decimal(net):.2f}")
    if compact_private_sub:
        return (
            f'<br/><font size="{FONT_SECONDARY}" color="{COLOR_MUTED_PDF}">'
            f"{esc_cat} → {esc_net} CHF HT</font>"
        )
    return f'<br/><font size="7" color="#6b7280">{esc_cat} → {esc_net} CHF HT</font>'


def _compact_private_date_paragraph(date_str: str, font_name: str) -> Any:
    """Colonne Date client privé — police compacte pour tenir sur une ligne dans la colonne étroite."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    display = date_str.strip() if date_str else ""
    if not display:
        display = "—"
    else:
        parts = display.split(".")
        if (
            len(parts) == 3
            and len(parts[0]) == 2
            and len(parts[1]) == 2
            and len(parts[2]) == 4
            and parts[0].isdigit()
            and parts[1].isdigit()
            and parts[2].isdigit()
        ):
            # Insécable avant l’année : évite « 22.04. » / « 2026 » sur deux lignes.
            display = f"{parts[0]}.{parts[1]}.\u00a0{parts[2]}"
    esc = _xml_escape_for_paragraph(display)
    _lead_d = int(round(FONT_BODY * 1.3))
    ps = ParagraphStyle(
        "CompactPrivateDateCell",
        fontName=font_name,
        fontSize=FONT_BODY,
        leading=_lead_d,
        textColor=colors.black,
    )
    # Évite la coupure au dernier caractère si le moteur peut garder la ligne entière.
    return Paragraph(f"<nobr>{esc}</nobr>", ps)


def _pdf_s2_amount_only_paragraph(
    net: Decimal,
    style: Any,
    *,
    is_round_trip: bool,
    ht_column_plain: bool = False,
) -> Any:
    """Colonne Montant / HT : montant facturé.

    ``ht_column_plain`` : comme l'aperçu HTML client privé — uniquement ``12.34`` (sans « CHF »
    qui sinon peut passer à la ligne suivante dans une colonne étroite).

    Ne pas utiliser de balise ``<para>`` dans le fragment HTML : ``Paragraph`` applique déjà
    le paraparser ; des ``<para>`` imbriqués provoquent « unclosed tags ».
    """
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    net_d = Decimal(net)
    raw_txt = f"{net_d:.2f}" if ht_column_plain else f"{net_d:.2f} CHF"
    esc = _xml_escape_for_paragraph(raw_txt)
    inner = f"<nobr>{esc}</nobr>"
    right_style = ParagraphStyle(
        "S2AmountColRight",
        parent=style,
        alignment=TA_RIGHT,
    )
    if is_round_trip:
        return Paragraph(f"<b>{inner}</b>", right_style)
    return Paragraph(inner, right_style)


def _pdf_minimal_amount_only_flowable(net: Decimal, normal_style: Any) -> Any:
    """Colonne Montant (PDF minimal) : montant facturé seul, sans texte de remise."""
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    net_d = Decimal(net)
    esc_one = _xml_escape_for_paragraph(f"{net_d:.2f}")
    right_style = ParagraphStyle(
        "MinimalAmountRight",
        parent=normal_style,
        alignment=TA_RIGHT,
    )
    return Paragraph(esc_one, right_style)


def _minimal_custom_detail_paragraph(
    line: Any,
    normal_style: Any,
    *,
    extra_html_suffix: str = "",
) -> Any:
    """Libellé prestation CUSTOM > 0 pour PDF minimal (aligné sur _build_s2_table / include_non_ride)."""
    from reportlab.platypus import Paragraph

    desc = (getattr(line, "description", None) or "").strip() or "Prestation"
    esc_d = _xml_escape_for_paragraph(desc[:500])
    sub = _custom_prestation_subline_for_pdf(line)
    if sub:
        esc_s = _xml_escape_for_paragraph(sub)
        body = f'{esc_d}<br/><font size="9" color="#64748b">{esc_s}</font>{extra_html_suffix}'
        return Paragraph(body, normal_style)
    return Paragraph(f"{esc_d}{extra_html_suffix}", normal_style)


def _format_global_discount_pdf_label(pct_gd: float) -> str:
    """Libellé réduction globale (PDF) : taux sur le sous-total HT concerné."""
    return f"Réduction globale ({pct_gd:g} %)"


def _sum_positive_billed_lines_excluding_global_discount(
    invoice: "Invoice",
) -> float | None:
    """Somme des HT **positifs** (ce qui apparaît dans le détail), hors ligne de remise globale (CUSTOM < 0).

    Aligne le « Sous-total HT » du bloc totaux avec la somme des montants listés (transports
    + lignes forfait/frais), et évite d'afficher la seule `meta['subtotal_before_ht']` (RIDE
    seuls, avant forfaits) qui ne correspondait pas à la somme des lignes du PDF.
    """
    total = Decimal("0")
    for line in invoice.lines:
        if line.line_total is None:
            continue
        if line.type == InvoiceLineType.CUSTOM and line.line_total < 0:
            continue
        if line.line_total > 0:
            total += line.line_total
    if total > 0:
        return float(total)
    return None


def _build_totals_table(
    invoice: "Invoice",
    is_s2: bool,
    is_third_party: bool,
    font_name: str,
    font_name_bold: str,
    *,
    template: str = "standard",
    reminder_level: int | None = None,
    reminder_fee: Decimal | None = None,
    reminder_total_due: Decimal | None = None,
    reminder_principal: Decimal | float | None = None,
) -> Any:
    """Construit le tableau des totaux (sous-total, TVA, total).

    Deux colonnes compactes (largeur cible = InvoiceLivePreview ``.totals``, ~280px),
    alignées à droite sous le détail — pas de colonnes fantômes pour caler sur la grille du tableau des lignes.

    En mode rappel : mini-table structurée
    « Montant facture initiale » + « Frais de rappel N°X » + « TOTAL À FACTURER ».
    """
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    _tw_l = INVOICE_PREVIEW_TOTALS_LABEL_CM * cm
    _tw_a = INVOICE_PREVIEW_TOTALS_AMOUNT_CM * cm
    _tot_col_widths: list[float] = [_tw_l, _tw_a]

    subtotal = float(invoice.subtotal_amount)
    vat_total = float(invoice.vat_total_amount)
    total = float(invoice.total_amount)
    vat_is_applicable = False
    vat_label_display = "TVA"
    if isinstance(invoice.meta, dict) and "vat" in invoice.meta:
        vat_meta = invoice.meta.get("vat", {})
        vat_is_applicable = bool(vat_meta.get("applicable", False))
        if vat_meta.get("label"):
            vat_label_display = str(vat_meta["label"])
    elif vat_total > 0:
        vat_is_applicable = True

    # Remise globale : méta enregistrée par apply_draft_global_discount (S1, S2, client, clinique).
    gd_meta: dict[str, Any] | None = None
    if isinstance(invoice.meta, dict) and invoice.meta.get("global_discount"):
        gd_meta = cast(dict[str, Any], invoice.meta["global_discount"])

    totals_extra_style_rules: list[tuple[Any, ...]] = []

    is_reminder = reminder_level is not None and reminder_fee is not None
    if is_reminder and reminder_total_due is not None:
        final_total = float(reminder_total_due)
        reminder_fee_float = float(reminder_fee) if reminder_fee is not None else 0.0
        principal_float = (
            float(reminder_principal)
            if reminder_principal is not None
            else (final_total - reminder_fee_float)
        )
    else:
        final_total = total
        reminder_fee_float = 0.0
        principal_float = subtotal

    total_label = "TOTAL À FACTURER :" if is_s2 else "TOTAL :"
    total_amt = _invoice_preview_chf_amount(final_total)

    reminder_fee_label = "Frais de rappel :"
    reminder_fee_amt = f"CHF {reminder_fee_float:.2f}"
    principal_amt = f"CHF {principal_float:.2f}"

    if is_reminder:
        principal_label = "Montant facture initiale :"
        if template == "detailed":
            if is_third_party:
                total_data = [
                    [principal_label, principal_amt],
                    [reminder_fee_label, reminder_fee_amt],
                    [total_label, total_amt],
                ]
            else:
                total_data = [
                    [principal_label, principal_amt],
                    [reminder_fee_label, reminder_fee_amt],
                    [total_label, total_amt],
                ]
        elif is_third_party:
            total_data = [
                [principal_label, principal_amt],
                [reminder_fee_label, reminder_fee_amt],
                [total_label, total_amt],
            ]
        else:
            total_data = [
                [principal_label, principal_amt],
                [reminder_fee_label, reminder_fee_amt],
                [total_label, total_amt],
            ]
        col_widths = list(_tot_col_widths)
    elif template == "detailed":
        if is_third_party:
            if gd_meta is not None and not is_reminder:
                _detail_sum = _sum_positive_billed_lines_excluding_global_discount(
                    invoice
                )
                gross_ht = (
                    _detail_sum
                    if _detail_sum is not None
                    else float(gd_meta.get("subtotal_before_ht", subtotal))
                )
                disc_ht = float(gd_meta.get("amount_ht", 0))
                pct_gd = float(gd_meta.get("percent", 0))
                note_gd = str(gd_meta.get("note") or "").strip()
                disc_label = _format_global_discount_pdf_label(pct_gd)
                total_data = [
                    [
                        "Sous-total HT (avant réduction globale)",
                        _invoice_preview_chf_amount(gross_ht),
                    ],
                    [disc_label, _format_chf_discount_pdf(disc_ht)],
                ]
                if note_gd:
                    total_data.append(
                        [_pdf_note_global_discount_for_totals_table(note_gd), ""]
                    )
                    totals_extra_style_rules.extend(
                        [
                            ("SPAN", (0, 2), (1, 2)),
                            ("ALIGN", (0, 2), (1, 2), "LEFT"),
                            ("FONTSIZE", (0, 2), (1, 2), 8),
                            ("TEXTCOLOR", (0, 2), (1, 2), colors.HexColor("#4b5563")),
                            ("FONTNAME", (0, 2), (1, 2), font_name),
                        ]
                    )
                if vat_is_applicable:
                    total_data.extend(
                        [
                            [
                                "Total HT après réduction globale",
                                _invoice_preview_chf_amount(subtotal),
                            ],
                            [
                                f"{vat_label_display} :",
                                _invoice_preview_chf_amount(vat_total),
                            ],
                            [total_label, _invoice_preview_chf_amount(total)],
                        ]
                    )
                else:
                    total_data.extend(
                        [
                            [
                                "Total HT après réduction globale",
                                _invoice_preview_chf_amount(subtotal),
                            ],
                            [total_label, _invoice_preview_chf_amount(total)],
                        ]
                    )
            elif vat_is_applicable:
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [
                        f"{vat_label_display} :",
                        _invoice_preview_chf_amount(vat_total),
                    ],
                    [total_label, _invoice_preview_chf_amount(total)],
                ]
            # Facture clinique S2 (tierce) : même pied que InvoiceLivePreview — Sous-total HT puis TOTAL À FACTURER.
            elif is_s2:
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [total_label, total_amt],
                ]
            else:
                # Tierce sans TVA ni S2 (ex. facturation OPAD) : même pied que les autres — sous-total + total.
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [total_label, total_amt],
                ]
            col_widths = list(_tot_col_widths)
        else:
            if gd_meta is not None and not is_reminder:
                # Aligné InvoiceLivePreview : encadré au-dessus du tableau des totaux ;
                # pied = sous-total HT et TOTAL (nets après remise globale).
                if vat_is_applicable:
                    total_data = [
                        ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                        [
                            f"{vat_label_display} :",
                            _invoice_preview_chf_amount(vat_total),
                        ],
                        [total_label, _invoice_preview_chf_amount(total)],
                    ]
                else:
                    total_data = [
                        ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                        [total_label, _invoice_preview_chf_amount(total)],
                    ]
            elif vat_is_applicable:
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [
                        f"{vat_label_display} :",
                        _invoice_preview_chf_amount(vat_total),
                    ],
                    [total_label, _invoice_preview_chf_amount(total)],
                ]
            else:
                # Aligné InvoiceLivePreview : Sous-total HT + TOTAL (montants « 304.00 CHF »).
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [total_label, _invoice_preview_chf_amount(total)],
                ]
            col_widths = list(_tot_col_widths)
    elif is_third_party:
        if gd_meta is not None and not is_reminder:
            _detail_sum = _sum_positive_billed_lines_excluding_global_discount(invoice)
            gross_ht = (
                _detail_sum
                if _detail_sum is not None
                else float(gd_meta.get("subtotal_before_ht", subtotal))
            )
            disc_ht = float(gd_meta.get("amount_ht", 0))
            pct_gd = float(gd_meta.get("percent", 0))
            note_gd = str(gd_meta.get("note") or "").strip()
            disc_label = _format_global_discount_pdf_label(pct_gd)
            total_data = [
                [
                    "Sous-total HT (avant réduction globale)",
                    _invoice_preview_chf_amount(gross_ht),
                ],
                [disc_label, _format_chf_discount_pdf(disc_ht)],
            ]
            if note_gd:
                total_data.append(
                    [_pdf_note_global_discount_for_totals_table(note_gd), ""]
                )
                _note_row = len(total_data) - 1
                totals_extra_style_rules.extend(
                    [
                        ("SPAN", (0, _note_row), (1, _note_row)),
                        ("ALIGN", (0, _note_row), (1, _note_row), "LEFT"),
                        ("FONTSIZE", (0, _note_row), (1, _note_row), 8),
                        (
                            "TEXTCOLOR",
                            (0, _note_row),
                            (1, _note_row),
                            colors.HexColor("#4b5563"),
                        ),
                        ("FONTNAME", (0, _note_row), (1, _note_row), font_name),
                    ]
                )
            if vat_is_applicable:
                total_data.extend(
                    [
                        [
                            "Total HT après réduction globale",
                            _invoice_preview_chf_amount(subtotal),
                        ],
                        [
                            f"{vat_label_display} :",
                            _invoice_preview_chf_amount(vat_total),
                        ],
                        [total_label, _invoice_preview_chf_amount(total)],
                    ]
                )
            else:
                total_data.extend(
                    [
                        [
                            "Total HT après réduction globale",
                            _invoice_preview_chf_amount(subtotal),
                        ],
                        [total_label, _invoice_preview_chf_amount(total)],
                    ]
                )
        elif vat_is_applicable:
            total_data = [
                ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                [f"{vat_label_display} :", _invoice_preview_chf_amount(vat_total)],
                [total_label, _invoice_preview_chf_amount(total)],
            ]
        # Facture clinique S2 : InvoiceLivePreview affiche toujours « Sous-total HT » puis le total.
        elif is_s2:
            total_data = [
                ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                [total_label, total_amt],
            ]
        else:
            # Tierce standard sans TVA ni S2 : sous-total HT + total (aligné factures client / aperçu).
            total_data = [
                ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                [total_label, total_amt],
            ]
        col_widths = list(_tot_col_widths)
    else:
        if gd_meta is not None and not is_reminder:
            if vat_is_applicable:
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [
                        f"{vat_label_display} :",
                        _invoice_preview_chf_amount(vat_total),
                    ],
                    [total_label, _invoice_preview_chf_amount(total)],
                ]
            else:
                total_data = [
                    ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                    [total_label, _invoice_preview_chf_amount(total)],
                ]
        elif vat_is_applicable:
            total_data = [
                ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                [f"{vat_label_display} :", _invoice_preview_chf_amount(vat_total)],
                [total_label, _invoice_preview_chf_amount(total)],
            ]
        else:
            # Aligné InvoiceLivePreview : Sous-total HT + TOTAL (montants « 304.00 CHF »).
            total_data = [
                ["Sous-total HT", _invoice_preview_chf_amount(subtotal)],
                [total_label, _invoice_preview_chf_amount(total)],
            ]
        col_widths = list(_tot_col_widths)

    total_table = Table(total_data, colWidths=col_widths)
    # Aligner la table des totaux sur le bord droit du frame pour que
    # la colonne montant soit exactement sous la colonne des montants du détail.
    total_table.hAlign = "RIGHT"
    # Tableau strictement à 2 colonnes (libellé | montant), largeur ≈ `.totals` HTML (280px).
    amount_col_idx = 1
    style_rules: list[tuple[Any, ...]] = [
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), FONT_BODY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]
    style_rules.extend(totals_extra_style_rules)
    style_rules.append(
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#e2e8f0"))
    )
    style_rules.append(("TOPPADDING", (0, -1), (-1, -1), 8))
    # Aligner strictement à droite la dernière ligne (TOTAL À FACTURER)
    # pour caler le dernier chiffre sur l'axe X de la colonne montant.
    if is_s2:
        style_rules.extend(
            [
                ("RIGHTPADDING", (amount_col_idx, -1), (amount_col_idx, -1), 0),
                ("LEFTPADDING", (amount_col_idx, -1), (amount_col_idx, -1), 0),
            ]
        )
    # Lignes intermédiaires corps ; dernière ligne (TOTAL) en gras.
    if len(total_data) > 1:
        style_rules.extend(
            [
                ("FONTSIZE", (0, 0), (-1, -2), FONT_BODY),
                ("FONTNAME", (0, 0), (-1, -2), font_name),
                ("FONTNAME", (0, -1), (-1, -1), font_name_bold),
                ("FONTSIZE", (0, -1), (-1, -1), FONT_TOTAL),
            ]
        )
    else:
        style_rules.append(("FONTNAME", (0, 0), (-1, -1), font_name_bold))
        style_rules.append(("FONTSIZE", (0, 0), (-1, -1), FONT_TOTAL))
    total_table.setStyle(TableStyle(style_rules))
    return total_table


class PDFService:
    """Service pour la génération de PDF de factures et rappels."""

    def __init__(self):
        super().__init__()
        from flask import current_app

        self.qrbill_service = QRBillService()
        # ✅ Chemin correct: /app/uploads (pas /app/services/uploads)
        self.uploads_dir = Path(current_app.config.get("UPLOAD_FOLDER", "/app/uploads"))
        self.invoices_dir = Path(self.uploads_dir, "invoices")

        # Créer les dossiers s'ils n'existent pas
        self.invoices_dir.mkdir(parents=True, exist_ok=True)

        # Builder pour templates HTML
        self.template_builder = InvoiceTemplateBuilder()

    def _get_company_address_for_pdf(self, company):
        """Récupère l'adresse de l'entreprise depuis le profil de facturation.

        Args:
            company: Instance de Company

        Returns:
            str: Adresse formatée pour affichage PDF
        """
        from services.billing import BillingProfileService

        # Essayer de récupérer le profil
        profile = BillingProfileService.get_by_company_id(company.id)

        if profile:
            # Utiliser l'adresse du profil (source unique)
            # Si building_number est vide, street_name contient déjà l'adresse complète
            if profile.building_number and profile.building_number.strip():
                address_line1 = (
                    f"{profile.street_name} {profile.building_number}".strip()
                )
            else:
                address_line1 = profile.street_name or ""
            address_line2 = f"{profile.postal_code} {profile.city}"
            country = profile.country_code
            return f"{address_line1}<br/>{address_line2} {country}"

        # Fallback : utiliser domicile_address (pas company.address qui est l'adresse opérationnelle)
        if company.domicile_address_line1:
            address_parts = [company.domicile_address_line1]
            if company.domicile_zip and company.domicile_city:
                address_parts.append(f"{company.domicile_zip} {company.domicile_city}")
            return "<br/>".join(address_parts)

        # Dernier fallback
        return company.address or "[Adresse non configurée]"

    def _test_builder_extraction(self, invoice):
        """Méthode de test pour valider l'extraction de données par le builder.

        Args:
            invoice: Instance d'Invoice

        Returns:
            InvoiceData | None: Données extraites ou None
        """
        return self.template_builder.extract_invoice_data(invoice)

    def generate_invoice_pdf(
        self,
        invoice,
        *,
        force_regenerate: bool = False,
        force_bypass_locked: bool = False,
    ):
        """Génère le PDF d'une facture.

        Args:
            invoice: La facture pour laquelle générer le PDF
            force_regenerate: Si True, régénère même si pdf_url existe déjà
                (utilisé par l'endpoint regenerate-pdf)
            force_bypass_locked: Si True, régénère même pour factures non éditables (payée, annulée)
                — réservé aux corrections admin manuelles.

        ⚠️ PROTECTION IMMUTABILITÉ:
        Ne régénère pas le contenu PDF si la facture n'est pas éditable (payée, annulée),
        sauf ``force_bypass_locked=True``. Réutilise ``invoice.pdf_url`` si présent.
        Si ``invoice.pdf_url`` existe déjà et ``force_regenerate=False``, retourne l'existant.
        """
        from application.invoices.edit_draft_invoice import invoice_allows_line_editing

        # ✅ Garde-fou 2: Log explicite pour diagnostic (invoice_id, force_regenerate, action)
        invoice_id = getattr(invoice, "id", None)
        has_existing_pdf_url = bool(getattr(invoice, "pdf_url", None))
        app_logger.info(
            "[PDF] generate_invoice_pdf entry: invoice_id=%s, force_regenerate=%s, force_bypass_locked=%s, has_existing_pdf_url=%s",
            invoice_id,
            force_regenerate,
            force_bypass_locked,
            has_existing_pdf_url,
        )

        # Aligné sur l'édition des lignes : pas de nouvelle génération si payée / annulée (sauf bypass admin).
        if not invoice_allows_line_editing(invoice) and not force_bypass_locked:
            app_logger.warning(
                "[PDF PROTECTION] Facture non éditable: invoice_id=%s, status=%s, pdf_url=%s. Action=SKIP_LOCKED",
                invoice.id,
                invoice.status.value,
                invoice.pdf_url,
            )
            # Retourner le PDF existant si disponible, sinon None
            return invoice.pdf_url if invoice.pdf_url else None

        # ✅ Si un PDF existe déjà et qu'on ne force pas la régénération, retourner l'existant
        if invoice.pdf_url and not force_regenerate:
            app_logger.info(
                "[PDF] Facture %s a déjà un PDF (%s). Action=SKIP_REUSE_EXISTING",
                invoice.id,
                invoice.pdf_url,
            )
            return invoice.pdf_url

        app_logger.info(
            "[PDF] Facture %s: régénération demandée. Action=REGENERATE",
            invoice.id,
        )

        try:
            # ✅ IMPORTANT: Forcer le rechargement depuis la DB pour avoir les données à jour
            # (adresses, courses, montants, format) - critique pour régénération PDF
            from ext import db

            db.session.expire_all()

            # Charger la facture avec toutes les relations (données fraîches)
            invoice = (
                Invoice.query.options(
                    joinedload(Invoice.company),
                    joinedload(Invoice.client).joinedload(Client.user),
                    selectinload(Invoice.lines),
                    joinedload(Invoice.payments),
                    joinedload(Invoice.billing_party),
                    joinedload(Invoice.billed_to_company),
                )
                .filter_by(id=invoice.id)
                .first()
            )

            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # Factures éditables : même source de vérité que l'éditeur (Σ TTC lignes après canonique HT+TVA).
            # Sinon le pied de PDF peut rester obsolète après modification des lignes (envoyée, etc.).
            from application.invoices.edit_draft_invoice import (
                _recompute_totals_from_lines,
            )

            if invoice.lines and invoice_allows_line_editing(invoice):
                db.session.expire(invoice, ["lines"])
                _recompute_totals_from_lines(invoice)
                db.session.flush()

            # ════════════════════════════════════════════════════════════════════
            # FILET DE SÉCURITÉ: Recalculer les totaux si incohérents
            # ════════════════════════════════════════════════════════════════════
            # Protège contre les factures avec totaux à 0 alors que des lignes existent.
            from infrastructure.invoices.invoice_calculator import (
                recompute_invoice_totals,
            )

            if invoice.lines and (invoice.total_amount or Decimal("0.00")) == Decimal(
                "0.00"
            ):
                app_logger.warning(
                    "[PDF] Facture %s a des lignes mais total_amount=0. Recalcul automatique...",
                    invoice.id,
                )
                recompute_invoice_totals(invoice.id, commit=True)
                # Recharger après recalcul
                db.session.refresh(invoice)

            # ✅ Forcer le rechargement des relations profondes pour avoir les données à jour
            # (adresses client, adresses entreprise, billing party, etc.)
            if invoice.client:
                db.session.refresh(invoice.client)
                if invoice.client.user:
                    db.session.refresh(invoice.client.user)
            if invoice.company:
                db.session.refresh(invoice.company)
            if hasattr(invoice, "billing_party") and invoice.billing_party:
                db.session.refresh(invoice.billing_party)
            if hasattr(invoice, "billed_to_company") and invoice.billed_to_company:
                db.session.refresh(invoice.billed_to_company)

            app_logger.info(
                "[PDF] Facture %s: données rechargées depuis la DB (client, company, billing_party)",
                invoice.id,
            )

            # ✅ Monitoring performance : mesurer le temps de génération
            start_time = perf_counter()
            pdf_content, nb_rows = self._create_invoice_pdf_content(invoice)
            generation_ms = int((perf_counter() - start_time) * 1000)

            # ✅ Déterminer billing_type pour métriques Prometheus
            strategy_value = None
            try:
                bs = getattr(invoice, "billing_strategy", None)
                if bs is None:
                    strategy_value = None
                else:
                    strategy_value = bs.value if hasattr(bs, "value") else str(bs)
            except Exception:
                strategy_value = None
            is_s2 = strategy_value == "s2_clinic_monthly"
            is_third_party = bool(
                getattr(invoice, "billing_party_id", None)
                or (
                    invoice.bill_to_client_id
                    and invoice.bill_to_client_id != invoice.client_id
                )
                or is_s2
            )
            if is_s2:
                billing_type = "clinic"
            elif is_third_party:
                billing_type = "partner"
            else:
                billing_type = "client"

            # ✅ Métriques Prometheus
            try:
                from services.monitoring.prometheus import observe_invoice_pdf_perf

                observe_invoice_pdf_perf(
                    pdf_kind="invoice",
                    billing_type=billing_type,
                    template_version=TEMPLATE_VERSION,
                    nb_rows=nb_rows,
                    duration_ms=generation_ms,
                    warning_threshold_rows=PERF_WARNING_ROWS_THRESHOLD,
                    warning_threshold_ms=PERF_WARNING_MS_THRESHOLD,
                )
            except ImportError:
                pass  # Prometheus non disponible
            except Exception as e:
                app_logger.debug("[PDF] Error tracking Prometheus metrics: %s", e)

            # ✅ Logging performance
            invoice_id = invoice.id

            # WARNING si seuils dépassés
            if (
                nb_rows > PERF_WARNING_ROWS_THRESHOLD
                or generation_ms > PERF_WARNING_MS_THRESHOLD
            ):
                warnings = []
                if nb_rows > PERF_WARNING_ROWS_THRESHOLD:
                    warnings.append(f"rows={nb_rows}>{PERF_WARNING_ROWS_THRESHOLD}")
                if generation_ms > PERF_WARNING_MS_THRESHOLD:
                    warnings.append(
                        f"time={generation_ms}ms>{PERF_WARNING_MS_THRESHOLD}ms"
                    )
                app_logger.warning(
                    (
                        "InvoicePDF slow/large (invoice_id=%s, nb_rows=%s, generation_ms=%s, "
                        "template_version=%s, thresholds_exceeded=[%s])"
                    ),
                    invoice_id,
                    nb_rows,
                    generation_ms,
                    TEMPLATE_VERSION,
                    ", ".join(warnings),
                )
            else:
                app_logger.info(
                    (
                        "InvoicePDF generated (invoice_id=%s, nb_rows=%s, generation_ms=%s, "
                        "template_version=%s)"
                    ),
                    invoice_id,
                    nb_rows,
                    generation_ms,
                    TEMPLATE_VERSION,
                )

            # Sauvegarder le fichier
            # ✅ Garde-fou 1: Quand force_regenerate=True, inclure un UUID pour garantir
            # une URL unique (évite cache/proxy qui resservirait l'ancien fichier)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            unique_suffix = f"{ts}_{uuid.uuid4().hex[:8]}" if force_regenerate else ts
            filename = f"invoice_{invoice.invoice_number}_{unique_suffix}.pdf"
            filepath = Path(self.invoices_dir, filename)

            pdf_bytes: bytes = pdf_content
            with filepath.open("wb") as f:
                f.write(pdf_bytes)

            # ✅ URL dynamique depuis config (127.0.0.1 en dev évite IPv6 localhost + ERR_CONNECTION_RESET)
            pdf_base_url = current_app.config.get(
                "PDF_BASE_URL", "http://127.0.0.1:5000"
            )
            uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")

            pdf_url = f"{pdf_base_url}{uploads_base}/invoices/{filename}"

            app_logger.info(
                "[PDF] PDF written to: invoice_id=%s, filename=%s, pdf_url=%s",
                invoice.id,
                filename,
                pdf_url,
            )
            return pdf_url

        except Exception as e:
            app_logger.error(
                "Erreur lors de la génération du PDF de facture: %s", str(e)
            )
            raise

    def generate_reminder_pdf(self, invoice, level, reminder=None):
        """Génère le PDF d'un rappel consolidé.

        Args:
            invoice: Facture principale (INTOUCHABLE - ne sera jamais modifiée)
            level: Niveau du rappel (1, 2, 3)
            reminder: InvoiceReminder avec les montants consolidés (optionnel pour rétrocompatibilité)

        Returns:
            str: URL du PDF du rappel (reminder_*.pdf, distinct de invoice.pdf_url)

        Important:
            - Ne modifie JAMAIS invoice.pdf_url, invoice.total_amount, invoice.lines
            - Génère un PDF séparé avec filename unique (reminder_*)
            - Le PDF est stocké dans reminder.pdf_url, pas invoice.pdf_url
        """
        import os

        REMINDER_DEBUG = os.getenv("REMINDER_DEBUG", "0") == "1"

        # ✅ Capturer l'état initial pour vérification (même si REMINDER_DEBUG est False)
        invoice_before = {
            "id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "pdf_url": invoice.pdf_url,
            "total_amount": float(invoice.total_amount) if invoice.total_amount else 0,
            "balance_due": float(invoice.balance_due) if invoice.balance_due else 0,
            "due_date": invoice.due_date.isoformat() if invoice.due_date else None,
        }

        try:
            # ✅ Logs de debug pour tracer les modifications
            if REMINDER_DEBUG:
                app_logger.info(
                    (
                        "[REMINDER_DEBUG] generate_reminder_pdf START: invoice_id=%s, level=%s, "
                        "invoice.pdf_url (avant)=%s, invoice.total (avant)=%s"
                    ),
                    invoice.id,
                    level,
                    invoice_before["pdf_url"],
                    invoice_before["total_amount"],
                )

            # Charger la facture avec toutes les relations
            invoice = (
                Invoice.query.options(
                    joinedload(Invoice.company),
                    joinedload(Invoice.client).joinedload(Client.user),
                    selectinload(Invoice.lines),
                    joinedload(Invoice.payments),
                    joinedload(Invoice.reminders),
                    joinedload(Invoice.billing_party),
                    joinedload(Invoice.billed_to_company),
                )
                .filter_by(id=invoice.id)
                .first()
            )

            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # ✅ Monitoring performance : mesurer le temps de génération
            # ✅ IMPORTANT: Réutiliser le template facture avec paramètres reminder
            start_time = perf_counter()
            reminder_fee = Decimal("0.00")
            reminder_total_due = invoice.total_amount
            reminder_principal = invoice.total_amount
            if reminder:
                reminder_fee = reminder.reminder_fee_amount or Decimal("0.00")
                reminder_total_due = reminder.total_due or invoice.total_amount
                reminder_principal = reminder.principal_amount or (
                    reminder_total_due - reminder_fee
                )
            elif invoice.reminder_fee_amount:
                reminder_fee = invoice.reminder_fee_amount
                reminder_total_due = invoice.balance_due
                reminder_principal = (invoice.balance_due or Decimal("0")) - (
                    invoice.reminder_fee_amount or Decimal("0")
                )

            if REMINDER_DEBUG:
                app_logger.info(
                    (
                        "[REMINDER_DEBUG] Génération PDF rappel avec template facture: "
                        "invoice_id=%s, level=%s, reminder_fee=%s, reminder_principal=%s, reminder_total_due=%s"
                    ),
                    invoice.id,
                    level,
                    float(reminder_fee),
                    float(reminder_principal),
                    float(reminder_total_due),
                )

            # ✅ Réutiliser le template facture avec paramètres reminder
            pdf_content, nb_rows = self._create_invoice_pdf_content(
                invoice,
                reminder_level=level,
                reminder_fee=reminder_fee,
                reminder_total_due=reminder_total_due,
                reminder_principal=reminder_principal,
            )
            generation_ms = int((perf_counter() - start_time) * 1000)

            # ✅ Déterminer billing_type pour métriques Prometheus
            strategy_value = None
            try:
                bs = getattr(invoice, "billing_strategy", None)
                if bs is None:
                    strategy_value = None
                else:
                    strategy_value = bs.value if hasattr(bs, "value") else str(bs)
            except Exception:
                strategy_value = None
            is_s2 = strategy_value == "s2_clinic_monthly"
            is_third_party = bool(
                getattr(invoice, "billing_party_id", None)
                or (
                    invoice.bill_to_client_id
                    and invoice.bill_to_client_id != invoice.client_id
                )
                or is_s2
            )
            if is_s2:
                billing_type = "clinic"
            elif is_third_party:
                billing_type = "partner"
            else:
                billing_type = "client"

            # ✅ Métriques Prometheus
            try:
                from services.monitoring.prometheus import observe_invoice_pdf_perf

                observe_invoice_pdf_perf(
                    pdf_kind="reminder",
                    billing_type=billing_type,
                    template_version=TEMPLATE_VERSION,
                    nb_rows=nb_rows,
                    duration_ms=generation_ms,
                    warning_threshold_rows=PERF_WARNING_ROWS_THRESHOLD,
                    warning_threshold_ms=PERF_WARNING_MS_THRESHOLD,
                )
            except ImportError:
                pass  # Prometheus non disponible
            except Exception as e:
                app_logger.debug("[PDF] Error tracking Prometheus metrics: %s", e)

            # ✅ Sauvegarder le fichier avec un filename unique (reminder_*)
            # IMPORTANT: Ce PDF est distinct de invoice.pdf_url (facture initiale)
            # Format: reminder_{invoice_number}_L{level}_{reminder_id}.pdf
            # L'ID du reminder garantit l'unicité même en cas de création simultanée
            reminder_id = (
                reminder.id
                if reminder and hasattr(reminder, "id") and reminder.id
                else None
            )
            if reminder_id:
                # Utiliser l'ID du reminder pour garantir l'unicité
                filename = (
                    f"reminder_{invoice.invoice_number}_L{level}_{reminder_id}.pdf"
                )
            else:
                # Fallback: utiliser timestamp + UUID si reminder_id n'est pas disponible
                import uuid

                unique_suffix = f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                filename = (
                    f"reminder_{invoice.invoice_number}_L{level}_{unique_suffix}.pdf"
                )
                app_logger.warning(
                    (
                        "[PDF] reminder_id non disponible pour facture %s, niveau %s. "
                        "Utilisation d'un suffixe UUID pour garantir l'unicité."
                    ),
                    invoice.id,
                    level,
                )
            filepath = Path(self.invoices_dir, filename)

            pdf_bytes: bytes = pdf_content
            with filepath.open("wb") as f:
                f.write(pdf_bytes)

            # ✅ URL dynamique (127.0.0.1 en dev évite IPv6 localhost + ERR_CONNECTION_RESET)
            pdf_base_url = current_app.config.get(
                "PDF_BASE_URL", "http://127.0.0.1:5000"
            )
            uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
            pdf_url = f"{pdf_base_url}{uploads_base}/invoices/{filename}"

            # ✅ Logs de debug pour vérifier que invoice n'a pas été modifié
            if REMINDER_DEBUG:
                invoice_after = {
                    "pdf_url": invoice.pdf_url,
                    "total_amount": float(invoice.total_amount)
                    if invoice.total_amount
                    else 0,
                    "balance_due": float(invoice.balance_due)
                    if invoice.balance_due
                    else 0,
                    "due_date": invoice.due_date.isoformat()
                    if invoice.due_date
                    else None,
                }
                app_logger.info(
                    (
                        "[REMINDER_DEBUG] generate_reminder_pdf END: invoice_id=%s, level=%s, "
                        "reminder_pdf_url=%s, invoice.pdf_url (INCHANGÉ)=%s, "
                        "invoice.total (INCHANGÉ)=%s, filepath=%s"
                    ),
                    invoice.id,
                    level,
                    pdf_url,
                    invoice_after["pdf_url"],
                    invoice_after["total_amount"],
                    filepath,
                )
                # Vérification de sécurité: invoice.pdf_url ne doit pas avoir changé
                if invoice_after["pdf_url"] != invoice_before.get("pdf_url"):
                    app_logger.error(
                        (
                            "[REMINDER_DEBUG] ⚠️ BUG DÉTECTÉ: invoice.pdf_url a changé ! "
                            "Avant=%s, Après=%s"
                        ),
                        invoice_before.get("pdf_url"),
                        invoice_after["pdf_url"],
                    )

            app_logger.info(
                "PDF de rappel généré: %s (facture initiale inchangée: %s)",
                pdf_url,
                invoice.pdf_url,
            )
            return pdf_url

        except Exception as e:
            app_logger.error(
                "Erreur lors de la génération du PDF de rappel: %s", str(e)
            )
            raise

    def _create_invoice_pdf_content(
        self,
        invoice,
        reminder_level: int | None = None,
        reminder_fee: Decimal | None = None,
        reminder_total_due: Decimal | None = None,
        reminder_principal: Decimal | float | None = None,
    ):
        """Crée le contenu PDF d'une facture selon la variante de template
        sélectionnée.

        Construit un reminder_ctx unique (is_reminder, display_reminder_level, etc.)
        avant tout rendu, puis le passe aux templates. Évite tout recalcul dans
        standard/detailed/minimal.

        Returns:
            tuple[bytes, int]: (contenu PDF, nombre de lignes après regroupement)
        """
        level = reminder_level
        is_reminder = bool(level)
        _reminder_labels = {
            1: "RAPPEL DE PAIEMENT · 1er avis",
            2: "RAPPEL DE PAIEMENT · 2e avis",
            3: "DERNIER RAPPEL DE PAIEMENT",
        }
        display_reminder_level = (
            _reminder_labels.get(level, f"RAPPEL N°{level}") if level else None
        )
        reminder_ctx = {
            "is_reminder": is_reminder,
            "display_reminder_level": display_reminder_level,
            "reminder_level": level,
            "reminder_fee": reminder_fee,
            "reminder_principal": reminder_principal,
            "reminder_total_due": reminder_total_due,
        }

        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=invoice.company_id
        ).first()
        template_variant = "standard"
        if billing_settings and billing_settings.pdf_template_variant:
            template_variant = billing_settings.pdf_template_variant.lower()

        bookings_by_id = _bookings_by_reservation_ids_for_pdf(invoice)

        if template_variant == "minimal":
            return self._create_minimal_invoice_pdf(
                invoice,
                billing_settings,
                reminder_ctx,
                bookings_by_id=bookings_by_id,
            )
        if template_variant == "detailed":
            return self._create_detailed_invoice_pdf(
                invoice,
                billing_settings,
                reminder_ctx,
                bookings_by_id=bookings_by_id,
            )
        return self._create_standard_invoice_pdf(
            invoice,
            billing_settings,
            reminder_ctx,
            bookings_by_id=bookings_by_id,
        )

    def _create_standard_invoice_pdf(
        self,
        invoice,
        billing_settings,
        reminder_ctx: dict[str, Any],
        *,
        bookings_by_id: dict[int, Any],
    ):
        """Crée le contenu PDF d'une facture avec le template standard.

        reminder_ctx: is_reminder, display_reminder_level, reminder_level, reminder_fee,
            reminder_principal, reminder_total_due (calculés une seule fois dans
            _create_invoice_pdf_content).
        """
        # Import ici pour éviter les problèmes de dépendances circulaires
        # ruff: noqa: I001
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.pagesizes import (
            A4,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        font_name, font_name_bold = _ensure_dejavu_pdf_fonts()

        buffer = BytesIO()

        # Styles basés sur le design de référence
        styles = getSampleStyleSheet()

        _body_leading = int(round(FONT_BODY * 1.3))
        # Style pour le texte normal (leftIndent=0 pour alignement marge gauche)
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_body_leading,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=6,
            fontName=font_name,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
        )

        # Style pour le texte centré (pied de page)
        centered_style = ParagraphStyle(
            "Centered",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_body_leading,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName=font_name,
        )
        s2_main_style = ParagraphStyle(
            "S2Main",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_body_leading,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
            fontName=font_name,
        )

        # Contenu
        story = []

        # === EN-TÊTE AVEC LOGO ET INFORMATIONS ENTREPRISE ===
        company = invoice.company

        # Logo de l'entreprise
        logo_img = None
        logo_path = None
        logo_width = 0.0
        logo_height = 0.0
        if hasattr(company, "logo_url") and company.logo_url:
            try:
                logo_url = company.logo_url.strip()

                # Vérifier si c'est une URL externe (http/https)
                if logo_url.startswith(("http://", "https://")):
                    # Pour les URLs externes, on ne peut pas les charger
                    # directement dans ReportLab
                    # On pourrait télécharger l'image, mais pour l'instant on ignore
                    app_logger.info(
                        "Logo externe détecté (non supporté pour PDF): %s", logo_url
                    )
                    logo_path = None
                else:
                    # Logo stocké localement : nettoyer le chemin
                    # Format attendu : /uploads/company_logos/company_{id}.{ext}
                    logo_url_clean = logo_url.lstrip("/")
                    if logo_url_clean.startswith("uploads/"):
                        logo_url_clean = logo_url_clean[8:]  # Supprimer 'uploads/'

                    # Construire le chemin absolu
                    # ✅ Chemin correct: /app/uploads
                    from flask import current_app

                    uploads_dir = Path(
                        current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                    )
                    logo_path = uploads_dir / logo_url_clean

                if logo_path and Path(logo_path).exists():
                    max_width_pt = 595 * 0.24
                    logo_img, logo_width, logo_height = _load_logo_ratio_safe(
                        logo_path, max_width_pt
                    )
            except Exception as e:
                app_logger.warning("Impossible de charger le logo: %s", e)

        # Informations de l'entreprise
        company_name = company.name or "[Nom entreprise non configuré]"
        company_address = self._get_company_address_for_pdf(company)
        company_phone = company.contact_phone or "[Téléphone non configuré]"
        company_email = (
            company.billing_email or company.contact_email or "[Email non configuré]"
        )
        company_uid = company.uid_ide or "[IDE/UID non configuré]"
        # ✅ Statut TVA : afficher uniquement si assujetti (pas de mention si non assujetti)
        vat_status_text = ""
        if billing_settings and billing_settings.vat_applicable:
            vat_number = billing_settings.vat_number or ""
            if vat_number:
                vat_status_text = f"N° TVA : {vat_number}"
            else:
                vat_status_text = f"TVA {billing_settings.vat_rate or 7.7}% incluse"

        # === EN-TÊTE : ENTREPRISE (gauche) | DESTINATAIRE (droite) — convention comptable ---
        recipient_para, _ = _build_recipient_block_flowable(
            invoice,
            normal_style,
            bookings_by_id=bookings_by_id,
            name_font_size=FONT_CLIENT_NAME,
            addr_font_size=FONT_BODY,
        )
        recipient_top_padding_mm = 25.0  # destinataire légèrement plus bas
        recipient_left_padding_mm = 15.0  # déplace le bloc destinataire vers la droite (pas d'espace volé à l'expéditeur)
        dest_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
        page_width_pt = A4[0]
        usable_width_pt = (
            page_width_pt
            - INVOICE_PAGE_LEFT_MARGIN_CM * cm
            - INVOICE_PAGE_RIGHT_MARGIN_CM * cm
        )
        company_width_pt = (
            usable_width_pt - dest_width_pt
        )  # expéditeur garde toute sa largeur

        vat_line = (
            f'<br/><font size="{FONT_BODY}" color="{COLOR_MUTED_PDF}">'
            f"{_xml_escape_for_paragraph(vat_status_text)}</font>"
            if vat_status_text
            else ""
        )
        company_info_left = (
            f'<font size="{FONT_HEADER_COMPANY}"><b>'
            f"{_xml_escape_for_paragraph(company_name)}</b></font><br/>"
            f'<font size="{FONT_BODY}">{_reportlab_safe_footer_html(company_address)}</font>'
            f"{vat_line}"
        )
        contact_bar = _format_company_contact_footer_bar(
            company_name, company_email, company_phone, company_uid
        )
        company_para = Paragraph(company_info_left, normal_style)

        left_cell_content: list[Any] = []  # Entreprise (expéditeur) — à gauche
        if logo_img:
            is_drawing = (
                hasattr(logo_img, "width")
                and hasattr(logo_img, "height")
                and hasattr(logo_img, "scale")
            )
            if is_drawing:
                logo_table = Table([[logo_img]], colWidths=[logo_width])
                logo_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                left_cell_content.append(logo_table)
            else:
                from reportlab.lib.styles import ParagraphStyle

                logo_style = ParagraphStyle(
                    "LogoStyle",
                    parent=styles["Normal"],
                    alignment=TA_LEFT,
                    leftIndent=0,
                    rightIndent=0,
                    spaceAfter=8,
                )
                logo_para = Paragraph(
                    (
                        f'<img src="{logo_path}" width="{logo_width}" '
                        f'height="{logo_height}"/>'
                    ),
                    logo_style,
                )
                left_cell_content.append(logo_para)
        left_cell_content.append(company_para)

        if recipient_para is not None:
            label_style = ParagraphStyle(
                "DestLabel",
                parent=normal_style,
                fontSize=FONT_SECONDARY,
                spaceAfter=2,
            )
            label_para = Paragraph("<b>Facturé à :</b>", label_style)
            recipient_block = Table(
                [[label_para], [recipient_para]],
                colWidths=[dest_width_pt],
            )
            recipient_block.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            # Convention comptable : entreprise (expéditeur) à gauche, destinataire (client) à droite
            # 2 colonnes : pas de spacer (évite de condenser l'expéditeur). LEFTPADDING déplace le destinataire à droite.
            header_table = Table(
                [[left_cell_content, recipient_block]],
                colWidths=[company_width_pt, dest_width_pt],
            )
            header_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (0, -1), 0),
                        ("RIGHTPADDING", (0, 0), (0, -1), 6),
                        (
                            "LEFTPADDING",
                            (1, 0),
                            (1, -1),
                            recipient_left_padding_mm * mm,
                        ),
                        ("RIGHTPADDING", (1, 0), (1, -1), 0),
                        ("TOPPADDING", (1, 0), (1, -1), recipient_top_padding_mm * mm),
                    ]
                )
            )
            story.append(header_table)
        else:
            if logo_img:
                is_drawing = (
                    hasattr(logo_img, "width")
                    and hasattr(logo_img, "height")
                    and hasattr(logo_img, "scale")
                )
                if is_drawing:
                    logo_table = Table([[logo_img]], colWidths=[logo_width])
                    logo_table.setStyle(
                        TableStyle(
                            [
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ]
                        )
                    )
                    story.append(logo_table)
                else:
                    from reportlab.lib.styles import ParagraphStyle

                    logo_style = ParagraphStyle(
                        "LogoStyle",
                        parent=styles["Normal"],
                        alignment=TA_LEFT,
                        leftIndent=0,
                        rightIndent=0,
                        spaceAfter=8,
                    )
                    logo_para = Paragraph(
                        (
                            f'<img src="{logo_path}" width="{logo_width}" '
                            f'height="{logo_height}"/>'
                        ),
                        logo_style,
                    )
                    story.append(logo_para)
            story.append(company_para)
        story.append(Spacer(1, 14))

        display_reminder_level = reminder_ctx.get("display_reminder_level")

        # === INFORMATIONS FACTURE (GAUCHE) ===
        echeance_label = (
            "Date d'échéance initiale :"
            if reminder_ctx.get("is_reminder")
            else "Date d'échéance :"
        )
        _mois_fr = (
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
        )
        period_label = (
            f"{_mois_fr[invoice.period_month - 1]} {invoice.period_year}"
            if 1 <= invoice.period_month <= MONTHS_PER_YEAR
            else f"{invoice.period_month:02d}.{invoice.period_year}"
        )
        _inv_n = _xml_escape_for_paragraph(str(invoice.invoice_number or ""))
        _per_lbl = _xml_escape_for_paragraph(period_label)
        invoice_info_left = (
            f'<font size="{FONT_META_NUMBER}"><b>Numéro de facture :</b></font> '
            f'<font size="{FONT_META_NUMBER}">{_inv_n}</font><br/>'
            f'<font size="{FONT_META_DATES}"><b>Date d\'émission :</b></font> '
            f'<font size="{FONT_META_DATES}">{invoice.issued_at.strftime("%d.%m.%Y")}</font><br/>'
            f'<font size="{FONT_META_DATES}"><b>'
            f"{_xml_escape_for_paragraph(echeance_label)}</b></font> "
            f'<font size="{FONT_META_DATES}">{invoice.due_date.strftime("%d.%m.%Y")}</font><br/>'
            f'<font size="{FONT_META_DATES}"><b>Période de facturation :</b></font> '
            f'<font size="{FONT_META_DATES}">{_per_lbl}</font>'
        )
        if reminder_ctx.get("is_reminder"):
            invoice_info_left += (
                f'<br/><font size="{FONT_META_DATES}"><b>Date du rappel :</b></font> '
                f'<font size="{FONT_META_DATES}">'
                f"{datetime.now(UTC).strftime('%d.%m.%Y')}</font>"
            )

        invoice_info_table = Table(
            [[Paragraph(invoice_info_left, normal_style)]],
            colWidths=[usable_width_pt],
        )
        invoice_info_table.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(invoice_info_table)
        story.append(Spacer(1, 14))

        # === TABLEAU DES COURSES ===
        # Fonction pour formater les adresses longues avec retour à la ligne
        # dans les colonnes
        def format_address_for_table(address, max_length=25):  # pyright: ignore[reportUnusedFunction]
            if not address or address == "Adresse inconnue":
                return "Adresse non renseignée"

            # Nettoyer l'adresse : supprimer "Suisse" et "Trajet"
            # mais garder les numéros d'adresse
            clean_address = address.replace(", Suisse", "").strip()
            # Supprimer le mot "Trajet" au début
            import re

            clean_address = re.sub(r"^Trajet\s+", "", clean_address)
            # Supprimer "Suisse" à la fin
            clean_address = clean_address.replace(" Suisse", "").strip()
            # Supprimer les points médians (·) mais garder les numéros d'adresse
            clean_address = clean_address.replace(" · ", " ").replace("·", "")

            # Si l'adresse est courte, la retourner telle quelle
            if len(clean_address) <= max_length:
                return clean_address

            # Diviser l'adresse en mots et créer des lignes
            words = clean_address.split(" ")
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if len(test_line) <= max_length:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return "\n".join(lines[:3])  # Maximum 3 lignes avec \n au lieu de <br/>

        # ✅ Unifier : utiliser toujours le tableau S2 (Date | Patient | Transport | Montant)
        # pour toutes les factures (client et clinique)
        strategy_value = None
        try:
            bs = getattr(invoice, "billing_strategy", None)
            if bs is None:
                strategy_value = None
            else:
                strategy_value = bs.value if hasattr(bs, "value") else str(bs)
        except Exception:
            strategy_value = None
        is_s2 = strategy_value == "s2_clinic_monthly"
        is_third_party = bool(
            getattr(invoice, "billing_party_id", None)
            or (
                invoice.bill_to_client_id
                and invoice.bill_to_client_id != invoice.client_id
            )
            or is_s2
        )

        # ✅ Utiliser toujours _build_s2_table (même pour factures client)
        _perf_s2_start = perf_counter()
        s2_table, consolidated_lines = _build_s2_table(
            invoice,
            font_name,
            font_name_bold,
            s2_main_style,
            bookings_by_id,
            include_non_ride=True,
            available_width_pt=usable_width_pt,
            max_simple_description_lines=2,
        )
        app_logger.info(
            "[PDF_PERF] _build_s2_table_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_s2_start) * 1000),
            getattr(invoice, "id", None),
        )

        # === MENTION RAPPEL (si mode rappel) ===
        if display_reminder_level:
            reminder_line = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "ReminderLine",
                                parent=styles["Normal"],
                                fontSize=11,
                                fontName=font_name_bold,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#374151"),
                            ),
                        )
                    ]
                ],
                colWidths=[usable_width_pt],
            )
            reminder_line.setStyle(
                TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (0, 0), 0.5, colors.HexColor("#D1D5DB")),
                    ]
                )
            )
            story.append(reminder_line)
            story.append(Spacer(1, 16))

        detail_title = Table(
            [[_detail_lines_heading_paragraph(styles, font_name_bold)]],
            colWidths=[usable_width_pt],
        )
        detail_title.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(detail_title)
        story.append(Spacer(1, 6))
        story.append(s2_table)
        # Légende A/R : même ordre que l'aperçu HTML (sous le tableau, avant remise globale / totaux).
        if _pdf_show_ar_legend(invoice, consolidated_lines, bookings_by_id):
            note_para = Paragraph(
                f'<font size="{FONT_SECONDARY}" color="{COLOR_MUTED_PDF}">'
                f"[A/R] = transport aller-retour</font>",
                normal_style,
            )
            story.append(Spacer(1, 8))
            story.append(note_para)
            story.append(Spacer(1, 6))
        _gd_hint_std = _global_discount_hint_flowable(
            invoice, styles, font_name, content_width_pt=usable_width_pt
        )
        if _gd_hint_std is not None:
            story.append(Spacer(1, 10))
            story.append(_gd_hint_std)
            story.append(Spacer(1, 9))
        else:
            story.append(Spacer(1, 2))

        # === TOTAL ===
        if reminder_ctx.get("is_reminder"):
            story.append(Spacer(1, 16))  # Respiration avant totaux (rappel)
        _preview_tot_w = (
            INVOICE_PREVIEW_TOTALS_LABEL_CM + INVOICE_PREVIEW_TOTALS_AMOUNT_CM
        ) * cm
        total_separator = Table([[""]], colWidths=[_preview_tot_w])
        total_separator.hAlign = "RIGHT"
        total_separator.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (0, 0), 0.75, colors.HexColor("#e2e8f0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(total_separator)
        story.append(Spacer(1, 12))

        # ✅ Utiliser toujours le format unifié pour les totaux (même libellé "TOTAL À FACTURER")
        # ✅ Si mode rappel, ajouter la ligne de frais de rappel
        total_table = _build_totals_table(
            invoice,
            is_s2,
            is_third_party,
            font_name,
            font_name_bold,
            template="standard",
            reminder_level=reminder_ctx.get("reminder_level"),
            reminder_fee=reminder_ctx.get("reminder_fee"),
            reminder_total_due=reminder_ctx.get("reminder_total_due"),
            reminder_principal=reminder_ctx.get("reminder_principal"),
        )
        story.append(total_table)
        story.append(Spacer(1, 30))

        # === PIED DE PAGE - NOTES DE FACTURATION ===

        # Utiliser billing_settings passé en paramètre
        # (déjà récupéré dans _create_invoice_pdf_content)

        # Délai de paiement (par défaut 10 jours)
        payment_terms_days = 10
        if billing_settings and billing_settings.payment_terms_days:
            payment_terms_days = int(billing_settings.payment_terms_days)

        # Frais de retard (par défaut 15 CHF)
        overdue_fee = Decimal("15.00")
        if billing_settings and billing_settings.overdue_fee:
            overdue_fee = billing_settings.overdue_fee

        # Message de facturation avec valeurs dynamiques
        jours_text = "jours" if payment_terms_days > 1 else "jour"

        # Informations bancaires (récupérer depuis billing_settings ou company)
        # En prod, si IBAN non lisible (déchiffrement invalide) : masquer, ne pas afficher "[IBAN non configuré]"
        iban_value = None
        if billing_settings and billing_settings.iban:
            iban_value = billing_settings.iban
        elif hasattr(company, "iban") and company.iban:
            iban_value = company.iban

        # Message du pied de page : en mode rappel, texte gradué selon le niveau.
        if display_reminder_level:
            reminder_level_val = reminder_ctx.get("reminder_level") or 1
            footer_message = _get_reminder_footer_message(reminder_level_val)
            if iban_value:
                footer_message = f"{footer_message} Paiement par virement bancaire : IBAN : {iban_value}"
        elif billing_settings and billing_settings.legal_footer:
            raw_footer = _resolve_legal_footer_placeholders(
                billing_settings.legal_footer,
                payment_terms_days,
                overdue_fee,
                jours_text,
            )
            footer_message = _sanitize_legal_footer_for_iban(raw_footer)
        else:
            footer_message = _build_default_legal_footer_html(
                payment_terms_days, overdue_fee, iban_value
            )
            if not iban_value:
                app_logger.warning(
                    "PDF (standard): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )

        # Pied de page légal : dessiné en zone fixe (marge inférieure), pas dans le flux
        mention = None
        footer_cb = _make_legal_footer_page_callback(
            footer_message,
            mention,
            centered_style,
            contact_bar=contact_bar,
        )

        def _on_first_page(canvas: Any, doc: Any) -> None:
            footer_cb(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # Doc avec page QR-Bill dédiée (marge bas 2 cm, pas de pied légal)
        doc = _make_invoice_doc_with_qrbill_page(
            buffer,
            top_margin_cm=2,
            bottom_margin_cm=2.5,  # Réserve espace pour pied de page légal
            left_margin_cm=INVOICE_PAGE_LEFT_MARGIN_CM,
            right_margin_cm=INVOICE_PAGE_RIGHT_MARGIN_CM,
            on_first_page=_on_first_page,
        )

        # === QR-BILL SUISSE OFFICIEL SUR PAGE SÉPARÉE ===
        # Forcer une nouvelle page pour le QR-Bill (marge bas 2 cm)
        from reportlab.platypus import (
            NextPageTemplate,
            PageBreak,
        )

        story.append(NextPageTemplate("QRBill"))
        story.append(PageBreak())

        # Espacement pour pousser le QR-Bill en bas de sa page (doit tenir dans usable_height)
        story.append(Spacer(1, QR_BILL_SPACER_PT))

        _perf_qr_start = perf_counter()
        try:
            qr_bill_service = self.qrbill_service
            qr_override = (
                reminder_ctx.get("reminder_total_due")
                if reminder_ctx.get("is_reminder")
                else None
            )
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(
                invoice, override_amount=qr_override
            )

            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
            else:
                story.append(Paragraph("QR-Bill non disponible", normal_style))

        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)
            story.append(Paragraph("QR-Bill non disponible", normal_style))
        app_logger.info(
            "[PDF_PERF] qr_bill_section_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_qr_start) * 1000),
            getattr(invoice, "id", None),
        )

        # Générer le PDF (callbacks dans PageTemplates)
        _perf_build_start = perf_counter()
        doc.build(story)
        app_logger.info(
            "[PDF_PERF] doc_build_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_build_start) * 1000),
            getattr(invoice, "id", None),
        )

        # Retourner le contenu et le nombre de lignes
        buffer.seek(0)
        # ✅ Calculer nb_rows depuis consolidated_lines (après regroupement aller/retour)
        nb_rows = len(consolidated_lines) if consolidated_lines else 0
        return (buffer.getvalue(), nb_rows)

    def _create_minimal_invoice_pdf(
        self,
        invoice,
        billing_settings,
        reminder_ctx: dict[str, Any],
        *,
        bookings_by_id: dict[int, Any],
    ):
        """Crée le contenu PDF d'une facture avec le template minimal.

        reminder_ctx: is_reminder, display_reminder_level, reminder_level, reminder_fee,
            reminder_principal, reminder_total_due (calculés une seule fois).
        """
        # ruff: noqa: I001
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.pagesizes import (
            A4,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        font_name, font_name_bold = _ensure_dejavu_pdf_fonts()

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=2.5 * cm,  # Réserve espace pour pied de page légal
            leftMargin=INVOICE_PAGE_LEFT_MARGIN_CM * cm,
            rightMargin=INVOICE_PAGE_RIGHT_MARGIN_CM * cm,
        )

        styles = getSampleStyleSheet()
        _min_lead = int(round(FONT_BODY * 1.3))
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_min_lead,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=4,
            fontName=font_name,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
        )
        centered_style = ParagraphStyle(
            "Centered",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_min_lead,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=4,
            fontName=font_name,
        )

        def _minimal_date_cell(date_str: str, inv_line: Any) -> Any:
            """Date brute ou Paragraph si note d'ajustement (ex. remise %)."""
            raw = getattr(inv_line, "adjustment_note", None)
            note = str(raw).strip() if raw is not None else ""
            if not note:
                return date_str
            esc_d = _xml_escape_for_paragraph(date_str)
            esc_n = _xml_escape_for_paragraph(note)
            return Paragraph(
                f'{esc_d}<br/><font size="7" color="#6b7280"><i>{esc_n}</i></font>',
                normal_style,
            )

        def _minimal_date_cell_with_discount_suffix(
            date_str: str, inv_line: Any, discount_suffix_html: str
        ) -> Any:
            """Date avec note d'ajustement et/ou ligne catalogue→net (HTML déjà formaté)."""
            raw = getattr(inv_line, "adjustment_note", None)
            note = str(raw).strip() if raw is not None else ""
            ds = discount_suffix_html or ""
            if not note and not ds:
                return date_str
            esc_d = _xml_escape_for_paragraph(date_str)
            parts = [esc_d]
            if note:
                esc_n = _xml_escape_for_paragraph(note)
                parts.append(
                    f'<br/><font size="7" color="#6b7280"><i>{esc_n}</i></font>'
                )
            if ds:
                parts.append(ds)
            return Paragraph("".join(parts), normal_style)

        story = []
        company = invoice.company

        # === EN-TÊTE SIMPLIFIÉ (SANS LOGO) : ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
        company_name = company.name or "[Nom entreprise non configuré]"
        company_address = self._get_company_address_for_pdf(company)
        company_phone = company.contact_phone or "[Téléphone non configuré]"
        company_email = (
            company.billing_email or company.contact_email or "[Email non configuré]"
        )
        company_uid = company.uid_ide or "[IDE/UID non configuré]"
        vat_status_text_m = ""
        if billing_settings and billing_settings.vat_applicable:
            vat_number = billing_settings.vat_number or ""
            if vat_number:
                vat_status_text_m = f"N° TVA : {vat_number}"
            else:
                vat_status_text_m = f"TVA {billing_settings.vat_rate or 7.7}% incluse"
        vat_line_m = (
            f'<br/><font size="{FONT_BODY}" color="{COLOR_MUTED_PDF}">'
            f"{_xml_escape_for_paragraph(vat_status_text_m)}</font>"
            if vat_status_text_m
            else ""
        )
        company_info = (
            f'<font size="{FONT_HEADER_COMPANY}"><b>'
            f"{_xml_escape_for_paragraph(company_name)}</b></font><br/>"
            f'<font size="{FONT_BODY}">{_reportlab_safe_footer_html(company_address)}</font>'
            f"{vat_line_m}"
        )
        contact_bar_min = _format_company_contact_footer_bar(
            company_name, company_email, company_phone, company_uid
        )
        company_para_min = Paragraph(company_info, normal_style)

        recipient_para_min, _ = _build_recipient_block_flowable(
            invoice,
            normal_style,
            bookings_by_id=bookings_by_id,
            name_font_size=FONT_CLIENT_NAME,
            addr_font_size=FONT_BODY,
        )
        recipient_top_padding_mm_min = 25.0  # destinataire légèrement plus bas
        recipient_left_padding_mm_min = (
            15.0  # déplace destinataire à droite (pas d'espace volé à l'expéditeur)
        )
        dest_width_pt_min = DEST_ADDR_MAX_WIDTH_MM * mm
        usable_width_pt_min = doc.pagesize[0] - doc.leftMargin - doc.rightMargin
        company_width_pt_min = usable_width_pt_min - dest_width_pt_min

        if recipient_para_min is not None:
            label_style_min = ParagraphStyle(
                "DestLabelMin",
                parent=normal_style,
                fontSize=FONT_SECONDARY,
                spaceAfter=2,
            )
            label_para_min = Paragraph("<b>Facturé à :</b>", label_style_min)
            recipient_block_min = Table(
                [[label_para_min], [recipient_para_min]],
                colWidths=[dest_width_pt_min],
            )
            recipient_block_min.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            # Convention comptable : entreprise à gauche, destinataire à droite. 2 colonnes, LEFTPADDING déplace destinataire.
            header_table_min = Table(
                [[company_para_min, recipient_block_min]],
                colWidths=[company_width_pt_min, dest_width_pt_min],
            )
            header_table_min.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (0, -1), 0),
                        ("RIGHTPADDING", (0, 0), (0, -1), 6),
                        (
                            "LEFTPADDING",
                            (1, 0),
                            (1, -1),
                            recipient_left_padding_mm_min * mm,
                        ),
                        ("RIGHTPADDING", (1, 0), (1, -1), 0),
                        (
                            "TOPPADDING",
                            (1, 0),
                            (1, -1),
                            recipient_top_padding_mm_min * mm,
                        ),
                    ]
                )
            )
            story.append(header_table_min)
        else:
            story.append(company_para_min)
        story.append(Spacer(1, 10))

        # === Contexte rappel ===
        display_reminder_level = reminder_ctx.get("display_reminder_level")
        is_reminder = reminder_ctx.get("is_reminder", False)

        # === INFORMATIONS FACTURE (SIMPLIFIÉES) ===
        echeance_label = "Échéance initiale:" if is_reminder else "Échéance:"
        _inv_m = _xml_escape_for_paragraph(str(invoice.invoice_number or ""))
        invoice_info = (
            f"<b>Facture {_inv_m}</b> - "
            f"{invoice.issued_at.strftime('%d.%m.%Y')} - "
            f"{echeance_label} {invoice.due_date.strftime('%d.%m.%Y')}"
        )
        if is_reminder:
            invoice_info += f" - Rappel: {datetime.now(UTC).strftime('%d.%m.%Y')}"
        invoice_info_table_m = Table(
            [[Paragraph(invoice_info, normal_style)]],
            colWidths=[usable_width_pt_min],
        )
        invoice_info_table_m.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(invoice_info_table_m)
        story.append(Spacer(1, 15))

        # === MENTION RAPPEL (si mode rappel) ===
        if display_reminder_level:
            reminder_line_m = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "ReminderLineM",
                                parent=styles["Normal"],
                                fontSize=11,
                                fontName=font_name_bold,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#374151"),
                            ),
                        )
                    ]
                ],
                colWidths=[usable_width_pt_min],
            )
            reminder_line_m.setStyle(
                TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (0, 0), 0.5, colors.HexColor("#D1D5DB")),
                    ]
                )
            )
            story.append(reminder_line_m)
            story.append(Spacer(1, 16))

        # === TABLEAU SIMPLIFIÉ (DATE + MONTANT SEULEMENT) ===
        strategy_value = None
        try:
            bs = getattr(invoice, "billing_strategy", None)
            if bs is None:
                strategy_value = None
            else:
                strategy_value = bs.value if hasattr(bs, "value") else str(bs)
        except Exception:
            strategy_value = None
        is_s2 = strategy_value == "s2_clinic_monthly"
        is_third_party = bool(
            getattr(invoice, "billing_party_id", None)
            or (
                invoice.bill_to_client_id
                and invoice.bill_to_client_id != invoice.client_id
            )
            or is_s2
        )
        detail_title_min = Table(
            [[_detail_lines_heading_paragraph(styles, font_name_bold)]],
            colWidths=[usable_width_pt_min],
        )
        detail_title_min.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(detail_title_min)
        story.append(Spacer(1, 6))
        suppress_line_discount_breakdown_min = False
        table_data: list[list[Any]] = (
            [["Date", "Patient", "Montant"]]
            if is_third_party
            else [["Date", "Montant"]]
        )

        for line in invoice.lines:
            # Aligné _build_s2_table : déduction manuelle HT (CUSTOM < 0) visible au détail
            if line.type == InvoiceLineType.CUSTOM:
                if not _custom_line_include_in_s2_detail_table(line):
                    continue
                cat_mc, net_mc = _line_catalog_vs_net_ht(line)
                cat_mshow = (
                    cat_mc if abs(cat_mc - net_mc) > _PDF_CATALOG_NET_EPS else None
                )
                if suppress_line_discount_breakdown_min:
                    cat_mshow = None
                disc_mc = ""
                if (
                    not suppress_line_discount_breakdown_min
                    and cat_mshow is not None
                    and abs(cat_mc - net_mc) > _PDF_CATALOG_NET_EPS
                ):
                    disc_mc = _pdf_s2_per_line_discount_suffix_html(cat_mc, net_mc)
                amt_flow_mc = _pdf_minimal_amount_only_flowable(net_mc, normal_style)
                detail_para = _minimal_custom_detail_paragraph(
                    line, normal_style, extra_html_suffix=disc_mc
                )
                svc_date_mc = _line_meta_service_date_display_fr(line)
                if is_third_party:
                    table_data.append(
                        [
                            _minimal_date_cell(svc_date_mc, line),
                            detail_para,
                            amt_flow_mc,
                        ]
                    )
                else:
                    table_data.append([detail_para, amt_flow_mc])
                continue
            if (
                line.type
                in (
                    InvoiceLineType.RIDE,
                    InvoiceLineType.MATERIAL_DELIVERY,
                )
                and line.reservation_id
            ):
                booking = bookings_by_id.get(line.reservation_id)
                if booking:
                    date_str = (
                        booking.scheduled_time.strftime("%d/%m/%Y")
                        if booking.scheduled_time
                        else ""
                    )
                    cat_mb, net_mb = _line_catalog_vs_net_ht(line)
                    cat_mbs = (
                        cat_mb if abs(cat_mb - net_mb) > _PDF_CATALOG_NET_EPS else None
                    )
                    if suppress_line_discount_breakdown_min:
                        cat_mbs = None
                    disc_mb = ""
                    if (
                        not suppress_line_discount_breakdown_min
                        and cat_mbs is not None
                        and abs(cat_mb - net_mb) > _PDF_CATALOG_NET_EPS
                    ):
                        disc_mb = _pdf_s2_per_line_discount_suffix_html(cat_mb, net_mb)
                    amt_flow_mb = _pdf_minimal_amount_only_flowable(
                        net_mb, normal_style
                    )
                    if is_third_party:
                        # ✅ S2: Utiliser le snapshot patient_name depuis line_meta (traçabilité juridique)
                        patient_name = "Patient"
                        lm_snap = (
                            line.line_meta if isinstance(line.line_meta, dict) else {}
                        )
                        patient_name = (
                            lm_snap.get("patient_name")
                            or booking.customer_name
                            or (
                                f"{booking.client.user.first_name or ''} "
                                f"{booking.client.user.last_name or ''}"
                            ).strip()
                            or "Patient"
                        )
                        if len(patient_name) > MAX_PATIENT_NAME_LENGTH:
                            patient_name = (
                                patient_name[: MAX_PATIENT_NAME_LENGTH - 1] + "."
                            )
                        pn_esc = _xml_escape_for_paragraph(patient_name)
                        patient_cell_mb = Paragraph(pn_esc, normal_style)
                        date_mb = _minimal_date_cell_with_discount_suffix(
                            date_str, line, disc_mb
                        )
                        table_data.append([date_mb, patient_cell_mb, amt_flow_mb])
                    else:
                        date_cell_el = _minimal_date_cell(date_str, line)
                        if disc_mb and isinstance(date_cell_el, str):
                            date_cell_el = Paragraph(
                                f"{_xml_escape_for_paragraph(date_cell_el)}{disc_mb}",
                                normal_style,
                            )
                        table_data.append([date_cell_el, amt_flow_mb])
                else:
                    cat_mz, net_mz = _line_catalog_vs_net_ht(line)
                    cat_mzs = (
                        cat_mz if abs(cat_mz - net_mz) > _PDF_CATALOG_NET_EPS else None
                    )
                    if suppress_line_discount_breakdown_min:
                        cat_mzs = None
                    disc_mz = ""
                    if (
                        not suppress_line_discount_breakdown_min
                        and cat_mzs is not None
                        and abs(cat_mz - net_mz) > _PDF_CATALOG_NET_EPS
                    ):
                        disc_mz = _pdf_s2_per_line_discount_suffix_html(cat_mz, net_mz)
                    amt_flow_mz = _pdf_minimal_amount_only_flowable(
                        net_mz, normal_style
                    )
                    if is_third_party:
                        date_mz = _minimal_date_cell_with_discount_suffix(
                            "", line, disc_mz
                        )
                        table_data.append([date_mz, "N/A", amt_flow_mz])
                    else:
                        dz_cell = _minimal_date_cell("", line)
                        if disc_mz and isinstance(dz_cell, str):
                            dz_cell = Paragraph(
                                f"{_xml_escape_for_paragraph(dz_cell)}{disc_mz}",
                                normal_style,
                            )
                        table_data.append([dz_cell, amt_flow_mz])
            else:
                cat_mw, net_mw = _line_catalog_vs_net_ht(line)
                cat_mws = (
                    cat_mw if abs(cat_mw - net_mw) > _PDF_CATALOG_NET_EPS else None
                )
                if suppress_line_discount_breakdown_min:
                    cat_mws = None
                disc_mw = ""
                if (
                    not suppress_line_discount_breakdown_min
                    and cat_mws is not None
                    and abs(cat_mw - net_mw) > _PDF_CATALOG_NET_EPS
                ):
                    disc_mw = _pdf_s2_per_line_discount_suffix_html(cat_mw, net_mw)
                amt_flow_mw = _pdf_minimal_amount_only_flowable(net_mw, normal_style)
                if is_third_party:
                    date_mw = _minimal_date_cell_with_discount_suffix("", line, disc_mw)
                    table_data.append([date_mw, "N/A", amt_flow_mw])
                else:
                    dw_cell = _minimal_date_cell("", line)
                    if disc_mw and isinstance(dw_cell, str):
                        dw_cell = Paragraph(
                            f"{_xml_escape_for_paragraph(dw_cell)}{disc_mw}",
                            normal_style,
                        )
                    table_data.append([dw_cell, amt_flow_mw])

        if is_third_party:
            services_table = Table(table_data, colWidths=[3 * cm, 4 * cm, 2.5 * cm])
        else:
            services_table = Table(table_data, colWidths=[4 * cm, 2.5 * cm])
        _hdr_bg_m = colors.HexColor("#f8fafc")
        _hdr_text_m = colors.HexColor("#475569")
        _row_sep_m = colors.HexColor("#f1f5f9")
        _hdr_rule_m = colors.HexColor("#e2e8f0")
        style_min_tbl: list[Any] = [
            ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), _hdr_text_m),
            ("BACKGROUND", (0, 0), (-1, 0), _hdr_bg_m),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, _hdr_rule_m),
            ("FONTNAME", (0, 1), (-1, -1), font_name),
            ("TEXTCOLOR", (0, 1), (0, -1), colors.HexColor("#334155")),
            ("TEXTCOLOR", (1, 1), (-1, -1), colors.HexColor("#0f172a")),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ("TOPPADDING", (0, 1), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (0, -1), 6),
            ("RIGHTPADDING", (0, 0), (0, -1), 2),
            ("LEFTPADDING", (1, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        _n_rows_min_tbl = len(table_data)
        if _n_rows_min_tbl > LEVEL_THRESHOLD:
            for _r_m in range(1, _n_rows_min_tbl - 1):
                style_min_tbl.append(
                    ("LINEBELOW", (0, _r_m), (-1, _r_m), 0.35, _row_sep_m)
                )
        services_table.setStyle(TableStyle(style_min_tbl))
        story.append(services_table)
        story.append(Spacer(1, 10))

        # Aligné InvoiceLivePreview + templates standard/détaillé : encadré explicatif remise globale.
        _gd_hint_min = _global_discount_hint_flowable(
            invoice, styles, font_name, content_width_pt=usable_width_pt_min
        )
        if _gd_hint_min is not None:
            story.append(Spacer(1, 10))
            story.append(_gd_hint_min)
            story.append(Spacer(1, 9))
        else:
            story.append(Spacer(1, 2))

        # === TOTAL SIMPLIFIÉ ===
        # ✅ Mode rappel : mini-table (Sous-total facture + Frais + TOTAL)
        gd_min: dict[str, Any] | None = None
        gd_min_style_extra: list[tuple[Any, ...]] = []
        if (
            is_reminder
            and reminder_ctx.get("reminder_fee") is not None
            and reminder_ctx.get("reminder_total_due") is not None
            and reminder_ctx.get("reminder_principal") is not None
        ):
            principal_float = float(reminder_ctx["reminder_principal"])
            reminder_fee_float = float(reminder_ctx["reminder_fee"])
            final_total = float(reminder_ctx["reminder_total_due"])
            reminder_fee_label = "Frais de rappel :"
            _tot_lbl_rem = "TOTAL À FACTURER :" if is_s2 else "TOTAL :"
            total_data = [
                ["Montant facture initiale :", f"CHF {principal_float:.2f}"],
                [reminder_fee_label, f"CHF {reminder_fee_float:.2f}"],
                [_tot_lbl_rem, f"CHF {final_total:.2f}"],
            ]
        else:
            total_amount = float(invoice.total_amount)
            if isinstance(invoice.meta, dict) and invoice.meta.get("global_discount"):
                gd_min = cast(dict[str, Any], invoice.meta["global_discount"])
            if gd_min is not None:
                # Priorité au catalogue « avant remise » en méta (apply_draft_global_discount).
                # La somme des lignes HT > 0 inclut déjà les transports remisés : elle ne doit pas servir de sous-total « avant % ».
                _detail_min = _sum_positive_billed_lines_excluding_global_discount(
                    invoice
                )
                gross_ht = float(gd_min.get("subtotal_before_ht") or 0)
                if gross_ht <= 0 and _detail_min is not None:
                    gross_ht = float(_detail_min)
                disc_ht = float(gd_min.get("amount_ht", 0))
                pct_gd = float(gd_min.get("percent", 0))
                note_gd = str(gd_min.get("note") or "").strip()
                disc_label = _format_global_discount_pdf_label(pct_gd)
                net_ht = float(invoice.subtotal_amount)
                vat_min = float(invoice.vat_total_amount)
                vat_appl_min = False
                if isinstance(invoice.meta, dict) and "vat" in invoice.meta:
                    vat_appl_min = bool(invoice.meta["vat"].get("applicable"))
                total_data = [
                    [
                        "Sous-total HT (avant réduction globale)",
                        _format_chf_pdf_mono(gross_ht),
                    ],
                    [disc_label, _format_chf_discount_pdf(disc_ht)],
                ]
                if note_gd:
                    total_data.append(
                        [_pdf_note_global_discount_for_totals_table(note_gd), ""]
                    )
                    gd_min_style_extra.extend(
                        [
                            ("SPAN", (0, 2), (1, 2)),
                            ("ALIGN", (0, 2), (1, 2), "LEFT"),
                            ("FONTSIZE", (0, 2), (1, 2), 8),
                            ("TEXTCOLOR", (0, 2), (1, 2), colors.HexColor("#4b5563")),
                            ("FONTNAME", (0, 2), (1, 2), font_name),
                        ]
                    )
                if vat_appl_min and vat_min > 0:
                    total_data.extend(
                        [
                            [
                                "Total HT après réduction globale",
                                _format_chf_pdf_mono(net_ht),
                            ],
                            ["TVA :", _format_chf_pdf_mono(vat_min)],
                            [
                                ("TOTAL À FACTURER :" if is_s2 else "TOTAL :"),
                                _format_chf_pdf_mono(total_amount),
                            ],
                        ]
                    )
                elif abs(net_ht - total_amount) < _PDF_CHF_AMOUNT_EQ_EPS:
                    # Sans TVA : net_ht et total_amount identiques -> une seule ligne finale.
                    total_data.append(
                        [
                            ("TOTAL À FACTURER :" if is_s2 else "TOTAL :"),
                            _format_chf_pdf_mono(total_amount),
                        ]
                    )
                else:
                    total_data.extend(
                        [
                            [
                                "Total HT après réduction globale",
                                _format_chf_pdf_mono(net_ht),
                            ],
                            [
                                ("TOTAL À FACTURER :" if is_s2 else "TOTAL :"),
                                _format_chf_pdf_mono(total_amount),
                            ],
                        ]
                    )
            else:
                total_data = [
                    [
                        ("TOTAL À FACTURER :" if is_s2 else "TOTAL :"),
                        (
                            f"CHF {total_amount:.2f}"
                            if is_s2
                            else f"{total_amount:.2f} CHF"
                        ),
                    ]
                ]
        # Tableau totaux : largeurs généreuses si plusieurs lignes (remise globale / rappel) pour éviter tout chevauchement.
        _reminder_ok = (
            is_reminder
            and reminder_ctx.get("reminder_fee") is not None
            and reminder_ctx.get("reminder_total_due") is not None
            and reminder_ctx.get("reminder_principal") is not None
        )
        if _reminder_ok or gd_min is not None:
            _tot_min_w = [10.5 * cm, 4.5 * cm]
        else:
            # Évite le chevauchement libellé/montant sur "TOTAL À FACTURER" avec montants à 6+ chiffres.
            _tot_min_w = [6.4 * cm, 4.2 * cm]
        total_table = Table(total_data, colWidths=_tot_min_w)
        style_rules = [
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]
        style_rules.extend(gd_min_style_extra)
        if is_reminder or gd_min is not None:
            style_rules.extend(
                [
                    ("FONTNAME", (0, 0), (-1, -2), font_name),
                    ("FONTNAME", (0, -1), (-1, -1), font_name_bold),
                ]
            )
        else:
            style_rules.append(("FONTNAME", (0, 0), (-1, -1), font_name_bold))
        # Remise globale (facture minimale) : montants en Courier + padding resserré + décimales alignées.
        if gd_min is not None:
            style_rules.extend(
                [
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ("FONTNAME", (1, 0), (1, -2), "Courier"),
                    ("FONTNAME", (1, -1), (1, -1), "Courier-Bold"),
                ]
            )
        style_rules.append(
            ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.HexColor("#e2e8f0"))
        )
        style_rules.append(("TOPPADDING", (0, -1), (-1, -1), 8))
        total_table.setStyle(TableStyle(style_rules))
        story.append(total_table)
        story.append(Spacer(1, 20))

        # === PIED DE PAGE SIMPLIFIÉ ===
        payment_terms_days = 10
        if billing_settings and billing_settings.payment_terms_days:
            payment_terms_days = int(billing_settings.payment_terms_days)
        overdue_fee = Decimal("15.00")
        if billing_settings and billing_settings.overdue_fee:
            overdue_fee = billing_settings.overdue_fee
        jours_text = "jours" if payment_terms_days > 1 else "jour"

        if is_reminder:
            reminder_level_val = reminder_ctx.get("reminder_level") or 1
            footer_message = _get_reminder_footer_message(reminder_level_val)
            iban_value_min = (
                billing_settings.iban
                if billing_settings and billing_settings.iban
                else None
            )
            if iban_value_min:
                footer_message += (
                    f" Paiement par virement bancaire : IBAN : {iban_value_min}"
                )
        elif billing_settings and billing_settings.legal_footer:
            raw_footer = _resolve_legal_footer_placeholders(
                billing_settings.legal_footer,
                payment_terms_days,
                overdue_fee,
                jours_text,
            )
            footer_message = _sanitize_legal_footer_for_iban(raw_footer)
        else:
            iban_value_min = (
                billing_settings.iban
                if billing_settings and billing_settings.iban
                else None
            )
            footer_message = _build_default_legal_footer_html(
                payment_terms_days, overdue_fee, iban_value_min
            )
            if not iban_value_min:
                app_logger.warning(
                    "PDF (minimal): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )

        # Pied de page légal : dessiné en zone fixe (marge inférieure)
        mention = None
        footer_cb_min = _make_legal_footer_page_callback(
            footer_message,
            mention,
            centered_style,
            contact_bar=contact_bar_min,
        )

        def _on_first_page_min(canvas: Any, doc: Any) -> None:
            footer_cb_min(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # === QR-BILL (SIMPLIFIÉ) ===
        story.append(PageBreak())
        story.append(Spacer(1, QR_BILL_SPACER_PT))

        try:
            qr_bill_service = self.qrbill_service
            qr_override_m = (
                reminder_ctx.get("reminder_total_due") if is_reminder else None
            )
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(
                invoice, override_amount=qr_override_m
            )
            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)

        doc.build(story, onFirstPage=_on_first_page_min)
        buffer.seek(0)
        nb_rows_min = len(invoice.lines) if getattr(invoice, "lines", None) else 0
        return (buffer.getvalue(), nb_rows_min)

    def _create_detailed_invoice_pdf(
        self,
        invoice,
        billing_settings,
        reminder_ctx: dict[str, Any],
        *,
        bookings_by_id: dict[int, Any],
    ):
        """Crée le contenu PDF d'une facture avec le template détaillé.

        reminder_ctx: is_reminder, display_reminder_level, reminder_level, reminder_fee,
            reminder_principal, reminder_total_due (calculés une seule fois).
        """
        # ruff: noqa: I001
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.pagesizes import (
            A4,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        font_name, font_name_bold = _ensure_dejavu_pdf_fonts()

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2.5 * cm,  # Réserve espace pour pied de page légal
            leftMargin=INVOICE_PAGE_LEFT_MARGIN_CM * cm,
            rightMargin=INVOICE_PAGE_RIGHT_MARGIN_CM * cm,
        )

        styles = getSampleStyleSheet()
        _det_lead = int(round(FONT_BODY * 1.3))
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_det_lead,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=6,
            fontName=font_name,
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0,
        )
        centered_style = ParagraphStyle(
            "Centered",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_det_lead,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName=font_name,
        )
        detail_style = ParagraphStyle(
            "Detail",
            parent=styles["Normal"],
            fontSize=FONT_COMPANY_CONTACT,
            textColor=colors.darkgrey,
            alignment=TA_LEFT,
            spaceAfter=4,
            fontName=font_name,
        )
        s2_main_style = ParagraphStyle(
            "S2Main",
            parent=styles["Normal"],
            fontSize=FONT_BODY,
            leading=_det_lead,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
            fontName=font_name,
        )

        story = []
        company = invoice.company

        # === EN-TÊTE AVEC LOGO (comme standard) ===
        logo_img = None
        logo_path = None
        logo_width = 0.0
        logo_height = 0.0
        if hasattr(company, "logo_url") and company.logo_url:
            try:
                logo_url = company.logo_url.strip()
                if not logo_url.startswith(("http://", "https://")):
                    logo_url_clean = logo_url.lstrip("/")
                    if logo_url_clean.startswith("uploads/"):
                        logo_url_clean = logo_url_clean[8:]
                    # ✅ Chemin correct: /app/uploads
                    from flask import current_app

                    uploads_dir = Path(
                        current_app.config.get("UPLOAD_FOLDER", "/app/uploads")
                    )
                    logo_path = uploads_dir / logo_url_clean
                    if logo_path and Path(logo_path).exists():
                        max_width_pt = 595 * 0.24
                        logo_img, logo_width, logo_height = _load_logo_ratio_safe(
                            logo_path, max_width_pt
                        )
            except Exception:
                pass

        # === EN-TÊTE DETAILED : ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
        recipient_para, _ = _build_recipient_block_flowable(
            invoice,
            normal_style,
            bookings_by_id=bookings_by_id,
            name_font_size=FONT_CLIENT_NAME,
            addr_font_size=FONT_BODY,
        )
        recipient_top_padding_mm = 25.0  # destinataire légèrement plus bas
        recipient_left_padding_mm = (
            15.0  # déplace destinataire à droite (pas d'espace volé à l'expéditeur)
        )
        dest_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
        usable_width_pt = doc.pagesize[0] - doc.leftMargin - doc.rightMargin
        company_width_pt = usable_width_pt - dest_width_pt

        company_name = company.name or "[Nom entreprise non configuré]"
        company_address = self._get_company_address_for_pdf(company)
        company_phone = company.contact_phone or "[Téléphone non configuré]"
        company_email = (
            company.billing_email or company.contact_email or "[Email non configuré]"
        )
        company_uid = company.uid_ide or "[IDE/UID non configuré]"
        # ✅ Statut TVA : afficher uniquement si assujetti (pas de mention si non assujetti)
        vat_status_text = ""
        if billing_settings and billing_settings.vat_applicable:
            vat_number = billing_settings.vat_number or ""
            if vat_number:
                vat_status_text = f"N° TVA : {vat_number}"
            else:
                vat_status_text = f"TVA {billing_settings.vat_rate or 7.7}% incluse"

        vat_line_d = (
            f'<br/><font size="{FONT_BODY}" color="{COLOR_MUTED_PDF}">'
            f"{_xml_escape_for_paragraph(vat_status_text)}</font>"
            if vat_status_text
            else ""
        )
        company_info_detailed = (
            f'<font size="{FONT_HEADER_COMPANY}"><b>'
            f"{_xml_escape_for_paragraph(company_name)}</b></font><br/>"
            f'<font size="{FONT_BODY}">{_reportlab_safe_footer_html(company_address)}</font>'
            f"{vat_line_d}"
        )
        contact_bar_det = _format_company_contact_footer_bar(
            company_name, company_email, company_phone, company_uid
        )
        company_para = Paragraph(company_info_detailed, normal_style)

        left_cell_content_d: list[Any] = []  # Entreprise (expéditeur) — à gauche
        if logo_img:
            is_drawing = (
                hasattr(logo_img, "width")
                and hasattr(logo_img, "height")
                and hasattr(logo_img, "scale")
            )
            if is_drawing:
                logo_table = Table([[logo_img]], colWidths=[logo_width])
                logo_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                left_cell_content_d.append(logo_table)
            else:
                logo_style = ParagraphStyle(
                    "LogoStyle",
                    parent=styles["Normal"],
                    alignment=TA_LEFT,
                    leftIndent=0,
                    rightIndent=0,
                    spaceAfter=8,
                )
                logo_para = Paragraph(
                    (
                        f'<img src="{logo_path}" width="{logo_width}" '
                        f'height="{logo_height}"/>'
                    ),
                    logo_style,
                )
                left_cell_content_d.append(logo_para)
        left_cell_content_d.append(company_para)

        if recipient_para is not None:
            label_style_d = ParagraphStyle(
                "DestLabelD",
                parent=normal_style,
                fontSize=FONT_SECONDARY,
                spaceAfter=2,
            )
            label_para_d = Paragraph("<b>Facturé à :</b>", label_style_d)
            recipient_block_d = Table(
                [[label_para_d], [recipient_para]],
                colWidths=[dest_width_pt],
            )
            recipient_block_d.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            # Convention comptable : entreprise à gauche, destinataire à droite. 2 colonnes, LEFTPADDING déplace destinataire.
            header_table_d = Table(
                [[left_cell_content_d, recipient_block_d]],
                colWidths=[company_width_pt, dest_width_pt],
            )
            header_table_d.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (0, -1), 0),
                        ("RIGHTPADDING", (0, 0), (0, -1), 6),
                        (
                            "LEFTPADDING",
                            (1, 0),
                            (1, -1),
                            recipient_left_padding_mm * mm,
                        ),
                        ("RIGHTPADDING", (1, 0), (1, -1), 0),
                        ("TOPPADDING", (1, 0), (1, -1), recipient_top_padding_mm * mm),
                    ]
                )
            )
            story.append(header_table_d)
        else:
            if logo_img:
                is_drawing = (
                    hasattr(logo_img, "width")
                    and hasattr(logo_img, "height")
                    and hasattr(logo_img, "scale")
                )
                if is_drawing:
                    logo_table = Table([[logo_img]], colWidths=[logo_width])
                    logo_table.setStyle(
                        TableStyle(
                            [
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ]
                        )
                    )
                    story.append(logo_table)
                else:
                    logo_style = ParagraphStyle(
                        "LogoStyle",
                        parent=styles["Normal"],
                        alignment=TA_LEFT,
                        leftIndent=0,
                        rightIndent=0,
                        spaceAfter=8,
                    )
                    logo_para = Paragraph(
                        (
                            f'<img src="{logo_path}" width="{logo_width}" '
                            f'height="{logo_height}"/>'
                        ),
                        logo_style,
                    )
                    story.append(logo_para)
            story.append(company_para)
        story.append(Spacer(1, 20))

        # ✅ Déterminer si c'est une facture S2 (pour le formatage)
        strategy_value = None
        try:
            bs = getattr(invoice, "billing_strategy", None)
            if bs is None:
                strategy_value = None
            else:
                strategy_value = bs.value if hasattr(bs, "value") else str(bs)
        except Exception:
            strategy_value = None
        is_s2 = strategy_value == "s2_clinic_monthly"

        display_reminder_level = reminder_ctx.get("display_reminder_level")

        # === INFORMATIONS FACTURE DÉTAILLÉES ===
        status_value = (
            invoice.status.value
            if hasattr(invoice.status, "value")
            else str(invoice.status)
        )
        _mois_fr_d = (
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
        )
        period_label_d = (
            f"{_mois_fr_d[invoice.period_month - 1]} {invoice.period_year}"
            if 1 <= invoice.period_month <= MONTHS_PER_YEAR
            else f"{invoice.period_month:02d}.{invoice.period_year}"
        )
        echeance_label = (
            "Date d'échéance initiale :"
            if reminder_ctx.get("is_reminder")
            else "Date d'échéance :"
        )

        _inv_d = _xml_escape_for_paragraph(str(invoice.invoice_number or ""))
        _per_d = _xml_escape_for_paragraph(period_label_d)
        _st_d = _xml_escape_for_paragraph(str(status_value))
        _ech_d = _xml_escape_for_paragraph(echeance_label)
        invoice_info_detailed = (
            f"<b>Numéro de facture :</b> {_inv_d}<br/>"
            f"<b>Date d'émission :</b> {invoice.issued_at.strftime('%d.%m.%Y')}<br/>"
            f"<b>{_ech_d}</b> {invoice.due_date.strftime('%d.%m.%Y')}<br/>"
            f"<b>Période de facturation :</b> {_per_d}<br/>"
            f"<b>Statut :</b> {_st_d}"
        )
        if reminder_ctx.get("is_reminder"):
            invoice_info_detailed += (
                f"<br/><b>Date du rappel :</b> {datetime.now(UTC).strftime('%d.%m.%Y')}"
            )
        invoice_info_table_d = Table(
            [[Paragraph(invoice_info_detailed, normal_style)]],
            colWidths=[usable_width_pt],
        )
        invoice_info_table_d.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(invoice_info_table_d)
        story.append(Spacer(1, 20))

        # === TABLEAU DÉTAILLÉ AVEC NOTES ===
        strategy_value = None
        try:
            bs = getattr(invoice, "billing_strategy", None)
            if bs is None:
                strategy_value = None
            else:
                strategy_value = bs.value if hasattr(bs, "value") else str(bs)
        except Exception:
            strategy_value = None
        is_s2 = strategy_value == "s2_clinic_monthly"
        is_third_party = bool(
            getattr(invoice, "billing_party_id", None)
            or (
                invoice.bill_to_client_id
                and invoice.bill_to_client_id != invoice.client_id
            )
            or is_s2
        )

        def format_address_for_table(address, max_length=25):  # pyright: ignore[reportUnusedFunction]
            """Formate une adresse pour le tableau (version compacte)."""
            if not address or address == "Adresse inconnue":
                return "Adresse non renseignée"
            clean_address = address.replace(", Suisse", "").strip()
            import re

            clean_address = re.sub(r"^Trajet\s+", "", clean_address)
            clean_address = clean_address.replace(" Suisse", "").strip()
            clean_address = clean_address.replace(" · ", " ").replace("·", "")
            # Pour S2, on veut des adresses plus courtes et lisibles
            if len(clean_address) <= max_length:
                return clean_address
            # Tronquer intelligemment (garder le début)
            return clean_address[: max_length - 3] + "..."

        # ✅ Unifier : utiliser toujours le tableau S2 (Date | Patient | Transport | Montant)
        # pour toutes les factures (client et clinique)
        _perf_s2d_start = perf_counter()
        s2_table, consolidated_lines = _build_s2_table(
            invoice,
            font_name,
            font_name_bold,
            s2_main_style,
            bookings_by_id,
            include_non_ride=True,
            available_width_pt=doc.width,
            max_simple_description_lines=None,
        )
        app_logger.info(
            "[PDF_PERF] _build_s2_table_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_s2d_start) * 1000),
            getattr(invoice, "id", None),
        )

        # === MENTION RAPPEL (si mode rappel) ===
        if display_reminder_level:
            reminder_line_d = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "ReminderLineD",
                                parent=styles["Normal"],
                                fontSize=11,
                                fontName=font_name_bold,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#374151"),
                            ),
                        )
                    ]
                ],
                colWidths=[usable_width_pt],
            )
            reminder_line_d.setStyle(
                TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (0, 0), 0.5, colors.HexColor("#D1D5DB")),
                    ]
                )
            )
            story.append(reminder_line_d)
            story.append(Spacer(1, 16))

        detail_title_d = Table(
            [[_detail_lines_heading_paragraph(styles, font_name_bold)]],
            colWidths=[usable_width_pt],
        )
        detail_title_d.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(detail_title_d)
        story.append(Spacer(1, 6))
        story.append(s2_table)
        # Légende A/R : même ordre que l'aperçu HTML (sous le tableau, avant remise globale / totaux).
        if _pdf_show_ar_legend(invoice, consolidated_lines, bookings_by_id):
            note_para = Paragraph(
                f'<font size="{FONT_SECONDARY}" color="{COLOR_MUTED_PDF}">'
                f"[A/R] = transport aller-retour</font>",
                normal_style,
            )
            story.append(Spacer(1, 8))
            story.append(note_para)
            story.append(Spacer(1, 6))
        _gd_hint_d = _global_discount_hint_flowable(
            invoice, styles, font_name, content_width_pt=doc.width
        )
        if _gd_hint_d is not None:
            story.append(Spacer(1, 10))
            story.append(_gd_hint_d)
            story.append(Spacer(1, 9))
        else:
            story.append(Spacer(1, 2))

        # === TOTAL DÉTAILLÉ ===
        _preview_tot_w_d = (
            INVOICE_PREVIEW_TOTALS_LABEL_CM + INVOICE_PREVIEW_TOTALS_AMOUNT_CM
        ) * cm
        total_separator = Table([[""]], colWidths=[_preview_tot_w_d])
        total_separator.hAlign = "RIGHT"
        total_separator.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (0, 0), 0.75, colors.HexColor("#e2e8f0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(total_separator)
        story.append(Spacer(1, 12))

        # ✅ Utiliser toujours le format unifié pour les totaux (même libellé "TOTAL À FACTURER")
        # ✅ Si mode rappel, ajouter la ligne de frais de rappel
        if reminder_ctx.get("is_reminder"):
            story.append(Spacer(1, 16))  # Respiration avant totaux (rappel)
        total_table = _build_totals_table(
            invoice,
            is_s2,
            is_third_party,
            font_name,
            font_name_bold,
            template="detailed",
            reminder_level=reminder_ctx.get("reminder_level"),
            reminder_fee=reminder_ctx.get("reminder_fee"),
            reminder_total_due=reminder_ctx.get("reminder_total_due"),
            reminder_principal=reminder_ctx.get("reminder_principal"),
        )
        story.append(total_table)
        story.append(Spacer(1, 30))

        # === NOTES ET INFORMATIONS SUPPLÉMENTAIRES ===
        if invoice.notes:
            story.append(Paragraph("<b>Notes :</b>", normal_style))
            story.append(
                Paragraph(
                    _reportlab_multiline_plain_to_html(invoice.notes), detail_style
                )
            )
            story.append(Spacer(1, 15))

        # === PIED DE PAGE DÉTAILLÉ ===
        payment_terms_days = 10
        if billing_settings and billing_settings.payment_terms_days:
            payment_terms_days = int(billing_settings.payment_terms_days)

        overdue_fee = Decimal("15.00")
        if billing_settings and billing_settings.overdue_fee:
            overdue_fee = billing_settings.overdue_fee

        jours_text = "jours" if payment_terms_days > 1 else "jour"
        iban_value = None
        if billing_settings and billing_settings.iban:
            iban_value = billing_settings.iban
        elif hasattr(company, "iban") and company.iban:
            iban_value = company.iban

        if display_reminder_level:
            reminder_level_val = reminder_ctx.get("reminder_level") or 1
            footer_message = _get_reminder_footer_message(reminder_level_val)
            if iban_value:
                footer_message += (
                    f" Paiement par virement bancaire : IBAN : {iban_value}"
                )
        elif billing_settings and billing_settings.legal_footer:
            raw_footer = _resolve_legal_footer_placeholders(
                billing_settings.legal_footer,
                payment_terms_days,
                overdue_fee,
                jours_text,
            )
            footer_message = _sanitize_legal_footer_for_iban(raw_footer)
            if iban_value and "IBAN" not in footer_message:
                footer_message += (
                    f"<br/>Paiement par virement bancaire : IBAN : {iban_value}"
                )
        else:
            footer_message = _build_default_legal_footer_html(
                payment_terms_days, overdue_fee, iban_value
            )
            if not iban_value:
                app_logger.warning(
                    "PDF (detailed): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )

        # Pied de page légal : dessiné en zone fixe (marge inférieure)
        mention = None
        footer_cb_det = _make_legal_footer_page_callback(
            footer_message,
            mention,
            centered_style,
            contact_bar=contact_bar_det,
        )

        def _on_first_page_det(canvas: Any, doc: Any) -> None:
            footer_cb_det(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # === QR-BILL ===
        story.append(PageBreak())
        story.append(Spacer(1, QR_BILL_SPACER_PT))

        _perf_qrd_start = perf_counter()
        try:
            qr_bill_service = self.qrbill_service
            qr_override_d = (
                reminder_ctx.get("reminder_total_due")
                if reminder_ctx.get("is_reminder")
                else None
            )
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(
                invoice, override_amount=qr_override_d
            )
            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)
        app_logger.info(
            "[PDF_PERF] qr_bill_section_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_qrd_start) * 1000),
            getattr(invoice, "id", None),
        )

        _perf_build_d_start = perf_counter()
        doc.build(story, onFirstPage=_on_first_page_det)
        app_logger.info(
            "[PDF_PERF] doc_build_ms=%s invoice_id=%s",
            int((perf_counter() - _perf_build_d_start) * 1000),
            getattr(invoice, "id", None),
        )
        buffer.seek(0)
        # ✅ Calculer nb_rows depuis consolidated_lines (après regroupement aller/retour)
        nb_rows = len(consolidated_lines) if consolidated_lines else 0
        return (buffer.getvalue(), nb_rows)

    def _create_swiss_qr_bill_layout(self, invoice, billing_settings, qr_image):
        """Crée le layout authentique du QR-Bill suisse."""
        # ruff: noqa: I001
        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.platypus import (
            Paragraph,
            Spacer,
        )

        font_name, font_name_bold = _ensure_dejavu_pdf_fonts()

        styles = getSampleStyleSheet()

        # Style pour le texte normal
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=2,
            fontName=font_name,
        )

        # Style pour les titres de section
        section_title_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Normal"],
            fontSize=12,
            fontName=font_name_bold,
            alignment=TA_LEFT,
            spaceAfter=8,
            textColor=colors.black,
        )

        # Style pour les labels
        label_style = ParagraphStyle(
            "Label",
            parent=styles["Normal"],
            fontSize=8,
            fontName=font_name,
            alignment=TA_LEFT,
            spaceAfter=2,
            textColor=colors.black,
        )

        # Style pour les valeurs
        value_style = ParagraphStyle(
            "Value",
            parent=styles["Normal"],
            fontSize=8,
            fontName=font_name_bold,
            alignment=TA_LEFT,
            spaceAfter=4,
            textColor=colors.black,
        )

        # === SECTION GAUCHE: EMPFANGSSCHEIN (Reçu) ===
        left_section = []

        # Titre
        left_section.append(Paragraph("Empfangsschein", section_title_style))

        # Informations créancier
        left_section.append(Paragraph("Konto / Zahlbar an", label_style))
        left_section.append(
            Paragraph(
                _xml_escape_for_paragraph(billing_settings.iban or ""), value_style
            )
        )
        company = invoice.company
        left_section.append(
            Paragraph(
                _xml_escape_for_paragraph(company.name or "[Nom non configuré]"),
                normal_style,
            )
        )
        # Utiliser l'adresse de domiciliation
        street = (
            company.domicile_address_line1
            or company.address
            or "[Adresse non configurée]"
        )
        left_section.append(Paragraph(_xml_escape_for_paragraph(street), normal_style))
        postal_city = (
            f"{company.domicile_zip or ''} {company.domicile_city or ''}".strip()
            or "[Code postal/ville non configuré]"
        )
        left_section.append(
            Paragraph(_xml_escape_for_paragraph(postal_city), normal_style)
        )
        left_section.append(Spacer(1, 8))

        # Informations débiteur
        left_section.append(Paragraph("Zahlbar durch", label_style))
        _zbd_name = (
            f"{invoice.client.user.first_name or ''} "
            f"{invoice.client.user.last_name or ''}"
        ).strip()
        left_section.append(
            Paragraph(_xml_escape_for_paragraph(_zbd_name), normal_style)
        )
        left_section.append(
            Paragraph(
                _xml_escape_for_paragraph(
                    invoice.client.domicile_address or "Adresse non renseignée"
                ),
                normal_style,
            )
        )
        _zbd_pc = (
            f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}"
        ).strip()
        left_section.append(Paragraph(_xml_escape_for_paragraph(_zbd_pc), normal_style))
        left_section.append(Spacer(1, 8))

        # Référence
        left_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_section.append(Paragraph(_xml_escape_for_paragraph(qr_ref), value_style))
        left_section.append(Spacer(1, 8))

        # Montant
        left_section.append(Paragraph("Währung", label_style))
        left_section.append(Paragraph("CHF", value_style))
        left_section.append(Paragraph("Betrag", label_style))
        left_section.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))
        left_section.append(Spacer(1, 20))

        # Annahmestelle
        left_section.append(
            Paragraph(
                "Annahmestelle",
                ParagraphStyle(
                    "Center", parent=styles["Normal"], fontSize=8, alignment=TA_CENTER
                ),
            )
        )

        # === SECTION DROITE: ZAHLTEIL (Partie paiement) ===
        right_section = []

        # Titre
        right_section.append(Paragraph("Zahlteil", section_title_style))

        # Informations créancier
        right_section.append(Paragraph("Konto / Zahlbar an", label_style))
        right_section.append(
            Paragraph(
                _xml_escape_for_paragraph(billing_settings.iban or ""), value_style
            )
        )
        right_section.append(
            Paragraph(
                _xml_escape_for_paragraph(company.name or "[Nom non configuré]"),
                normal_style,
            )
        )
        right_section.append(Paragraph(_xml_escape_for_paragraph(street), normal_style))
        right_section.append(
            Paragraph(_xml_escape_for_paragraph(postal_city), normal_style)
        )
        right_section.append(Spacer(1, 8))

        # QR Code
        right_section.append(qr_image)
        right_section.append(Spacer(1, 8))

        # Informations débiteur
        right_section.append(Paragraph("Zahlbar durch", label_style))
        right_section.append(
            Paragraph(_xml_escape_for_paragraph(_zbd_name), normal_style)
        )
        right_section.append(
            Paragraph(
                _xml_escape_for_paragraph(
                    invoice.client.domicile_address or "Adresse non renseignée"
                ),
                normal_style,
            )
        )
        right_section.append(
            Paragraph(_xml_escape_for_paragraph(_zbd_pc), normal_style)
        )
        right_section.append(Spacer(1, 8))

        # Référence
        right_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_section.append(Paragraph(_xml_escape_for_paragraph(qr_ref), value_style))
        right_section.append(Spacer(1, 8))

        # Montant
        right_section.append(Paragraph("Währung", label_style))
        right_section.append(Paragraph("CHF", value_style))
        right_section.append(Paragraph("Betrag", label_style))
        right_section.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))

        # === LIGNE DE COUPE ===
        cut_line = [
            Paragraph(
                "✂",
                ParagraphStyle(
                    "CutLine", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER
                ),
            )
        ]

        # Retourner les données du tableau
        return [[left_section, cut_line, right_section]]

    def _create_official_swiss_qr_bill(self, invoice, billing_settings, qr_image):
        """Crée un QR-Bill suisse officiel avec le format exact."""
        # ruff: noqa: I001
        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        font_name, font_name_bold = _ensure_dejavu_pdf_fonts()

        styles = getSampleStyleSheet()

        # Styles spécifiques pour le QR-Bill suisse
        title_style = ParagraphStyle(
            "QRTitle",
            parent=styles["Normal"],
            fontSize=11,
            fontName=font_name_bold,
            alignment=TA_LEFT,
            spaceAfter=6,
            textColor=colors.black,
        )

        label_style = ParagraphStyle(
            "QRLabel",
            parent=styles["Normal"],
            fontSize=7,
            fontName=font_name,
            alignment=TA_LEFT,
            spaceAfter=1,
            textColor=colors.black,
        )

        value_style = ParagraphStyle(
            "QRValue",
            parent=styles["Normal"],
            fontSize=7,
            fontName=font_name_bold,
            alignment=TA_LEFT,
            spaceAfter=3,
            textColor=colors.black,
        )

        normal_text_style = ParagraphStyle(
            "QRNormal",
            parent=styles["Normal"],
            fontSize=7,
            fontName=font_name,
            alignment=TA_LEFT,
            spaceAfter=1,
            textColor=colors.black,
        )

        # === CONSTRUCTION DU QR-BILL ===

        # Section gauche - Empfangsschein
        left_content = []
        left_content.append(Paragraph("Empfangsschein", title_style))
        left_content.append(Spacer(1, 4))

        # Konto / Zahlbar an
        left_content.append(Paragraph("Konto / Zahlbar an", label_style))
        left_content.append(
            Paragraph(
                _xml_escape_for_paragraph(billing_settings.iban or ""), value_style
            )
        )
        company = invoice.company
        left_content.append(
            Paragraph(
                _xml_escape_for_paragraph(company.name or "[Nom non configuré]"),
                normal_text_style,
            )
        )
        street = (
            company.domicile_address_line1
            or company.address
            or "[Adresse non configurée]"
        )
        left_content.append(
            Paragraph(_xml_escape_for_paragraph(street), normal_text_style)
        )
        postal_city = (
            f"{company.domicile_zip or ''} {company.domicile_city or ''}".strip()
            or "[Code postal/ville non configuré]"
        )
        left_content.append(
            Paragraph(_xml_escape_for_paragraph(postal_city), normal_text_style)
        )
        left_content.append(Spacer(1, 6))

        # Zahlbar durch
        left_content.append(Paragraph("Zahlbar durch", label_style))
        _ofl_zbd = (
            f"{invoice.client.user.first_name or ''} "
            f"{invoice.client.user.last_name or ''}"
        ).strip()
        left_content.append(
            Paragraph(_xml_escape_for_paragraph(_ofl_zbd), normal_text_style)
        )
        left_content.append(
            Paragraph(
                _xml_escape_for_paragraph(
                    invoice.client.domicile_address or "Adresse non renseignée"
                ),
                normal_text_style,
            )
        )
        _ofl_zpc = (
            f"{invoice.client.domicile_zip or ''} {invoice.client.domicile_city or ''}"
        ).strip()
        left_content.append(
            Paragraph(_xml_escape_for_paragraph(_ofl_zpc), normal_text_style)
        )
        left_content.append(Spacer(1, 6))

        # Referenz
        left_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_content.append(Paragraph(_xml_escape_for_paragraph(qr_ref), value_style))
        left_content.append(Spacer(1, 6))

        # Währung et Betrag
        left_content.append(Paragraph("Währung", label_style))
        left_content.append(Paragraph("CHF", value_style))
        left_content.append(Paragraph("Betrag", label_style))
        left_content.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))
        left_content.append(Spacer(1, 20))

        # Annahmestelle
        left_content.append(
            Paragraph(
                "Annahmestelle",
                ParagraphStyle(
                    "Center", parent=styles["Normal"], fontSize=7, alignment=TA_CENTER
                ),
            )
        )

        # Section droite - Zahlteil
        right_content = []
        right_content.append(Paragraph("Zahlteil", title_style))
        right_content.append(Spacer(1, 4))

        # Konto / Zahlbar an
        right_content.append(Paragraph("Konto / Zahlbar an", label_style))
        right_content.append(
            Paragraph(
                _xml_escape_for_paragraph(billing_settings.iban or ""), value_style
            )
        )
        right_content.append(
            Paragraph(
                _xml_escape_for_paragraph(company.name or "[Nom non configuré]"),
                normal_text_style,
            )
        )
        right_content.append(
            Paragraph(_xml_escape_for_paragraph(street), normal_text_style)
        )
        right_content.append(
            Paragraph(_xml_escape_for_paragraph(postal_city), normal_text_style)
        )
        right_content.append(Spacer(1, 6))

        # QR Code
        right_content.append(qr_image)
        right_content.append(Spacer(1, 6))

        # Zahlbar durch
        right_content.append(Paragraph("Zahlbar durch", label_style))
        right_content.append(
            Paragraph(_xml_escape_for_paragraph(_ofl_zbd), normal_text_style)
        )
        right_content.append(
            Paragraph(
                _xml_escape_for_paragraph(
                    invoice.client.domicile_address or "Adresse non renseignée"
                ),
                normal_text_style,
            )
        )
        right_content.append(
            Paragraph(_xml_escape_for_paragraph(_ofl_zpc), normal_text_style)
        )
        right_content.append(Spacer(1, 6))

        # Referenz
        right_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_content.append(Paragraph(_xml_escape_for_paragraph(qr_ref), value_style))
        right_content.append(Spacer(1, 6))

        # Währung et Betrag
        right_content.append(Paragraph("Währung", label_style))
        right_content.append(Paragraph("CHF", value_style))
        right_content.append(Paragraph("Betrag", label_style))
        right_content.append(Paragraph(f"{invoice.total_amount:.2f}", value_style))

        # Créer le tableau avec ligne de coupe
        qr_bill_data = [[left_content, "", right_content]]

        # Tableau QR-Bill avec ligne de coupe
        qr_bill_table = Table(qr_bill_data, colWidths=[8.5 * cm, 0.3 * cm, 8.5 * cm])
        qr_bill_table.setStyle(
            TableStyle(
                [
                    # Bordures extérieures
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    # Ligne de coupe verticale
                    ("LINEBEFORE", (1, 0), (1, -1), 1, colors.black),
                    # Alignement
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    # Padding
                    ("PADDING", (0, 0), (0, -1), 8),  # Section gauche
                    ("PADDING", (2, 0), (2, -1), 8),  # Section droite
                    ("PADDING", (1, 0), (1, -1), 0),  # Ligne de coupe
                    # Fond blanc
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ]
            )
        )

        return qr_bill_table

    def _create_reminder_pdf_content(self, invoice, level, reminder=None):
        """⚠️ DÉPRÉCIÉ: Cette fonction n'est plus utilisée.

        Le PDF de rappel utilise maintenant le template facture via _create_invoice_pdf_content()
        avec les paramètres reminder_level, reminder_fee, reminder_total_due.

        Cette fonction est conservée uniquement pour rétrocompatibilité mais ne devrait plus être appelée.

        TODO: remove after v5.1 — ne pas laisser du code mort indéfiniment.

        Args:
            invoice: Facture principale
            level: Niveau du rappel (1, 2, 3)
            reminder: InvoiceReminder avec montants consolidés (optionnel pour rétrocompatibilité)
        """
        # Import ici pour éviter les problèmes de dépendances circulaires
        from io import BytesIO

        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
        )
        from reportlab.lib.pagesizes import (
            A4,
        )
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm
        )

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.red,
            alignment=TA_CENTER,
            spaceAfter=30,
        )

        # Contenu
        story = []

        # Titre selon le niveau
        if level == LEVEL_ONE:
            title = "RAPPEL DE PAIEMENT"
        elif level == LEVEL_THRESHOLD:
            title = "DEUXIÈME RAPPEL DE PAIEMENT"
        else:
            title = "DERNIER RAPPEL DE PAIEMENT"

        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # ✅ Informations enrichies du rappel
        invoice_info = [
            ["Numéro de facture:", invoice.invoice_number],
            ["Date d'émission initiale:", invoice.issued_at.strftime("%d.%m.%Y")],
            ["Nouvelle échéance:", invoice.due_date.strftime("%d.%m.%Y")],
        ]

        if reminder and reminder.total_due > 0:
            invoice_info.extend(
                [
                    [
                        "Montant facture initiale :",
                        f"CHF {reminder.principal_amount:.2f}",
                    ],
                    [
                        "Frais de rappel :",
                        f"CHF {reminder.reminder_fee_amount:.2f}",
                    ],
                    ["", ""],
                    ["Total à payer :", f"CHF {reminder.total_due:.2f}"],
                ]
            )
        elif invoice.reminder_fee_amount and invoice.reminder_fee_amount > 0:
            initial_amount = invoice.total_amount - invoice.reminder_fee_amount
            invoice_info.extend(
                [
                    ["Montant facture initiale :", f"CHF {initial_amount:.2f}"],
                    [
                        "Frais de rappel :",
                        f"CHF {invoice.reminder_fee_amount:.2f}",
                    ],
                    ["Solde total dû:", f"CHF {invoice.balance_due:.2f}"],
                ]
            )
        else:
            # Si frais = 0, afficher juste le solde dû
            invoice_info.append(["Solde total dû:", f"CHF {invoice.balance_due:.2f}"])

        invoice_table = Table(invoice_info, colWidths=[6 * cm, 6 * cm])
        invoice_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]
            )
        )

        story.append(invoice_table)
        story.append(Spacer(1, 30))

        # Informations du client
        client = invoice.client
        client_name = (
            f"{client.user.first_name or ''} {client.user.last_name or ''}".strip()
            or client.user.username
            or "Client"
        )

        story.append(
            Paragraph(
                f"Cher/Chère {_xml_escape_for_paragraph(client_name)},",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 20))

        # Message selon le niveau
        billing_settings = CompanyBillingSettings.query.filter_by(
            company_id=invoice.company_id
        ).first()

        # ✅ FIX: Utiliser getattr pour rétro-compatibilité et éviter AttributeError
        # Supporte à la fois reminder1template (ancien) et reminder1_template (nouveau)
        if level == LEVEL_ONE:
            template = None
            if billing_settings:
                # Essayer d'abord le nouveau nom (avec underscore)
                template = getattr(billing_settings, "reminder1_template", None)
                # Fallback vers l'ancien nom (sans underscore) pour rétro-compatibilité
                if not template:
                    template = getattr(billing_settings, "reminder1template", None)
            if template:
                message = template
            else:
                message = (
                    f"Nous vous rappelons que votre facture "
                    f"{invoice.invoice_number} d'un montant de "
                    f"{invoice.balance_due:.2f}"
                )
        elif level == LEVEL_THRESHOLD:
            template = None
            if billing_settings:
                # Essayer d'abord le nouveau nom (avec underscore)
                template = getattr(billing_settings, "reminder2_template", None)
                # Fallback vers l'ancien nom (sans underscore) pour rétro-compatibilité
                if not template:
                    template = getattr(billing_settings, "reminder2template", None)
            if template:
                message = template
            else:
                message = (
                    f"Conformément à nos CG, un montant de CHF 40.- a été ajouté "
                    f"à votre facture {invoice.invoice_number}. "
                    f"À défaut de règlement dans ce délai, une procédure de mise "
                    f"en demeure sera engagée."
                )
        else:
            # LEVEL_THREE ou autre
            template = None
            if billing_settings:
                # Essayer d'abord le nouveau nom (avec underscore)
                template = getattr(billing_settings, "reminder3_template", None)
                # Fallback vers l'ancien nom (sans underscore) pour rétro-compatibilité
                if not template:
                    template = getattr(billing_settings, "reminder3template", None)
            if template:
                message = template
            else:
                message = (
                    "Dernier rappel : Merci d'effectuer votre règlement net "
                    "sous 5 jours. En l'absence de paiement, une mise en "
                    "demeure sera engagée, entraînant des frais "
                    "supplémentaires et une éventuelle "
                    "procédure légale."
                )

        story.append(Paragraph(_reportlab_safe_footer_html(message), styles["Normal"]))
        story.append(Spacer(1, 20))

        # Informations bancaires
        if billing_settings and billing_settings.iban:
            banking_info = (
                f"Paiement par virement bancaire : IBAN : {billing_settings.iban}"
            )
            story.append(
                Paragraph(_xml_escape_for_paragraph(banking_info), styles["Normal"])
            )

        # ✅ QR-BILL pour le rappel consolidé (si reminder fourni avec montant total)
        if reminder and reminder.total_due > 0 and reminder.qr_reference:
            story.append(Spacer(1, 30))
            story.append(PageBreak())
            story.append(Spacer(1, QR_BILL_SPACER_PT))

            try:
                # Créer une facture "virtuelle" pour le QR-bill avec le montant total
                class VirtualInvoiceForReminder:
                    def __init__(self, invoice, reminder):  # pyright: ignore[reportMissingSuperCall]
                        self.id = reminder.id
                        self.invoice_number = f"{invoice.invoice_number}-R{level}"
                        self.company_id = invoice.company_id
                        self.company = invoice.company
                        self.client = invoice.client
                        self.client_id = invoice.client_id
                        self.bill_to_client_id = invoice.bill_to_client_id
                        self.billing_party_id = invoice.billing_party_id
                        self.billed_to_company_id = invoice.billed_to_company_id
                        self.billing_strategy = invoice.billing_strategy
                        self.total_amount = reminder.total_due
                        # Aligné avec QRBillService.resolve_qr_bill_amount_decimal (balance_due prioritaire)
                        self.balance_due = reminder.total_due
                        self.qr_reference = reminder.qr_reference

                virtual_invoice = VirtualInvoiceForReminder(invoice, reminder)
                qr_bill_svg_content = self.qrbill_service.generate_qr_bill_svg(
                    virtual_invoice
                )

                if qr_bill_svg_content:
                    drawing = _svg_content_to_drawing(qr_bill_svg_content)
                    if drawing:
                        story.append(_make_qr_bill_table(drawing))
                        app_logger.info("QR-bill ajouté au PDF de rappel consolidé")
            except Exception as e:
                app_logger.warning(
                    "Échec de l'ajout du QR-bill au PDF de rappel: %s", str(e)
                )
                # Ne pas bloquer si le QR-bill échoue

        # Générer le PDF (onFirstPage pour mode debug PDF_DEBUG_ENVELOPE=1)
        doc.build(story, onFirstPage=_on_first_page_debug_envelope)

        # Retourner le contenu et le nombre de lignes
        # Note: Les rappels n'ont pas de tableau de transports, donc nb_rows = 0
        buffer.seek(0)
        return (buffer.getvalue(), 0)
