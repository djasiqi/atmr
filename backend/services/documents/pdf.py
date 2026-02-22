import contextlib
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

from flask import current_app
from sqlalchemy.orm import joinedload

from infrastructure.invoices.invoice_calculator import round_to_5_cents
from models import Client, CompanyBillingSettings, Invoice, InvoiceLineType
from services.documents.invoice_template_builder import InvoiceTemplateBuilder
from services.documents.qrbill import QRBillService

LEVEL_ONE = 1
LEVEL_THRESHOLD = 2
MAX_PATIENT_NAME_LENGTH = 18
MONTHS_PER_YEAR = 12

app_logger = logging.getLogger("pdf_service")

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


def _make_legal_footer_page_callback(
    footer_message: str,
    mention: str | None,
    centered_style: Any,
) -> Any:
    """Crée un callback pour dessiner le pied de page légal en bas de page (zone fixe).

    Le pied de page est dessiné dans la marge inférieure, pas dans le flux du contenu.
    """

    def _draw_footer(canvas: Any, doc: Any) -> None:
        from reportlab.lib.units import cm

        from reportlab.platypus import Paragraph

        canvas.saveState()
        page_w = doc.pagesize[0]
        avail_width = page_w - 2 * cm
        y_pos = 1.2 * cm

        if footer_message:
            p = Paragraph(footer_message, centered_style)
            w, h = p.wrap(avail_width, 150)
            p.drawOn(canvas, (page_w - w) / 2, y_pos)
            y_pos += h + 6

        if mention:
            p2 = Paragraph(
                f'<font size="8" color="grey">{mention}</font>',
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
    status_str = (
        getattr(status_raw, "value", None) or str(status_raw) or ""
    )
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


def _format_patient_name_s2(patient_name: str) -> str:
    """Formate le nom patient pour S2 : **Prénom NOM** (point d'entrée visuel clinique).

    Accepte "Nom Prénom" (ex. Hauser Ariane, JASIQI Drin) ou "Prénom Nom" (ex. Ariane Hauser).
    Détection : si le premier token est tout en majuscules → "Nom Prénom", sinon "Prénom Nom".
    Retourne du HTML pour Paragraph : "<b>Prénom NOM</b>" ou "<b>Nom</b>".
    """
    if not patient_name or not patient_name.strip():
        return "Patient"
    s = patient_name.strip()
    parts = s.split(None, 1)
    if len(parts) == 2:  # noqa: PLR2004
        a, b = (parts[0] or "").strip(), (parts[1] or "").strip()
        if not a and not b:
            return "<b>Patient</b>"
        # "Nom Prénom" si premier token tout en majuscules (ex. HAUSER, JASIQI)
        if a and a.isupper() and len(a) > 1:
            prenom, nom = b.capitalize(), a.upper()
        else:
            prenom, nom = a.capitalize(), (b or a).upper()
        bits = [x for x in (prenom, nom) if x]
        return f"<b>{' '.join(bits)}</b>" if bits else "<b>Patient</b>"
    nom_only = (parts[0] or "").strip().upper()
    return f"<b>{nom_only}</b>" if nom_only else "<b>Patient</b>"


# Constantes pour la détection d'aller-retour
_MIN_ITEMS_FOR_ROUND_TRIP = 2
# Colonne Transport : pas de limite de caractères — le Paragraph wrap automatiquement
# sur plusieurs lignes selon transport_w (11.5 cm).
# Tolérance pour les montants (delta acceptable en CHF)
_AMOUNT_TOLERANCE_CHF = Decimal("5.00")
# Fenêtre temporelle maximale pour un aller-retour (en heures)
_MAX_ROUND_TRIP_TIME_WINDOW_HOURS = 12


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
        if _is_booking_cancelled(parent_booking) or _is_booking_cancelled(return_booking):
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
                    item["transport_display"] = _get_cancellation_transport_display(booking)
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

        # Chercher les paires aller-retour avec validations strictes
        matched_pairs = []
        used_indices = set()
        candidate_pairs = []  # Stocker toutes les paires candidates avant validation

        for i, pair1 in enumerate(normalized_pairs):
            if i in used_indices:
                continue
            for j, pair2 in enumerate(normalized_pairs[i + 1 :], start=i + 1):
                if j in used_indices:
                    continue
                # Vérifier si pickup1 == dropoff2 ET dropoff1 == pickup2 (aller-retour)
                if (
                    pair1["pickup_norm"] == pair2["dropoff_norm"]
                    and pair1["dropoff_norm"] == pair2["pickup_norm"]
                ):
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

            # Créer ligne consolidée A/R avec détails aller/retour
            # Déterminer quel est l'aller et quel est le retour
            # Dans la détection, on a vérifié que :
            # - pair1["pickup_norm"] == pair2["dropoff_norm"]
            # - pair1["dropoff_norm"] == pair2["pickup_norm"]
            # Donc pair1 : A → B et pair2 : B → A
            # L'aller est celui avec la date/heure la plus tôt
            date1 = item1.get("date")
            date2 = item2.get("date")

            if (
                date1
                and date2
                and isinstance(date1, datetime)
                and isinstance(date2, datetime)
            ):
                # Utiliser l'ordre temporel pour déterminer aller/retour
                if date1 <= date2:
                    # item1 est l'aller, item2 est le retour
                    pickup_aller = pair1["pickup_orig"]
                    dropoff_aller = pair1["dropoff_orig"]
                else:
                    # item2 est l'aller, item1 est le retour
                    pickup_aller = pair2["pickup_orig"]
                    dropoff_aller = pair2["dropoff_orig"]
            else:
                # Pas d'horaire disponible, utiliser l'ordre de détection
                pickup_aller = pair1["pickup_orig"]
                dropoff_aller = pair1["dropoff_orig"]

            raw_sum = item1.get("amount", Decimal("0")) + item2.get(
                "amount", Decimal("0")
            )
            amount_rounded = round_to_5_cents(Decimal(str(raw_sum)))
            short_a = _short_label_for_transport(pickup_aller)
            short_b = _short_label_for_transport(dropoff_aller)
            detail_a = _short_detail_label(pickup_aller)
            detail_b = _short_detail_label(dropoff_aller)
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
                "aller_detail": f"{short_a} → {short_b}",
                "retour_detail": f"{short_b} → {short_a}",
                "aller_detail_short": f"{detail_a} → {detail_b}",
                "retour_detail_short": f"{detail_b} → {detail_a}",
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
                    item["transport_display"] = _get_cancellation_transport_display(booking)
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
def _format_address_for_display(address: str) -> str:  # pyright: ignore[reportUnusedFunction]  # noqa: PLR0911
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


def _format_billed_to_three_lines(
    raw: str, company_country: str | None = None
) -> str:
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


def _load_logo_ratio_safe(  # noqa: PLR0911
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


def _name_with_uppercase_last_name(name: str) -> str:
    """Met le nom de famille (dernier mot) en majuscules pour le bloc « Facturé à »."""
    if not name or not str(name).strip():
        return name
    parts = name.strip().split()
    if not parts:
        return name
    parts[-1] = parts[-1].upper()
    return " ".join(parts)


def _get_billed_to(invoice: "Invoice") -> tuple[str, str]:
    """Retourne (nom, adresse formatée) pour le bloc « Facturé à »."""
    company_country = None
    if getattr(invoice, "company", None) and getattr(invoice.company, "domicile_country", None):
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
            or _bp_type in (BillingPartyType.CLINIC, BillingPartyType.EMS, BillingPartyType.HOSPITAL)
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
            _link = (
                ClientBillingParty.query.filter_by(
                    client_id=_client_id, billing_party_id=_bp_id
                ).first()
            )
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
                    client_name = _name_with_uppercase_last_name(client_name or "Client")
                    bp_name = _name_with_uppercase_last_name(bp.display_name or "Payeur")
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
                else:
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
                _tr = TransportRequest.query.filter_by(
                    institution_id=client.linked_institution_id,
                ).order_by(TransportRequest.id.desc()).first()
                if _tr and _tr.patient_id:
                    _ip = InstitutionPatient.query.get(_tr.patient_id)
                    if _ip:
                        if not _patient_name:
                            _patient_name = f"{_ip.first_name or ''} {_ip.last_name or ''}".strip()
                        parts = [_ip.address or "", _ip.postal_code or "", _ip.city or ""]
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
) -> tuple[Any | None, list[str]]:
    """Construit le flowable pour le bloc destinataire compatible zone C5.

    Zone fenêtre : uniquement nom + adresse (pas de label « Facturé à : »).
    - Filtre les lignes vides (no data => no UI).
    - Wrap via stringWidth/simpleSplit (font metrics ReportLab).
    - Ne dessine rien si aucune ligne utile.

    Returns:
        (Paragraph ou None, recipient_lines pour tests).
    """
    from reportlab.lib.units import mm

    name, addr = _get_billed_to(invoice)
    lines: list[str] = []
    if name and str(name).strip():
        for name_line in str(name).strip().split("\n"):
            if name_line.strip():
                lines.append(name_line.strip())
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
    font_size = getattr(normal_style, "fontSize", 10) or 10
    max_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
    max_lines = max(1, int(DEST_ADDR_ZONE_HEIGHT_MM / DEST_ADDR_LINE_HEIGHT_MM))
    max_chars_fallback = max(30, int(DEST_ADDR_MAX_WIDTH_MM * 3))

    visual_lines: list[str] = []
    for line in lines:
        if line == "":
            visual_lines.append("")
        else:
            wrapped = _wrap_line_by_width(line, font_name, font_size, max_width_pt)
            visual_lines.extend(wrapped)

    if len(visual_lines) > max_lines:
        visual_lines = visual_lines[:max_lines]
        last = visual_lines[-1]
        if len(last) + 1 > max_chars_fallback:
            truncated = last[: max_chars_fallback - 1]
            last = (
                truncated.rsplit(" ", 1)[0] + "…"
                if " " in truncated
                else truncated + "…"
            )
        else:
            last = last + "…"
        visual_lines[-1] = last

    # Zone fenêtre : pas de label « Facturé à : », uniquement destinataire
    parts: list[str] = []
    for vl in visual_lines:
        parts.append(vl)
        parts.append("<br/>")
    text = "".join(parts).rstrip("<br/>")
    from reportlab.platypus import Paragraph

    para = Paragraph(text, normal_style)
    return (para, lines)


def _build_s2_table(
    invoice: "Invoice",
    font_name: str,
    font_name_bold: str,
    s2_main_style: Any,
    *,
    include_non_ride: bool = False,
    available_width_pt: float | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Construit le tableau unifié (Date | Client/Patient | Transport | Montant).

    Utilisé pour toutes les factures (client et clinique).
    - Factures client : colonne "Client" = nom du client
    - Factures tierce partie/S2 : colonne "Patient" = nom du patient

    Si available_width_pt est fourni (ex. doc.width), la colonne Transport prend toute la largeur
    restante. Sinon, largeurs fixes de repli.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle
    from models import Booking

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
        booking = Booking.query.get(line.reservation_id)
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

    # ✅ Header dynamique : "Client" pour factures client, "Patient" pour factures tierce partie/S2
    patient_column_header = (
        "Client" if not (is_third_party_invoice or is_s2_invoice) else "Patient"
    )
    table_data: list[list[Any]] = [
        ["Date", patient_column_header, "Transport", "Montant"]
    ]
    s2_patient_separator_after_rows: list[int] = []
    for i, item in enumerate(consolidated):
        if i > 0:
            prev = consolidated[i - 1]
            pk_prev = (prev.get("patient_id"), prev.get("patient_name", ""))
            pk_cur = (item.get("patient_id"), item.get("patient_name", ""))
            if pk_prev != pk_cur:
                s2_patient_separator_after_rows.append(len(table_data))
        date_str = item["date"].strftime("%d.%m.%Y") if item.get("date") else ""
        pn_raw = item.get("patient_name", "Patient")
        if len(pn_raw) > MAX_PATIENT_NAME_LENGTH:
            pn_raw = pn_raw[: MAX_PATIENT_NAME_LENGTH - 1] + "."
        patient_cell = Paragraph(_format_patient_name_s2(pn_raw), s2_main_style)
        amt = item.get("amount") or Decimal("0")
        amount_val = f"{Decimal(amt):.2f}"
        is_ar = (
            item.get("is_round_trip")
            and item.get("aller_detail")
            and item.get("retour_detail")
        )
        if is_ar:
            base = item.get("transport_display", "")
            main_text = (
                f"{base}&nbsp;<font color='#aaaaaa' size='8'>↔</font>&nbsp;[A/R]"
            )
            para_main = Paragraph(f"<b>{main_text}</b>", s2_main_style)
            transport_cell = para_main
            amount_cell = Paragraph(
                f'<para align="right"><b>{amount_val}</b></para>',
                s2_main_style,
            )
        else:
            transport = item.get("transport_display", "")
            transport_cell = Paragraph(transport, s2_main_style)
            amount_cell = Paragraph(
                f'<para align="right">{amount_val}</para>',
                s2_main_style,
            )
        table_data.append([date_str, patient_cell, transport_cell, amount_cell])

    if include_non_ride:
        for line in invoice.lines:
            # LATE_FEE, REMINDER_FEE, CUSTOM (pas RIDE ni MATERIAL_DELIVERY)
            if line.type not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            ):
                amt = line.line_total if line.line_total is not None else Decimal("0")
                amount = f"{Decimal(amt):.2f}"
                table_data.append(["", "", line.description[:30], amount])

    # Date et Client inchangés (éviter décalage gauche). Montant réduit pour donner
    # plus d'espace à Transport à droite, sans déplacer les colonnes de gauche.
    date_w = 2 * cm
    patient_w = 3 * cm
    amount_w = 2 * cm
    if available_width_pt is not None and available_width_pt > 0:
        rest = available_width_pt - (date_w + patient_w + amount_w)
        transport_w = max(rest, 1 * cm)
    else:
        transport_w = 11.5 * cm
    col_widths = [date_w, patient_w, transport_w, amount_w]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    style_rules: list[Any] = [
        ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 0), (-1, 0), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("FONTNAME", (0, 1), (-1, -1), font_name),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 1),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (0, -1), 0),  # Colonne Date : alignement marge gauche
        ("LEFTPADDING", (1, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (2, 0), (2, -1), 1),
        ("RIGHTPADDING", (2, 0), (2, -1), 10),
        ("LEFTPADDING", (3, 0), (3, -1), 4),
    ]
    for r in s2_patient_separator_after_rows:
        style_rules.append(("LINEBELOW", (0, r), (-1, r), 0.15, colors.lightgrey))
    tbl.setStyle(TableStyle(style_rules))
    return (tbl, consolidated)


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

    En mode rappel : mini-table structurée
    « Montant facture initiale » + « Frais de rappel N°X » + « TOTAL À FACTURER ».
    """
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

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
    total_amt = f"CHF {final_total:.2f}" if is_s2 else f"{final_total:.2f}"

    reminder_fee_label = (
        f"Frais de rappel N°{reminder_level} :"
        if is_reminder and reminder_level
        else "Frais de rappel :"
    )
    reminder_fee_amt = f"CHF {reminder_fee_float:.2f}"
    principal_amt = f"CHF {principal_float:.2f}"

    # ✅ Mode rappel : mini-table structurée (Sous-total facture + Frais + TOTAL)
    if is_reminder:
        principal_label = "Montant facture initiale :"
        if template == "detailed":
            if is_third_party:
                total_data = [
                    ["", "", "", "", principal_label, principal_amt],
                    ["", "", "", "", reminder_fee_label, reminder_fee_amt],
                    ["", "", "", "", total_label, total_amt],
                ]
                col_widths = [2 * cm, 2.5 * cm, 3 * cm, 3 * cm, 2.5 * cm, 2 * cm]
            else:
                total_data = [
                    ["", "", "", principal_label, principal_amt],
                    ["", "", "", reminder_fee_label, reminder_fee_amt],
                    ["", "", "", total_label, total_amt],
                ]
                col_widths = [2.5 * cm, 4 * cm, 4 * cm, 2.5 * cm, 2.5 * cm]
        elif is_third_party:
            total_data = [
                ["", "", "", principal_label, principal_amt],
                ["", "", "", reminder_fee_label, reminder_fee_amt],
                ["", "", "", total_label, total_amt],
            ]
            col_widths = [2 * cm, 3 * cm, 4.5 * cm, 2 * cm, 2.5 * cm]
        else:
            total_data = [
                ["", "", principal_label, principal_amt],
                ["", "", reminder_fee_label, reminder_fee_amt],
                ["", "", total_label, total_amt],
            ]
            col_widths = [2.5 * cm, 6 * cm, 2.5 * cm, 2.5 * cm]
    elif template == "detailed":
        if is_third_party:
            if vat_is_applicable:
                total_data = [
                    ["", "", "", "", "Sous-total :", f"{subtotal:.2f}"],
                    ["", "", "", "", f"{vat_label_display} :", f"{vat_total:.2f}"],
                    ["", "", "", "", total_label, total_amt],
                ]
            else:
                total_data = [["", "", "", "", total_label, total_amt]]
            col_widths = [2 * cm, 2.5 * cm, 3 * cm, 3 * cm, 2.5 * cm, 2 * cm]
        else:
            if vat_is_applicable:
                total_data = [
                    ["", "", "", "Sous-total :", f"{subtotal:.2f}"],
                    ["", "", "", f"{vat_label_display} :", f"{vat_total:.2f}"],
                    ["", "", "", total_label, total_amt],
                ]
            else:
                total_data = [["", "", "", total_label, total_amt]]
            col_widths = [2.5 * cm, 4 * cm, 4 * cm, 2.5 * cm, 2.5 * cm]
    elif is_third_party:
        if vat_is_applicable:
            total_data = [
                ["", "", "", "Sous-total :", f"{subtotal:.2f}"],
                ["", "", "", f"{vat_label_display} :", f"{vat_total:.2f}"],
                ["", "", "", total_label, total_amt],
            ]
        else:
            total_data = [["", "", "", total_label, total_amt]]
        col_widths = [2 * cm, 3 * cm, 4.5 * cm, 2 * cm, 2.5 * cm]
    else:
        if vat_is_applicable:
            total_data = [
                ["", "", "Sous-total :", f"{subtotal:.2f}"],
                ["", "", f"{vat_label_display} :", f"{vat_total:.2f}"],
                ["", "", total_label, total_amt],
            ]
        else:
            total_data = [["", "", total_label, total_amt]]
        col_widths = [2.5 * cm, 6 * cm, 2.5 * cm, 2.5 * cm]

    total_table = Table(total_data, colWidths=col_widths)
    if template == "detailed":
        if is_third_party or is_s2:
            style_rules = [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (4, 0), (5, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        else:
            style_rules = [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (3, 0), (4, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
    elif is_third_party:
        style_rules = [
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (3, 0), (4, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]
    else:
        style_rules = [
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (2, 0), (3, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]
    # ✅ Style des polices : dernière ligne (total) en gras, autres en normal
    # Si mode rappel, la ligne de frais est aussi en normal, seule la dernière ligne (total) est en gras
    if vat_is_applicable or is_reminder:
        # Plusieurs lignes : toutes sauf la dernière en normal, dernière en gras
        style_rules.extend(
            [
                ("FONTNAME", (0, 0), (-1, -2), font_name),
                ("FONTNAME", (0, -1), (-1, -1), font_name_bold),
            ]
        )
    else:
        # Une seule ligne : en gras
        style_rules.append(("FONTNAME", (0, 0), (-1, -1), font_name_bold))
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

    def generate_invoice_pdf(self, invoice, *, force_regenerate: bool = False):
        """Génère le PDF d'une facture.

        Args:
            invoice: La facture pour laquelle générer le PDF
            force_regenerate: Si True, régénère même si pdf_url existe déjà
                (utilisé par l'endpoint regenerate-pdf)

        ⚠️ PROTECTION IMMUTABILITÉ:
        Ne modifie JAMAIS invoice.pdf_url si:
        - invoice.status est SENT, PARTIALLY_PAID, ou PAID (facture verrouillée)
        - invoice.pdf_url existe déjà ET force_regenerate=False
        """
        from models.enums import InvoiceStatus

        # ✅ Garde-fou 2: Log explicite pour diagnostic (invoice_id, force_regenerate, action)
        invoice_id = getattr(invoice, "id", None)
        has_existing_pdf_url = bool(getattr(invoice, "pdf_url", None))
        app_logger.info(
            "[PDF] generate_invoice_pdf entry: invoice_id=%s, force_regenerate=%s, has_existing_pdf_url=%s",
            invoice_id,
            force_regenerate,
            has_existing_pdf_url,
        )

        # ✅ PROTECTION: Vérifier si la facture est "verrouillée" (déjà envoyée/payée)
        locked_statuses = {
            InvoiceStatus.SENT,
            InvoiceStatus.PARTIALLY_PAID,
            InvoiceStatus.PAID,
        }
        if invoice.status in locked_statuses:
            app_logger.warning(
                "[PDF PROTECTION] Tentative de régénération PDF pour facture verrouillée: invoice_id=%s, status=%s, pdf_url=%s. Action=SKIP_LOCKED",
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
                    joinedload(Invoice.lines),
                    joinedload(Invoice.payments),
                )
                .filter_by(id=invoice.id)
                .first()
            )

            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # ════════════════════════════════════════════════════════════════════
            # FILET DE SÉCURITÉ: Recalculer les totaux si incohérents
            # ════════════════════════════════════════════════════════════════════
            # Protège contre les factures avec totaux à 0 alors que des lignes existent.
            from infrastructure.invoices.invoice_calculator import recompute_invoice_totals

            if invoice.lines and (invoice.total_amount or Decimal("0.00")) == Decimal("0.00"):
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

            pdf_bytes: bytes = pdf_content if isinstance(pdf_content, bytes) else b""
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
                    joinedload(Invoice.lines),
                    joinedload(Invoice.payments),
                    joinedload(Invoice.reminders),
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

            pdf_bytes: bytes = pdf_content if isinstance(pdf_content, bytes) else b""
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
        display_reminder_level = f"RAPPEL N°{level}" if level else None
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

        if template_variant == "minimal":
            return self._create_minimal_invoice_pdf(
                invoice, billing_settings, reminder_ctx
            )
        if template_variant == "detailed":
            return self._create_detailed_invoice_pdf(
                invoice, billing_settings, reminder_ctx
            )
        return self._create_standard_invoice_pdf(
            invoice, billing_settings, reminder_ctx
        )

    def _create_standard_invoice_pdf(
        self,
        invoice,
        billing_settings,
        reminder_ctx: dict[str, Any],
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
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import (
            TTFont,
        )
        from reportlab.platypus import (
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        # ✅ Enregistrer une police TrueType pour supporter l'Unicode (caractères accentués)
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
            font_name = "DejaVuSans"
            font_name_bold = "DejaVuSans-Bold"
        except Exception:
            font_name = "Helvetica"
            font_name_bold = "Helvetica-Bold"

        buffer = BytesIO()

        # Styles basés sur le design de référence
        styles = getSampleStyleSheet()

        # Style pour le texte normal (leftIndent=0 pour alignement marge gauche)
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
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
            fontSize=10,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName=font_name,
        )
        s2_main_style = ParagraphStyle(
            "S2Main",
            parent=styles["Normal"],
            fontSize=9,
            leading=9.5,
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
        recipient_para, _ = _build_recipient_block_flowable(invoice, normal_style)
        recipient_top_padding_mm = 25.0  # destinataire légèrement plus bas
        recipient_left_padding_mm = 15.0  # déplace le bloc destinataire vers la droite (pas d'espace volé à l'expéditeur)
        dest_width_pt = DEST_ADDR_MAX_WIDTH_MM * mm
        page_width_pt = A4[0]
        usable_width_pt = page_width_pt - 2 * cm - 2 * cm
        company_width_pt = (
            usable_width_pt - dest_width_pt
        )  # expéditeur garde toute sa largeur

        vat_line = f"<br/>{vat_status_text}" if vat_status_text else ""
        company_info_left = f"""
        {company_name}<br/>
        {company_address}<br/>
        {company_email}<br/>
        {company_phone}<br/>
        IDE/UID : {company_uid}{vat_line}
        """
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
                fontSize=8,
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
        story.append(Spacer(1, 20))

        # === BANDEAU RAPPEL (si mode rappel) ===
        display_reminder_level = reminder_ctx.get("display_reminder_level")
        if display_reminder_level:
            bandeau = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "Bandeau",
                                parent=styles["Normal"],
                                fontSize=14,
                                fontName=font_name_bold,
                                alignment=TA_CENTER,
                                textColor=colors.HexColor("#8B0000"),
                            ),
                        )
                    ]
                ],
                colWidths=[17 * cm],
            )
            bandeau.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFE5E5")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CC0000")),
                    ]
                )
            )
            story.append(bandeau)
            story.append(Spacer(1, 12))

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
        invoice_info_left = f"""
        <b>Numéro de facture :</b> {invoice.invoice_number}<br/>
        <b>Date d'émission :</b> {invoice.issued_at.strftime("%d.%m.%Y")}<br/>
        <b>{echeance_label}</b> {invoice.due_date.strftime("%d.%m.%Y")}<br/>
        <b>Période de facturation :</b> {period_label}
        """

        story.append(Paragraph(invoice_info_left, normal_style))
        story.append(Spacer(1, 20))

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
        s2_table, consolidated_lines = _build_s2_table(
            invoice,
            font_name,
            font_name_bold,
            s2_main_style,
            include_non_ride=False,
            available_width_pt=usable_width_pt,
        )

        # ✅ Utiliser toujours le tableau unifié S2
        story.append(Paragraph("<b>DÉTAIL DES TRANSPORTS</b>", normal_style))
        story.append(Spacer(1, 6))
        story.append(s2_table)
        story.append(Spacer(1, 2))

        # === TOTAL ===
        if reminder_ctx.get("is_reminder"):
            story.append(Spacer(1, 16))  # Respiration avant totaux (rappel)
        total_separator = Table([[""]], colWidths=[17 * cm])
        total_separator.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (0, 0), 1, colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(total_separator)
        story.append(Spacer(1, 8))

        if consolidated_lines:
            note_para = Paragraph(
                '<font size="7" color="grey">[A/R] = transport aller-retour</font>',
                normal_style,
            )
            story.append(note_para)
            story.append(Spacer(1, 8))

        # ✅ Utiliser toujours le format unifié pour les totaux (même libellé "TOTAL À FACTURER")
        # ✅ Si mode rappel, ajouter la ligne de frais de rappel
        total_table = _build_totals_table(
            invoice,
            True,
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

        # Message du pied de page : utiliser legal_footer si disponible,
        # sinon message dynamique. En mode rappel, texte dédié (soft mais ferme).
        if display_reminder_level:
            footer_message = (
                "Sauf erreur de notre part, cette facture est restée impayée à ce jour. "
                "Nous vous remercions de bien vouloir procéder à son règlement dans les plus brefs délais. "
                "Des frais de rappel ont été ajoutés conformément à nos conditions générales."
            )
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
            base = (
                f"En votre aimable règlement net sous {payment_terms_days} "
                f"{jours_text} avec nos remerciements anticipés. "
                f"En cas de retard de paiement, des frais de rappel d'un montant "
                f"de CHF {overdue_fee:.2f} vous seront facturés, "
                f"conformément à nos conditions générales."
            )
            if iban_value:
                footer_message = (
                    f"{base} Paiement par virement bancaire : IBAN : {iban_value}"
                )
            else:
                footer_message = base
                app_logger.warning(
                    "PDF (standard): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )

        # Pied de page légal : dessiné en zone fixe (marge inférieure), pas dans le flux
        mention = None
        if display_reminder_level:
            mention = f"Document généré automatiquement – facture initiale n° {invoice.invoice_number} inchangée."
        footer_cb = _make_legal_footer_page_callback(
            footer_message, mention, centered_style
        )

        def _on_first_page(canvas: Any, doc: Any) -> None:
            footer_cb(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # Doc avec page QR-Bill dédiée (marge bas 2 cm, pas de pied légal)
        doc = _make_invoice_doc_with_qrbill_page(
            buffer,
            top_margin_cm=2,
            bottom_margin_cm=2.5,  # Réserve espace pour pied de page légal
            left_margin_cm=2,
            right_margin_cm=2,
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

        try:
            # Générer le QR-Bill suisse officiel avec la vraie bibliothèque
            qr_bill_service = self.qrbill_service
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(invoice)

            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
            else:
                story.append(Paragraph("QR-Bill non disponible", normal_style))

        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)
            story.append(Paragraph("QR-Bill non disponible", normal_style))

        # Générer le PDF (callbacks dans PageTemplates)
        doc.build(story)

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
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import (
            TTFont,
        )
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        # ✅ Enregistrer une police TrueType pour supporter l'Unicode (caractères accentués)
        try:
            pdfmetrics.registerFont(
                TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            )
            font_name = "DejaVuSans"
        except Exception:
            # Fallback sur Helvetica si DejaVu n'est pas disponible
            font_name = "Helvetica"

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=2.5 * cm,  # Réserve espace pour pied de page légal
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
        )

        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=9,
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
            fontSize=9,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=4,
            fontName=font_name,
        )

        story = []
        company = invoice.company

        # === EN-TÊTE SIMPLIFIÉ (SANS LOGO) : ENTREPRISE (gauche) | DESTINATAIRE (droite) ===
        company_name = company.name or "[Nom entreprise non configuré]"
        company_address = self._get_company_address_for_pdf(company)
        company_info = f"{company_name}<br/>{company_address}"
        company_para_min = Paragraph(company_info, normal_style)

        recipient_para_min, _ = _build_recipient_block_flowable(invoice, normal_style)
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
                fontSize=8,
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

        # === Contexte rappel (défini avant Numéro de facture / Facture) ===
        display_reminder_level = reminder_ctx.get("display_reminder_level")
        is_reminder = reminder_ctx.get("is_reminder", False)

        # === BANDEAU RAPPEL (si mode rappel) ===
        if display_reminder_level:
            bandeau = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "BandeauM",
                                parent=styles["Normal"],
                                fontSize=14,
                                fontName="Helvetica-Bold",
                                alignment=TA_CENTER,
                                textColor=colors.HexColor("#8B0000"),
                            ),
                        )
                    ]
                ],
                colWidths=[17 * cm],
            )
            bandeau.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFE5E5")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CC0000")),
                    ]
                )
            )
            story.append(bandeau)
            story.append(Spacer(1, 12))

        # === INFORMATIONS FACTURE (SIMPLIFIÉES) ===
        echeance_label = "Échéance initiale:" if is_reminder else "Échéance:"
        invoice_info = (
            f"<b>Facture {invoice.invoice_number}</b> - "
            f"{invoice.issued_at.strftime('%d.%m.%Y')} - "
            f"{echeance_label} {invoice.due_date.strftime('%d.%m.%Y')}"
        )
        story.append(Paragraph(invoice_info, normal_style))
        story.append(Spacer(1, 15))

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
        table_data = (
            [["Date", "Patient", "Montant"]]
            if is_third_party
            else [["Date", "Montant"]]
        )

        for line in invoice.lines:
            if (
                line.type
                in (
                    InvoiceLineType.RIDE,
                    InvoiceLineType.MATERIAL_DELIVERY,
                )
                and line.reservation_id
            ):
                from models import Booking

                booking = Booking.query.get(line.reservation_id)
                if booking:
                    date_str = (
                        booking.scheduled_time.strftime("%d/%m/%Y")
                        if booking.scheduled_time
                        else ""
                    )
                    amount = f"{line.line_total:.2f}"
                    if is_third_party:
                        # ✅ S2: Utiliser le snapshot patient_name depuis line.meta (traçabilité juridique)
                        # Ne jamais recalculer depuis booking.client.user (le nom peut avoir changé)
                        patient_name = "Patient"
                        if hasattr(line, "meta") and isinstance(line.meta, dict):
                            # Priorité: meta.patient_name (snapshot S2) > booking.customer_name > booking.client.user
                            patient_name = (
                                line.meta.get("patient_name")
                                or booking.customer_name
                                or (
                                    f"{booking.client.user.first_name or ''} "
                                    f"{booking.client.user.last_name or ''}"
                                ).strip()
                                or "Patient"
                            )
                        else:
                            # Fallback si meta n'existe pas (rétro-compatibilité)
                            patient_name = (
                                booking.customer_name
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
                        table_data.append([date_str, patient_name, amount])
                    else:
                        table_data.append([date_str, amount])
                else:
                    amount = f"{line.line_total:.2f}"
                    if is_third_party:
                        table_data.append(["", "N/A", amount])
                    else:
                        table_data.append(["", amount])
            else:
                amount = f"{line.line_total:.2f}"
                if is_third_party:
                    table_data.append(["", "N/A", amount])
                else:
                    table_data.append(["", amount])

        if is_third_party:
            services_table = Table(table_data, colWidths=[3 * cm, 4 * cm, 2.5 * cm])
        else:
            services_table = Table(table_data, colWidths=[4 * cm, 2.5 * cm])
        services_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                    ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.lightgrey),
                ]
            )
        )
        story.append(services_table)
        story.append(Spacer(1, 10))

        # === TOTAL SIMPLIFIÉ ===
        # ✅ Mode rappel : mini-table (Sous-total facture + Frais + TOTAL)
        if (
            is_reminder
            and reminder_ctx.get("reminder_fee") is not None
            and reminder_ctx.get("reminder_total_due") is not None
            and reminder_ctx.get("reminder_principal") is not None
        ):
            principal_float = float(reminder_ctx["reminder_principal"])
            reminder_fee_float = float(reminder_ctx["reminder_fee"])
            final_total = float(reminder_ctx["reminder_total_due"])
            level = reminder_ctx.get("reminder_level")
            reminder_fee_label = (
                f"Frais de rappel N°{level} :" if level else "Frais de rappel :"
            )
            total_data = [
                ["Montant facture initiale :", f"CHF {principal_float:.2f}"],
                [reminder_fee_label, f"CHF {reminder_fee_float:.2f}"],
                ["TOTAL :", f"CHF {final_total:.2f}"],
            ]
        else:
            total_amount = float(invoice.total_amount)
            total_data = [["TOTAL :", f"{total_amount:.2f} CHF"]]
        total_table = Table(total_data, colWidths=[4 * cm, 2.5 * cm])
        style_rules = [
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]
        if is_reminder:
            style_rules.extend(
                [
                    ("FONTNAME", (0, 0), (-1, -2), font_name),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]
            )
        else:
            style_rules.append(("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"))
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
            footer_message = (
                "Sauf erreur de notre part, cette facture est restée impayée à ce jour. "
                "Merci de procéder à son règlement dans les plus brefs délais. "
                "Des frais de rappel ont été ajoutés conformément à nos conditions générales."
            )
            iban_value_min = (
                billing_settings.iban
                if billing_settings and billing_settings.iban
                else None
            )
            if iban_value_min:
                footer_message += f" IBAN: {iban_value_min}"
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
            if iban_value_min:
                footer_message = f"Merci de votre règlement. IBAN: {iban_value_min}"
            else:
                footer_message = "Merci de votre règlement."
                app_logger.warning(
                    "PDF (minimal): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )

        # Pied de page légal : dessiné en zone fixe (marge inférieure)
        mention = None
        if is_reminder:
            mention = f"Document généré automatiquement – facture initiale n° {invoice.invoice_number} inchangée."
        footer_cb_min = _make_legal_footer_page_callback(
            footer_message, mention, centered_style
        )

        def _on_first_page_min(canvas: Any, doc: Any) -> None:
            footer_cb_min(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # === QR-BILL (SIMPLIFIÉ) ===
        story.append(PageBreak())
        story.append(Spacer(1, QR_BILL_SPACER_PT))

        try:
            qr_bill_service = self.qrbill_service
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(invoice)
            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)

        doc.build(story, onFirstPage=_on_first_page_min)
        buffer.seek(0)
        return buffer.getvalue()

    def _create_detailed_invoice_pdf(
        self,
        invoice,
        billing_settings,
        reminder_ctx: dict[str, Any],
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
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import (
            TTFont,
        )
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        # ✅ Enregistrer une police TrueType pour supporter l'Unicode (caractères accentués)
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
            font_name = "DejaVuSans"
            font_name_bold = "DejaVuSans-Bold"
        except Exception:
            font_name = "Helvetica"
            font_name_bold = "Helvetica-Bold"

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2.5 * cm,  # Réserve espace pour pied de page légal
            leftMargin=2 * cm,
            rightMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
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
            fontSize=10,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName=font_name,
        )
        detail_style = ParagraphStyle(
            "Detail",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.darkgrey,
            alignment=TA_LEFT,
            spaceAfter=4,
            fontName=font_name,
        )
        s2_main_style = ParagraphStyle(
            "S2Main",
            parent=styles["Normal"],
            fontSize=9,
            leading=9.5,
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
        recipient_para, _ = _build_recipient_block_flowable(invoice, normal_style)
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

        vat_line_d = f"<br/>{vat_status_text}" if vat_status_text else ""
        company_info_detailed = f"""
        <b>{company_name}</b><br/>
        {company_address}<br/>
        {company_phone}<br/>
        {company_email}<br/>
        IDE/UID: {company_uid}{vat_line_d}
        """
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
                fontSize=8,
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

        # === BANDEAU RAPPEL (si mode rappel) ===
        display_reminder_level = reminder_ctx.get("display_reminder_level")
        if display_reminder_level:
            bandeau = Table(
                [
                    [
                        Paragraph(
                            f"<b>{display_reminder_level}</b>",
                            ParagraphStyle(
                                "BandeauD",
                                parent=styles["Normal"],
                                fontSize=14,
                                fontName=font_name_bold,
                                alignment=TA_CENTER,
                                textColor=colors.HexColor("#8B0000"),
                            ),
                        )
                    ]
                ],
                colWidths=[17 * cm],
            )
            bandeau.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFE5E5")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CC0000")),
                    ]
                )
            )
            story.append(bandeau)
            story.append(Spacer(1, 12))

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

        invoice_info_detailed = f"""
        <b>Numéro de facture :</b> {invoice.invoice_number}<br/>
        <b>Date d'émission :</b> {invoice.issued_at.strftime("%d.%m.%Y")}<br/>
        <b>{echeance_label}</b> {invoice.due_date.strftime("%d.%m.%Y")}<br/>
        <b>Période de facturation :</b> {period_label_d}<br/>
        <b>Statut :</b> {status_value}
        """
        story.append(Paragraph(invoice_info_detailed, normal_style))
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
        s2_table, consolidated_lines = _build_s2_table(
            invoice,
            font_name,
            font_name_bold,
            s2_main_style,
            include_non_ride=True,
            available_width_pt=doc.width,
        )

        story.append(Paragraph("<b>DÉTAIL DES TRANSPORTS</b>", normal_style))
        story.append(Spacer(1, 6))
        story.append(s2_table)
        story.append(Spacer(1, 2))

        # === TOTAL DÉTAILLÉ ===
        total_separator = Table([[""]], colWidths=[17 * cm])
        total_separator.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (0, 0), 1, colors.black),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(total_separator)
        story.append(Spacer(1, 8))

        if consolidated_lines:
            note_para = Paragraph(
                '<font size="7" color="grey">[A/R] = transport aller-retour</font>',
                detail_style,
            )
            story.append(note_para)
            story.append(Spacer(1, 8))

        # ✅ Utiliser toujours le format unifié pour les totaux (même libellé "TOTAL À FACTURER")
        # ✅ Si mode rappel, ajouter la ligne de frais de rappel
        if reminder_ctx.get("is_reminder"):
            story.append(Spacer(1, 16))  # Respiration avant totaux (rappel)
        total_table = _build_totals_table(
            invoice,
            True,
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
            story.append(Paragraph(invoice.notes, detail_style))
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

        # ✅ Pied de page : rappel dédié ou legal_footer / modalités standard
        if display_reminder_level:
            footer_message = (
                "Sauf erreur de notre part, cette facture est restée impayée à ce jour. "
                "Nous vous remercions de bien vouloir procéder à son règlement dans les plus brefs délais. "
                "Des frais de rappel ont été ajoutés conformément à nos conditions générales."
            )
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
            payment_info = ""
            if iban_value:
                payment_info = f"<br/><br/><b>Paiement par virement bancaire :</b><br/>IBAN : {iban_value}"
            else:
                app_logger.warning(
                    "PDF (detailed): IBAN non affiché (absent ou illisible, ex. erreur déchiffrement)."
                )
            footer_message = (
                f"<b>Modalités de paiement</b><br/>"
                f"En votre aimable règlement net sous {payment_terms_days} "
                f"{jours_text} avec nos remerciements anticipés.<br/>"
                f"En cas de retard de paiement, des frais de rappel d'un montant "
                f"de CHF {overdue_fee:.2f} vous seront facturés, "
                f"conformément à nos conditions générales."
                f"{payment_info}"
            )

        # Pied de page légal : dessiné en zone fixe (marge inférieure)
        mention = None
        if display_reminder_level:
            mention = f"Document généré automatiquement – facture initiale n° {invoice.invoice_number} inchangée."
        footer_cb_det = _make_legal_footer_page_callback(
            footer_message, mention, centered_style
        )

        def _on_first_page_det(canvas: Any, doc: Any) -> None:
            footer_cb_det(canvas, doc)
            _on_first_page_debug_envelope(canvas, doc)

        # === QR-BILL ===
        story.append(PageBreak())
        story.append(Spacer(1, QR_BILL_SPACER_PT))

        try:
            qr_bill_service = self.qrbill_service
            qr_bill_svg_content = qr_bill_service.generate_qr_bill_svg(invoice)
            if qr_bill_svg_content:
                drawing = _svg_content_to_drawing(qr_bill_svg_content)
                if drawing:
                    story.append(_make_qr_bill_table(drawing))
        except Exception as e:
            app_logger.warning("Impossible de générer le QR-Bill: %s", e)

        doc.build(story, onFirstPage=_on_first_page_det)
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
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import (
            TTFont,
        )
        from reportlab.platypus import (
            Paragraph,
            Spacer,
        )

        # ✅ Enregistrer une police TrueType pour supporter l'Unicode (caractères accentués)
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
            font_name = "DejaVuSans"
            font_name_bold = "DejaVuSans-Bold"
        except Exception:
            # Fallback sur Helvetica si DejaVu n'est pas disponible
            font_name = "Helvetica"
            font_name_bold = "Helvetica-Bold"

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
        left_section.append(Paragraph(billing_settings.iban, value_style))
        company = invoice.company
        left_section.append(
            Paragraph(company.name or "[Nom non configuré]", normal_style)
        )
        # Utiliser l'adresse de domiciliation
        street = (
            company.domicile_address_line1
            or company.address
            or "[Adresse non configurée]"
        )
        left_section.append(Paragraph(street, normal_style))
        postal_city = (
            f"{company.domicile_zip or ''} {company.domicile_city or ''}".strip()
            or "[Code postal/ville non configuré]"
        )
        left_section.append(Paragraph(postal_city, normal_style))
        left_section.append(Spacer(1, 8))

        # Informations débiteur
        left_section.append(Paragraph("Zahlbar durch", label_style))
        left_section.append(
            Paragraph(
                (
                    f"{invoice.client.user.first_name or ''} "
                    f"{invoice.client.user.last_name or ''}"
                ),
                normal_style,
            )
        )
        left_section.append(
            Paragraph(
                invoice.client.domicile_address or "Adresse non renseignée",
                normal_style,
            )
        )
        left_section.append(
            Paragraph(
                (
                    f"{invoice.client.domicile_zip or ''} "
                    f"{invoice.client.domicile_city or ''}"
                ),
                normal_style,
            )
        )
        left_section.append(Spacer(1, 8))

        # Référence
        left_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_section.append(Paragraph(qr_ref, value_style))
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
        right_section.append(Paragraph(billing_settings.iban, value_style))
        right_section.append(
            Paragraph(company.name or "[Nom non configuré]", normal_style)
        )
        right_section.append(Paragraph(street, normal_style))
        right_section.append(Paragraph(postal_city, normal_style))
        right_section.append(Spacer(1, 8))

        # QR Code
        right_section.append(qr_image)
        right_section.append(Spacer(1, 8))

        # Informations débiteur
        right_section.append(Paragraph("Zahlbar durch", label_style))
        right_section.append(
            Paragraph(
                (
                    f"{invoice.client.user.first_name or ''} "
                    f"{invoice.client.user.last_name or ''}"
                ),
                normal_style,
            )
        )
        right_section.append(
            Paragraph(
                invoice.client.domicile_address or "Adresse non renseignée",
                normal_style,
            )
        )
        right_section.append(
            Paragraph(
                (
                    f"{invoice.client.domicile_zip or ''} "
                    f"{invoice.client.domicile_city or ''}"
                ),
                normal_style,
            )
        )
        right_section.append(Spacer(1, 8))

        # Référence
        right_section.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_section.append(Paragraph(qr_ref, value_style))
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
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import (
            TTFont,
        )
        from reportlab.platypus import (
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        # ✅ Enregistrer une police TrueType pour supporter l'Unicode (caractères accentués)
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
            font_name = "DejaVuSans"
            font_name_bold = "DejaVuSans-Bold"
        except Exception:
            # Fallback sur Helvetica si DejaVu n'est pas disponible
            font_name = "Helvetica"
            font_name_bold = "Helvetica-Bold"

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
        left_content.append(Paragraph(billing_settings.iban, value_style))
        company = invoice.company
        left_content.append(
            Paragraph(company.name or "[Nom non configuré]", normal_text_style)
        )
        street = (
            company.domicile_address_line1
            or company.address
            or "[Adresse non configurée]"
        )
        left_content.append(Paragraph(street, normal_text_style))
        postal_city = (
            f"{company.domicile_zip or ''} {company.domicile_city or ''}".strip()
            or "[Code postal/ville non configuré]"
        )
        left_content.append(Paragraph(postal_city, normal_text_style))
        left_content.append(Spacer(1, 6))

        # Zahlbar durch
        left_content.append(Paragraph("Zahlbar durch", label_style))
        left_content.append(
            Paragraph(
                (
                    f"{invoice.client.user.first_name or ''} "
                    f"{invoice.client.user.last_name or ''}"
                ),
                normal_text_style,
            )
        )
        left_content.append(
            Paragraph(
                invoice.client.domicile_address or "Adresse non renseignée",
                normal_text_style,
            )
        )
        left_content.append(
            Paragraph(
                (
                    f"{invoice.client.domicile_zip or ''} "
                    f"{invoice.client.domicile_city or ''}"
                ),
                normal_text_style,
            )
        )
        left_content.append(Spacer(1, 6))

        # Referenz
        left_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        left_content.append(Paragraph(qr_ref, value_style))
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
        right_content.append(Paragraph(billing_settings.iban, value_style))
        right_content.append(
            Paragraph(company.name or "[Nom non configuré]", normal_text_style)
        )
        right_content.append(Paragraph(street, normal_text_style))
        right_content.append(Paragraph(postal_city, normal_text_style))
        right_content.append(Spacer(1, 6))

        # QR Code
        right_content.append(qr_image)
        right_content.append(Spacer(1, 6))

        # Zahlbar durch
        right_content.append(Paragraph("Zahlbar durch", label_style))
        right_content.append(
            Paragraph(
                (
                    f"{invoice.client.user.first_name or ''} "
                    f"{invoice.client.user.last_name or ''}"
                ),
                normal_text_style,
            )
        )
        right_content.append(
            Paragraph(
                invoice.client.domicile_address or "Adresse non renseignée",
                normal_text_style,
            )
        )
        right_content.append(
            Paragraph(
                (
                    f"{invoice.client.domicile_zip or ''} "
                    f"{invoice.client.domicile_city or ''}"
                ),
                normal_text_style,
            )
        )
        right_content.append(Spacer(1, 6))

        # Referenz
        right_content.append(Paragraph("Referenz", label_style))
        qr_ref = self.qrbill_service.generate_qr_reference(invoice) or ""
        right_content.append(Paragraph(qr_ref, value_style))
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

        # ✅ Afficher le montant consolidé (rappel consolidé)
        if reminder and reminder.total_due > 0:
            # Mode rappel consolidé : afficher principal + frais = total
            invoice_info.extend(
                [
                    ["Montant initial:", f"CHF {reminder.principal_amount:.2f}"],
                    [
                        f"Frais de rappel N°{level}:",
                        f"CHF {reminder.reminder_fee_amount:.2f}",
                    ],
                    ["", ""],  # Ligne vide
                    ["Total à payer:", f"CHF {reminder.total_due:.2f}"],
                ]
            )
        elif invoice.reminder_fee_amount and invoice.reminder_fee_amount > 0:
            # Mode legacy (rétrocompatibilité) : utiliser les valeurs de la facture
            initial_amount = invoice.total_amount - invoice.reminder_fee_amount
            invoice_info.extend(
                [
                    ["Montant facture initiale:", f"CHF {initial_amount:.2f}"],
                    [
                        f"Frais de rappel N°{level}:",
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

        story.append(Paragraph(f"Cher/Chère {client_name},", styles["Normal"]))
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

        story.append(Paragraph(message, styles["Normal"]))
        story.append(Spacer(1, 20))

        # Informations bancaires
        if billing_settings and billing_settings.iban:
            banking_info = (
                f"Paiement par virement bancaire : IBAN : {billing_settings.iban}"
            )
            story.append(Paragraph(banking_info, styles["Normal"]))

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
