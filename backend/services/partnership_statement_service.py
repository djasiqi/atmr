# services/partnership_statement_service.py
"""Service pour générer les décomptes de partenariats (documents comptables)."""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from flask import current_app  # pyright: ignore[reportMissingImports]
from reportlab.lib import colors  # pyright: ignore[reportMissingModuleSource]
from reportlab.lib.pagesizes import A4  # pyright: ignore[reportMissingModuleSource]
from reportlab.lib.styles import (  # pyright: ignore[reportMissingModuleSource]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import cm  # pyright: ignore[reportMissingModuleSource]
from reportlab.platypus import (  # pyright: ignore[reportMissingModuleSource]
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import or_

from ext import db
from models.booking import Booking
from models.booking_transfer import BookingTransfer
from models.company import Company
from models.enums import PartnershipStatus, TransferStatus
from models.partnership import Partnership
from services.pdf_service import PDFService

logger = logging.getLogger(__name__)


class PartnershipStatementService:
    """Service pour générer les décomptes de partenariats."""

    def __init__(self, pdf_service: PDFService | None = None):
        """Initialise le service."""
        super().__init__()
        self.pdf_service = pdf_service or PDFService()

    def generate_consolidated_statement(
        self,
        company_id: int,
        period_type: str,
        year: int | None = None,
        month: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Génère un décompte consolidé de tous les partenaires.

        Args:
            company_id: ID de l'entreprise
            period_type: Type de période ('annual', 'monthly', 'periodic')
            year: Année (pour annual/monthly)
            month: Mois (pour monthly)
            start_date: Date de début (pour periodic)
            end_date: Date de fin (pour periodic)

        Returns:
            URL du PDF généré

        Raises:
            ValueError: Si les paramètres sont invalides
        """
        # Calculer les dates de la période
        period_dates = self._calculate_period_dates(
            period_type, year, month, start_date, end_date
        )
        start = period_dates["start"]
        end = period_dates["end"]
        period_label = period_dates["label"]

        # Récupérer l'entreprise
        company = Company.query.get(company_id)
        if not company:
            raise ValueError(f"Entreprise {company_id} introuvable")

        # Récupérer tous les partenariats actifs
        partnerships = (
            db.session.query(Partnership)
            .filter(
                or_(
                    Partnership.owner_company_id == company_id,
                    Partnership.partner_company_id == company_id,
                ),
                Partnership.status == PartnershipStatus.ACCEPTED,
                Partnership.is_active.is_(True),
            )
            .all()
        )

        if not partnerships:
            raise ValueError("Aucun partenariat actif trouvé")

        # Récupérer tous les transferts de la période
        transfers = (
            db.session.query(BookingTransfer)
            .join(Partnership)
            .filter(
                or_(
                    Partnership.owner_company_id == company_id,
                    Partnership.partner_company_id == company_id,
                ),
                BookingTransfer.status == TransferStatus.COMPLETED,
                BookingTransfer.completed_at >= start,
                BookingTransfer.completed_at < end,
            )
            .order_by(BookingTransfer.completed_at)
            .all()
        )

        # Organiser les données par partenaire
        statement_data = self._organize_statement_data(
            company, partnerships, transfers, start, end, period_label
        )

        # Générer le PDF
        return self._generate_statement_pdf(statement_data, "consolidated")

    def generate_partnership_statement(
        self,
        partnership_id: int,
        company_id: int,
        period_type: str,
        year: int | None = None,
        month: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Génère un décompte pour un partenariat spécifique.

        Args:
            partnership_id: ID du partenariat
            company_id: ID de l'entreprise qui demande le décompte
            period_type: Type de période ('annual', 'monthly', 'periodic')
            year: Année (pour annual/monthly)
            month: Mois (pour monthly)
            start_date: Date de début (pour periodic)
            end_date: Date de fin (pour periodic)

        Returns:
            URL du PDF généré

        Raises:
            ValueError: Si les paramètres sont invalides
        """

        # Calculer les dates de la période
        try:
            period_dates = self._calculate_period_dates(
                period_type, year, month, start_date, end_date
            )
        except ValueError:
            raise
        start = period_dates["start"]
        end = period_dates["end"]
        period_label = period_dates["label"]

        # Récupérer le partenariat
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        # Vérifier que l'entreprise est liée au partenariat
        if company_id not in {
            partnership.owner_company_id,
            partnership.partner_company_id,
        }:
            raise ValueError(
                "Vous n'êtes pas autorisé à générer un décompte pour ce partenariat"
            )

        # Récupérer l'entreprise
        company = Company.query.get(company_id)
        if not company:
            raise ValueError(f"Entreprise {company_id} introuvable")

        # Déterminer le partenaire
        if partnership.owner_company_id == company_id:
            partner_company = partnership.partner_company
        else:
            partner_company = partnership.owner_company

        # Récupérer les transferts de la période
        transfers = (
            db.session.query(BookingTransfer)
            .filter(
                BookingTransfer.partnership_id == partnership_id,
                BookingTransfer.status == TransferStatus.COMPLETED,
                BookingTransfer.completed_at >= start,
                BookingTransfer.completed_at < end,
            )
            .order_by(BookingTransfer.completed_at)
            .all()
        )

        # Organiser les données
        statement_data = self._organize_single_partnership_data(
            company, partner_company, partnership, transfers, start, end, period_label
        )

        # Générer le PDF
        return self._generate_statement_pdf(statement_data, "single")

    def _calculate_period_dates(
        self,
        period_type: str,
        year: int | None,
        month: int | None,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> dict[str, Any]:
        """Calcule les dates de début et fin de période."""
        now = datetime.now(UTC)

        if period_type == "annual":
            if not year:
                year = now.year
            start = datetime(year, 1, 1, tzinfo=UTC)
            end = datetime(year + 1, 1, 1, tzinfo=UTC)
            label = f"Année {year}"
        elif period_type == "monthly":
            DECEMBER = 12
            MIN_MONTH = 1
            MAX_MONTH = 12
            if not year:
                year = now.year
            if month is None:
                month = now.month
            if month < MIN_MONTH or month > MAX_MONTH:
                raise ValueError(
                    f"Le mois doit être entre {MIN_MONTH} et {MAX_MONTH}, reçu: {month}"
                )
            start = datetime(year, month, 1, tzinfo=UTC)
            if month == DECEMBER:
                end = datetime(year + 1, 1, 1, tzinfo=UTC)
            else:
                end = datetime(year, month + 1, 1, tzinfo=UTC)
            month_names = [
                "",
                "Janvier",
                "Février",
                "Mars",
                "Avril",
                "Mai",
                "Juin",
                "Juillet",
                "Août",
                "Septembre",
                "Octobre",
                "Novembre",
                "Décembre",
            ]
            label = f"{month_names[month]} {year}"
        elif period_type == "periodic":
            if not start_date or not end_date:
                raise ValueError(
                    "Les dates de début et fin sont requises pour un décompte périodique"
                )
            start = start_date
            end = end_date
            label = f"Du {start_date.strftime('%d.%m.%Y')} au {end_date.strftime('%d.%m.%Y')}"
        else:
            raise ValueError(f"Type de période invalide: {period_type}")

        return {"start": start, "end": end, "label": label}

    def _organize_statement_data(
        self,
        company: Company,
        partnerships: list[Partnership],
        transfers: list[BookingTransfer],
        start: datetime,
        end: datetime,
        period_label: str,
    ) -> dict[str, Any]:
        """Organise les données pour un décompte consolidé."""
        # Grouper les transferts par partenariat
        transfers_by_partnership: dict[int, list[BookingTransfer]] = {}
        for transfer in transfers:
            partnership_id = transfer.partnership_id
            if partnership_id not in transfers_by_partnership:
                transfers_by_partnership[partnership_id] = []
            transfers_by_partnership[partnership_id].append(transfer)

        # Calculer les totaux par partenariat
        partnership_summaries = []
        total_courses = 0
        total_client_price = Decimal("0")
        total_partner_cost = Decimal("0")

        for partnership in partnerships:
            partnership_transfers = transfers_by_partnership.get(partnership.id, [])
            if not partnership_transfers:
                continue

            # Déterminer le partenaire
            if partnership.owner_company_id == company.id:
                partner_company = partnership.partner_company
            else:
                partner_company = partnership.owner_company

            # Calculer les totaux pour ce partenariat
            partnership_client_price = sum(
                t.client_price for t in partnership_transfers
            )
            partnership_partner_cost = sum(
                t.partner_cost or Decimal("0") for t in partnership_transfers
            )

            partnership_summaries.append(
                {
                    "partnership": partnership,
                    "partner_company": partner_company,
                    "transfers": partnership_transfers,
                    "count": len(partnership_transfers),
                    "client_price": partnership_client_price,
                    "partner_cost": partnership_partner_cost,
                    "balance": partnership_partner_cost - partnership_client_price,
                }
            )

            total_courses += len(partnership_transfers)
            total_client_price += partnership_client_price
            total_partner_cost += partnership_partner_cost

        return {
            "company": company,
            "type": "consolidated",
            "period_label": period_label,
            "start_date": start,
            "end_date": end,
            "partnership_summaries": partnership_summaries,
            "total_courses": total_courses,
            "total_client_price": total_client_price,
            "total_partner_cost": total_partner_cost,
            "net_balance": total_partner_cost - total_client_price,
        }

    def _organize_single_partnership_data(
        self,
        company: Company,
        partner_company: Company,
        partnership: Partnership,
        transfers: list[BookingTransfer],
        start: datetime,
        end: datetime,
        period_label: str,
    ) -> dict[str, Any]:
        """Organise les données pour un décompte d'un seul partenariat."""
        total_client_price = sum(t.client_price for t in transfers)
        total_partner_cost = sum(t.partner_cost or Decimal("0") for t in transfers)

        return {
            "company": company,
            "partner_company": partner_company,
            "partnership": partnership,
            "type": "single",
            "period_label": period_label,
            "start_date": start,
            "end_date": end,
            "transfers": transfers,
            "total_courses": len(transfers),
            "total_client_price": total_client_price,
            "total_partner_cost": total_partner_cost,
            "net_balance": total_partner_cost - total_client_price,
        }

    def _generate_statement_pdf(
        self, statement_data: dict[str, Any], statement_type: str
    ) -> str:
        """Génère le PDF du décompte."""
        company = statement_data["company"]
        period_label = statement_data["period_label"]

        # Nom du fichier
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        if statement_type == "consolidated":
            filename = f"decompte_consolide_{company.id}_{timestamp}.pdf"
        else:
            partnership_id = statement_data["partnership"].id
            filename = f"decompte_partenaire_{partnership_id}_{timestamp}.pdf"

        # Chemin du fichier
        statements_dir = (
            Path(current_app.config.get("UPLOADS_DIR", "uploads")) / "statements"
        )
        statements_dir.mkdir(parents=True, exist_ok=True)
        filepath = statements_dir / filename

        # Créer le document PDF
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#00695C"),
            spaceAfter=30,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#00695C"),
            spaceAfter=12,
        )

        # Contenu du document
        story = []

        # En-tête
        story.append(Paragraph("DÉCOMPTE DE PARTENARIAT", title_style))
        story.append(Spacer(1, 0.5 * cm))

        # Informations de l'entreprise
        story.append(Paragraph(f"<b>Entreprise:</b> {company.name}", styles["Normal"]))
        # Construire l'adresse à partir des champs disponibles
        address_parts = []
        if company.address:
            address_parts.append(company.address)
        elif company.domicile_address_line1:
            address_parts.append(company.domicile_address_line1)
            if company.domicile_address_line2:
                address_parts.append(company.domicile_address_line2)
            if company.domicile_zip and company.domicile_city:
                address_parts.append(f"{company.domicile_zip} {company.domicile_city}")
            elif company.domicile_city:
                address_parts.append(company.domicile_city)
        if address_parts:
            story.append(
                Paragraph(
                    f"<b>Adresse:</b> {', '.join(address_parts)}", styles["Normal"]
                )
            )
        story.append(Spacer(1, 0.3 * cm))

        # Période
        story.append(Paragraph(f"<b>Période:</b> {period_label}", styles["Normal"]))
        story.append(
            Paragraph(
                f"<b>Date de génération:</b> {datetime.now(UTC).strftime('%d.%m.%Y %H:%M')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.5 * cm))

        if statement_type == "consolidated":
            # Décompte consolidé
            story.append(
                Paragraph("DÉCOMPTE CONSOLIDÉ - TOUS PARTENAIRES", heading_style)
            )

            # Résumé global
            summary_data = [
                ["Nombre total de courses", f"{statement_data['total_courses']}"],
                [
                    "Total prix client (HT)",
                    f"{float(statement_data['total_client_price']):.2f} CHF",
                ],
                [
                    "Total coût partenaires (HT)",
                    f"{float(statement_data['total_partner_cost']):.2f} CHF",
                ],
                [
                    "Solde net",
                    f"{float(statement_data['net_balance']):.2f} CHF",
                ],
            ]

            summary_table = Table(summary_data, colWidths=[10 * cm, 5 * cm])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2F1")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00695C")),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ]
                )
            )
            story.append(summary_table)
            story.append(Spacer(1, 0.5 * cm))

            # Détail par partenaire
            story.append(Paragraph("DÉTAIL PAR PARTENAIRE", heading_style))

            for summary in statement_data["partnership_summaries"]:
                partner_name = (
                    summary["partner_company"].name
                    if summary["partner_company"]
                    else "Partenaire inconnu"
                )
                story.append(
                    Paragraph(f"<b>Partenaire: {partner_name}</b>", styles["Normal"])
                )

                partner_data = [
                    ["Nombre de courses", f"{summary['count']}"],
                    [
                        "Total prix client",
                        f"{float(summary['client_price']):.2f} CHF",
                    ],
                    [
                        "Total coût partenaire",
                        f"{float(summary['partner_cost']):.2f} CHF",
                    ],
                    [
                        "Solde",
                        f"{float(summary['balance']):.2f} CHF",
                    ],
                ]

                partner_table = Table(partner_data, colWidths=[10 * cm, 5 * cm])
                partner_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ]
                    )
                )
                story.append(partner_table)
                story.append(Spacer(1, 0.3 * cm))

        else:
            # Décompte par partenaire
            partner_company = statement_data["partner_company"]
            story.append(
                Paragraph(
                    f"DÉCOMPTE PARTENAIRE - {partner_company.name if partner_company else 'Inconnu'}",
                    heading_style,
                )
            )

            # Résumé
            summary_data = [
                ["Nombre de courses", f"{statement_data['total_courses']}"],
                [
                    "Total prix client (HT)",
                    f"{float(statement_data['total_client_price']):.2f} CHF",
                ],
                [
                    "Total coût partenaire (HT)",
                    f"{float(statement_data['total_partner_cost']):.2f} CHF",
                ],
                [
                    "Solde net",
                    f"{float(statement_data['net_balance']):.2f} CHF",
                ],
            ]

            summary_table = Table(summary_data, colWidths=[10 * cm, 5 * cm])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2F1")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00695C")),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ]
                )
            )
            story.append(summary_table)
            story.append(Spacer(1, 0.5 * cm))

            # Liste détaillée des transferts
            if statement_data["transfers"]:
                story.append(Paragraph("DÉTAIL DES COURSES", heading_style))

                # En-têtes du tableau
                transfer_headers = [
                    "Date",
                    "Client",
                    "Trajet",
                    "Prix client",
                    "Coût partenaire",
                ]

                transfer_rows = [transfer_headers]

                # Charger les bookings pour avoir les détails
                transfer_ids = [t.booking_id for t in statement_data["transfers"]]
                bookings = (
                    db.session.query(Booking).filter(Booking.id.in_(transfer_ids)).all()
                )
                bookings_dict = {b.id: b for b in bookings}

                for transfer in statement_data["transfers"]:
                    booking = bookings_dict.get(transfer.booking_id)
                    client_name = booking.customer_full_name if booking else "N/A"
                    pickup = booking.pickup_location if booking else "N/A"
                    dropoff = booking.dropoff_location if booking else "N/A"
                    trajet = f"{pickup} → {dropoff}"

                    date_str = (
                        transfer.completed_at.strftime("%d.%m.%Y %H:%M")
                        if transfer.completed_at
                        else "N/A"
                    )

                    transfer_rows.append(
                        [
                            date_str,
                            client_name[:30],  # Limiter la longueur
                            trajet[:40],
                            f"{float(transfer.client_price):.2f} CHF",
                            f"{float(transfer.partner_cost or 0):.2f} CHF",
                        ]
                    )

                transfer_table = Table(
                    transfer_rows,
                    colWidths=[3 * cm, 4 * cm, 5 * cm, 2.5 * cm, 2.5 * cm],
                )
                transfer_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2F1")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00695C")),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#F5F5F5")],
                            ),
                        ]
                    )
                )
                story.append(transfer_table)

        # Notes
        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                (
                    "<i>Ce document est un décompte professionnel à des fins comptables. "
                    "Il ne constitue pas une facture.</i>"
                ),
                styles["Normal"],
            )
        )

        # Générer le PDF
        doc.build(story)

        # Retourner l'URL
        pdf_base_url = current_app.config.get("PDF_BASE_URL", "http://localhost:5000")
        uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        pdf_url = f"{pdf_base_url}{uploads_base}/statements/{filename}"

        logger.info("Décompte PDF généré: %s", pdf_url)
        return pdf_url
