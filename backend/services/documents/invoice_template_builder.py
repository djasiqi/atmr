"""Invoice Template Builder - Système de templating modulaire pour les factures PDF.

Ce module fournit une architecture propre pour générer des factures PDF avec
différents layouts (Standard/Minimal/Detailed) en utilisant CompanyBillingProfile
comme source unique de données.
"""

import html
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from services.billing import BillingProfileService

logger = logging.getLogger(__name__)


@dataclass
class InvoiceData:
    """Conteneur de données pour une facture - Single Source of Truth."""

    # Facture
    invoice_number: str
    issue_date: datetime
    due_date: datetime
    period: str
    total_amount: Decimal
    balance_due: Decimal

    # Émetteur (depuis CompanyBillingProfile)
    emitter_name: str
    emitter_street: str
    emitter_postal_code: str
    emitter_city: str
    emitter_country: str
    emitter_uid: str
    emitter_email: str
    emitter_phone: str
    emitter_iban: str

    # Client
    client_name: str
    client_address: str

    # Lignes de facture
    lines: list[dict[str, Any]]

    # Options (avec valeurs par défaut)
    is_reminder: bool = False
    reminder_level: int | None = None
    vat_applicable: bool = False
    vat_number: str | None = None
    vat_rate: Decimal | None = None
    payment_reference: str | None = None
    show_patient_column: bool = False


class InvoiceTemplateBuilder:
    """Builder pour générer des factures PDF avec différents templates.

    Cette classe centralise la logique de génération de factures et utilise
    CompanyBillingProfile comme source unique de données.

    Templates supportés:
    - Standard: Layout riche avec toutes les informations
    - Minimal: Layout compact pour économiser le papier
    - Detailed: Layout exhaustif avec détails maximaux
    """

    def __init__(self):  # type: ignore[reportMissingSuperCall]
        """Initialise le builder."""
        self.profile_service = BillingProfileService()

    def extract_invoice_data(self, invoice) -> InvoiceData | None:
        """Extrait toutes les données d'une facture depuis le profil.

        Args:
            invoice: Instance d'Invoice (SQLAlchemy model)

        Returns:
            InvoiceData: Données structurées ou None si profil manquant
        """
        try:
            # Récupérer le profil de facturation
            profile = self.profile_service.get_by_company_id(invoice.company_id)

            if not profile:
                logger.warning(
                    "Pas de profil de facturation pour company_id=%s",
                    invoice.company_id,
                )
                return None

            # Informations de l'émetteur depuis le profil
            emitter_street = f"{profile.street_name} {profile.building_number}".strip()

            # Période de facturation
            period_str = f"{invoice.period_month:02d}.{invoice.period_year}"

            # Destinataire ("Facturé à") - priorité: BillingParty, puis bill_to_client, sinon client
            billed_to_name, billed_to_address = self._resolve_billed_to(invoice)

            # Lignes de facture
            lines = self._extract_invoice_lines(invoice)

            # Référence de paiement (si configurée)
            payment_reference = None
            if profile.payment_reference_mode == "SCOR":
                from services.billing import generate_scor_reference

                payment_reference = generate_scor_reference(
                    invoice.invoice_number, company_id=invoice.company_id
                )

            # Détection de rappel
            is_reminder = (
                len(invoice.reminders) > 0
                if hasattr(invoice, "reminders") and invoice.reminders
                else False
            )
            reminder_level = None
            if is_reminder:
                reminder_level = len(invoice.reminders)

            # ✅ Affichage colonne "Patient" (obligatoire pour S2 multi-patients)
            strategy_value: str | None = None
            try:
                bs = getattr(invoice, "billing_strategy", None)
                if bs is not None:
                    strategy_value = bs.value if hasattr(bs, "value") else str(bs)
            except Exception:
                strategy_value = None

            meta_strategy: str | None = None
            try:
                meta = getattr(invoice, "meta", None)
                if isinstance(meta, dict):
                    ms = meta.get("billing_strategy")
                    meta_strategy = str(ms) if ms is not None else None
            except Exception:
                meta_strategy = None

            is_s2 = (strategy_value == "s2_clinic_monthly") or (
                meta_strategy == "s2_clinic_monthly"
            )
            is_third_party = bool(
                getattr(invoice, "bill_to_client_id", None)
                and getattr(invoice, "client_id", None)
                and invoice.bill_to_client_id != invoice.client_id
            )
            has_billing_party = bool(getattr(invoice, "billing_party_id", None))
            show_patient_column = bool(is_s2 or is_third_party or has_billing_party)

            return InvoiceData(
                # Facture
                invoice_number=invoice.invoice_number,
                issue_date=invoice.issued_at,
                due_date=invoice.due_date,
                period=period_str,
                total_amount=invoice.total_amount,
                balance_due=invoice.balance_due,
                is_reminder=is_reminder,
                reminder_level=reminder_level,
                # Émetteur
                emitter_name=profile.legal_name,
                emitter_street=emitter_street,
                emitter_postal_code=profile.postal_code,
                emitter_city=profile.city,
                emitter_country=profile.country_code,
                emitter_uid=profile.uid_ide,
                emitter_email=profile.billing_email,
                emitter_phone=profile.billing_phone,
                emitter_iban=profile.iban or profile.qr_iban or "[IBAN non configuré]",
                # TVA
                vat_applicable=profile.vat_registered,
                vat_number=profile.vat_number,
                vat_rate=profile.vat_rate,
                # Destinataire
                client_name=billed_to_name,
                client_address=billed_to_address,
                # Lignes
                lines=lines,
                # Référence
                payment_reference=payment_reference,
                show_patient_column=show_patient_column,
            )

        except Exception as e:
            logger.error("Erreur extraction données facture: %s", e)
            return None

    def _name_with_uppercase_last_name(self, name: str) -> str:
        """Met le nom de famille (dernier mot) en majuscules pour le bloc « Facturé à »."""
        if not name or not str(name).strip():
            return name
        parts = name.strip().split()
        if not parts:
            return name
        parts[-1] = parts[-1].upper()
        return " ".join(parts)

    def _format_client_name(self, client) -> str:
        """Formate le nom du client.

        Args:
            client: Instance de Client

        Returns:
            str: Nom formaté (nom de famille en majuscules)
        """
        if hasattr(client, "user") and client.user:
            first_name = client.user.first_name or ""
            last_name = (client.user.last_name or "").upper()
            full_name = f"{first_name} {last_name}".strip()
            return self._name_with_uppercase_last_name(
                full_name or client.user.username or "Client"
            )
        return "Client"

    def _format_client_address(self, client) -> str:
        """Formate l'adresse du client.

        Args:
            client: Instance de Client

        Returns:
            str: Adresse formatée (avec <br/> pour retours à la ligne)
        """

        def _to_multiline(value: str) -> str:
            """Normalise une adresse texte en HTML multi-ligne."""
            if not (value or "").strip():
                return ""
            # Supporter les adresses déjà multi-lignes + format "a, b, c"
            return (
                (value or "")
                .strip()
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n", "<br/>")
                .replace(", ", "<br/>")
            )

        # ✅ Priorité 1: Coordonnées de facturation du client (cas curatelle/institution)
        # Ces champs existent dans le modèle Client: billing_address / contact_email / contact_phone.
        billing_address = ""
        try:
            # Utiliser la version déchiffrée si disponible
            billing_address = getattr(client, "billing_address_secure", None) or ""
        except Exception:
            billing_address = getattr(client, "billing_address", "") or ""

        billing_address_html = _to_multiline(billing_address)

        # Contact facturation (optionnel)
        contact_email = getattr(client, "contact_email", None) or ""
        contact_phone = ""
        try:
            contact_phone = getattr(client, "contact_phone_secure", None) or ""
        except Exception:
            contact_phone = getattr(client, "contact_phone", "") or ""

        contact_lines: list[str] = []
        if contact_email.strip():
            contact_lines.append(f"Email facturation : {contact_email.strip()}")
        if contact_phone.strip():
            contact_lines.append(f"Téléphone : {contact_phone.strip()}")

        # Si une adresse de facturation est renseignée, c'est elle qu'on affiche.
        if billing_address_html:
            if contact_lines:
                return f"{billing_address_html}<br/>{'<br/>'.join(contact_lines)}"
            return billing_address_html

        # Priorité 2: Adresse du domicile (fallback)
        if hasattr(client, "domicile_address") and client.domicile_address:
            address = _to_multiline(str(client.domicile_address))
            postal_code = (getattr(client, "domicile_zip", "") or "").strip()
            city = (getattr(client, "domicile_city", "") or "").strip()

            if postal_code and city:
                return f"{address}<br/>{postal_code} {city}"
            return address

        # Priorité 2: Adresse de l'utilisateur
        if (
            hasattr(client, "user")
            and client.user
            and hasattr(client.user, "address")
            and client.user.address
        ):
            return _to_multiline(str(client.user.address))

        return "Adresse non renseignée"

    def _resolve_billed_to(self, invoice) -> tuple[str, str]:
        """Résout le destinataire de facture (bloc 'Facturé à')."""
        # 1) BillingParty (nouveau modèle unifié)
        try:
            from models.billing_party import ClientBillingParty
            from models.enums import BillingPartyType

            # Si le client n'a plus de lien avec ce tiers payeur (lien supprimé), facturer au domicile du client
            client_id = getattr(invoice, "client_id", None)
            bp_id = getattr(invoice, "billing_party_id", None)
            if client_id is not None and bp_id is not None:
                link = (
                    ClientBillingParty.query.filter_by(
                        client_id=client_id, billing_party_id=bp_id
                    ).first()
                )
                if link is None:
                    logger.info(
                        "[InvoiceTemplateBuilder] Lien client↔tiers payeur supprimé (invoice_id=%s). Facturé à = domicile du client.",
                        getattr(invoice, "id", None),
                    )
                    # Ne pas utiliser le tiers payeur : on passe au fallback client plus bas
                    raise ValueError("use_client_fallback")

            bp = getattr(invoice, "billing_party", None)
            if bp is not None:
                bp_name = (getattr(bp, "display_name", None) or "Payeur").strip()
                addr = (getattr(bp, "billing_address", None) or "").strip()
                if addr:
                    addr_html = (
                        addr.replace("\r\n", "\n")
                        .replace("\r", "\n")
                        .replace("\n", "<br/>")
                        .replace(", ", "<br/>")
                    )
                else:
                    addr_html = "Adresse non renseignée"
                # Contact (optionnel)
                contact_lines: list[str] = []
                email = (getattr(bp, "contact_email", None) or "").strip()
                phone = (getattr(bp, "contact_phone", None) or "").strip()
                ext = (getattr(bp, "external_ref", None) or "").strip()
                if email:
                    contact_lines.append(f"Email facturation : {email}")
                if phone:
                    contact_lines.append(f"Téléphone : {phone}")
                if ext:
                    contact_lines.append(f"Référence : {ext}")
                if contact_lines:
                    addr_html = f"{addr_html}<br/>{'<br/>'.join(contact_lines)}"
                # Tiers payeur : plusieurs lignes = client, puis c/o tiers payeur, puis adresse
                name = bp_name
                if getattr(invoice, "client_id", None) and getattr(bp, "type", None) in (
                    BillingPartyType.FAMILY,
                    BillingPartyType.CURATORSHIP,
                    BillingPartyType.OPAD,
                    BillingPartyType.LAWYER,
                    BillingPartyType.INSURANCE,
                    BillingPartyType.OTHER,
                ):
                    client = getattr(invoice, "client", None)
                    if client is not None:
                        client_name = self._format_client_name(client)
                        if client_name.strip():
                            name = f"{client_name}<br/>c/o {bp_name}"
                # Si le tiers payeur est SPC : ajouter le numéro SPC après l'adresse (2 sauts de ligne)
                if (bp_name or "").upper().find("SPC") >= 0:
                    from models.billing_party import ClientBillingParty

                    client_id = getattr(invoice, "client_id", None)
                    bp_id = getattr(bp, "id", None) or getattr(invoice, "billing_party_id", None)
                    if client_id is not None and bp_id is not None:
                        link = (
                            ClientBillingParty.query.filter_by(
                                client_id=client_id, billing_party_id=bp_id
                            )
                            .first()
                        )
                        if (
                            link
                            and getattr(link, "client_reference", None)
                            and (link.client_reference or "").strip()
                        ):
                            # 2 lignes vides puis numéro SPC (3 <br/> = 2 lignes vides)
                            addr_html = (
                                f"{addr_html}<br/><br/><br/>No. SPC : {(link.client_reference or '').strip()}"
                            )
                return (name, addr_html)
        except Exception:
            pass

        # Si un billing_party_id est défini mais qu'on n'a pas réussi à charger le BP,
        # on log pour observabilité (fallback PDF).
        try:
            if getattr(invoice, "billing_party_id", None):
                logger.warning(
                    "[InvoiceTemplateBuilder] billing_party_id=%s défini mais BillingParty non résolu (invoice_id=%s). Fallback sur legacy/client.",
                    getattr(invoice, "billing_party_id", None),
                    getattr(invoice, "id", None),
                )
        except Exception:
            pass

        # 2) Legacy: bill_to_client_id (Client institution)
        import contextlib

        with contextlib.suppress(Exception):
            bill_to = getattr(invoice, "bill_to_client", None)
            if (
                getattr(invoice, "bill_to_client_id", None)
                and getattr(invoice, "client_id", None)
                and invoice.bill_to_client_id != invoice.client_id
                and bill_to is not None
            ):
                logger.info(
                    "[InvoiceTemplateBuilder] Fallback legacy bill_to_client_id utilisé (invoice_id=%s, bill_to_client_id=%s).",
                    getattr(invoice, "id", None),
                    getattr(invoice, "bill_to_client_id", None),
                )
                return self._format_client_name(bill_to), self._format_client_address(
                    bill_to
                )

        # 3) Fallback: client bénéficiaire (domicile + éventuellement établissement de résidence)
        client = getattr(invoice, "client", None)
        if client is None:
            return "Client", "Adresse non renseignée"
        with contextlib.suppress(Exception):
            logger.info(
                "[InvoiceTemplateBuilder] Fallback client bénéficiaire utilisé (invoice_id=%s, client_id=%s).",
                getattr(invoice, "id", None),
                getattr(invoice, "client_id", None),
            )
        name = self._format_client_name(client)
        residence_facility = (getattr(client, "residence_facility", None) or "").strip()
        if residence_facility:
            name = f"{name}<br/>{residence_facility}"
        return name, self._format_client_address(client)

    def _extract_invoice_lines(self, invoice) -> list[dict[str, Any]]:
        """Extrait les lignes de facture.

        Args:
            invoice: Instance d'Invoice

        Returns:
            list[dict]: Liste des lignes avec date, départ, arrivée, montant
        """
        lines = []

        for line in invoice.lines:
            # Filtrer les lignes de type RIDE seulement
            if line.type not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            ):
                continue

            booking = line.booking if hasattr(line, "booking") else None

            if not booking:
                continue

            # ✅ Patient (utile pour S2 multi-patients / facturation tierce)
            # Priorité: snapshot depuis line.meta (traçabilité juridique) > booking.client.user
            patient_name = "N/A"
            try:
                # ✅ S2: Utiliser le snapshot patient_name depuis line.meta si disponible
                if hasattr(line, "meta") and isinstance(line.meta, dict):
                    patient_name = (
                        line.meta.get("patient_name")
                        or getattr(booking, "customer_name", None)
                        or "N/A"
                    )
                else:
                    # Fallback si meta n'existe pas (rétro-compatibilité)
                    cli = getattr(booking, "client", None)
                    if cli is not None:
                        user = getattr(cli, "user", None)
                        if user is not None:
                            full = (
                                f"{(getattr(user, 'first_name', '') or '').strip()} "
                                f"{(getattr(user, 'last_name', '') or '').strip()}"
                            ).strip()
                            patient_name = full or (
                                getattr(user, "username", None) or "N/A"
                            )
                        else:
                            patient_name = (
                                getattr(cli, "institution_name", None) or "N/A"
                            )
                    else:
                        patient_name = (
                            getattr(booking, "customer_full_name", None)
                            or getattr(booking, "customer_name", None)
                            or "N/A"
                        )
            except Exception:
                patient_name = "N/A"

            adj_raw = getattr(line, "adjustment_note", None)
            adj_note = str(adj_raw).strip() if adj_raw is not None else ""

            row: dict[str, Any] = {
                "date": booking.pickup_datetime.strftime("%d/%m/%Y")
                if booking.pickup_datetime
                else "N/A",
                "patient": patient_name,
                "departure": booking.pickup_address or "N/A",
                "arrival": booking.dropoff_address or "N/A",
                "amount": float(line.line_total),
            }
            if adj_note:
                row["adjustment_note"] = adj_note
            lines.append(row)

        return lines

    def build_header_html(self, data: InvoiceData) -> str:
        """Génère le header HTML de la facture.

        Args:
            data: Données de la facture

        Returns:
            str: HTML du header
        """
        reminder_badge = ""
        if data.is_reminder:
            reminder_badge = f'<span style="color: #e53e3e; font-weight: bold;">RAPPEL N°{data.reminder_level}</span>'

        vat_info = ""
        if data.vat_applicable and data.vat_number:
            vat_info = f"<br/>N° TVA : {data.vat_number}"
        elif data.vat_applicable:
            vat_info = "<br/>TVA non assujettie"

        return f"""
        <div style="margin-bottom: 30px;">
            <div style="float: left; width: 50%;">
                <h2 style="margin: 0; color: #2d3748;">{data.emitter_name}</h2>
                <p style="margin: 5px 0; color: #4a5568; line-height: 1.6;">
                    {data.emitter_street}<br/>
                    {data.emitter_postal_code} {data.emitter_city}<br/>
                    {data.emitter_country}
                </p>
                <p style="margin: 10px 0; color: #4a5568; line-height: 1.6;">
                    Email : {data.emitter_email}<br/>
                    Téléphone : {data.emitter_phone}<br/>
                    IDE/UID : {data.emitter_uid}
                    {vat_info}
                </p>
            </div>
            <div style="float: right; width: 45%; text-align: right;">
                <h1 style="margin: 0; color: #2d3748;">FACTURE {reminder_badge}</h1>
                <p style="margin: 10px 0; color: #4a5568; line-height: 1.6;">
                    <strong>N° {data.invoice_number}</strong><br/>
                    Date : {data.issue_date.strftime("%d.%m.%Y")}<br/>
                    Échéance : {data.due_date.strftime("%d.%m.%Y")}<br/>
                    Période : {data.period}
                </p>
            </div>
            <div style="clear: both;"></div>
        </div>

        <div style="margin: 30px 0; padding: 15px; background-color: #f7fafc; border-left: 4px solid #4299e1;">
            <p style="margin: 0; font-weight: bold; color: #2d3748;">Facturé à :</p>
            <p style="margin: 5px 0; color: #4a5568; line-height: 1.6;">
                {data.client_name}<br/>
                {data.client_address}
            </p>
        </div>
        """

    def build_footer_html(self, data: InvoiceData) -> str:
        """Génère le footer HTML de la facture.

        Args:
            data: Données de la facture

        Returns:
            str: HTML du footer
        """
        payment_info = ""
        if data.payment_reference:
            payment_info = f"<br/><strong>Référence :</strong> {data.payment_reference}"

        return f"""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #e2e8f0;">
            <h3 style="color: #2d3748; margin-bottom: 10px;">Informations de paiement</h3>
            <p style="margin: 5px 0; color: #4a5568; line-height: 1.6;">
                <strong>IBAN :</strong> {data.emitter_iban}
                {payment_info}
            </p>
            <p style="margin: 15px 0; color: #718096; font-size: 0.9em;">
                En votre aimable règlement. Merci de votre confiance.
            </p>
        </div>
        """

    def build_lines_table_standard(self, data: InvoiceData) -> str:
        """Génère une table HTML rich pour les lignes de facture (template Standard).

        Args:
            data: Données de la facture

        Returns:
            str: HTML de la table des lignes
        """
        if not data.lines:
            return "<p style='color: #718096;'>Aucune course facturée</p>"

        patient_header = (
            '<th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #2d3748;">Patient</th>'
            if data.show_patient_column
            else ""
        )
        rows_html = ""
        for line in data.lines:
            patient_cell = (
                f'<td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["patient"]}</td>'
                if data.show_patient_column
                else ""
            )
            adj = line.get("adjustment_note")
            adj_html = ""
            if adj:
                adj_html = (
                    f'<br/><span style="font-size:0.8em;color:#718096;font-weight:400;">'
                    f"{html.escape(str(adj))}</span>"
                )
            rows_html += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["date"]}</td>
                {patient_cell}
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["departure"]}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["arrival"]}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: bold;">{line["amount"]:.2f} CHF{adj_html}</td>
            </tr>
            """

        total_ht = float(data.total_amount)
        tva_amount = 0.0
        if data.vat_applicable and data.vat_rate:
            tva_amount = total_ht * float(data.vat_rate) / 100
            total_ttc = total_ht + tva_amount
        else:
            total_ttc = total_ht

        return f"""
        <div style="margin: 30px 0;">
            <h3 style="color: #2d3748; margin-bottom: 15px;">📋 Détail des courses</h3>
            <table style="width: 100%; border-collapse: collapse; background-color: white;">
                <thead>
                    <tr style="background-color: #f7fafc;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #2d3748;">Date</th>
                        {patient_header}
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #2d3748;">Départ</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e2e8f0; color: #2d3748;">Arrivée</th>
                        <th style="padding: 12px; text-align: right; border-bottom: 2px solid #e2e8f0; color: #2d3748;">Montant</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div style="margin-top: 20px; padding: 20px; background-color: #4299e1; color: white; text-align: right;">
                <p style="margin: 5px 0; font-size: 1.1em;"><strong>Total HT :</strong> {total_ht:.2f} CHF</p>
                {f'<p style="margin: 5px 0;"><strong>TVA ({data.vat_rate}%) :</strong> {tva_amount:.2f} CHF</p>' if data.vat_applicable else ""}
                <p style="margin: 10px 0 0 0; font-size: 1.3em; font-weight: bold;">Total TTC : {total_ttc:.2f} CHF</p>
            </div>
        </div>
        """

    def build_lines_table_minimal(self, data: InvoiceData) -> str:
        """Génère une table HTML compacte pour les lignes de facture (template Minimal).

        Args:
            data: Données de la facture

        Returns:
            str: HTML de la table des lignes (version compacte)
        """
        if not data.lines:
            return "<p style='color: #718096;'>Aucune course</p>"

        rows_html = ""
        for line in data.lines:
            adj = line.get("adjustment_note")
            date_cell = line["date"]
            if adj:
                date_cell = (
                    f'{line["date"]}<br/>'
                    f'<span style="font-size:0.75em;color:#718096;">'
                    f"{html.escape(str(adj))}</span>"
                )
            rows_html += f"""
            <tr>
                <td style="padding: 6px; font-size: 0.85em;">{date_cell}</td>
                <td style="padding: 6px; font-size: 0.85em;">{line["amount"]:.2f} CHF</td>
            </tr>
            """

        return f"""
        <div style="margin: 15px 0;">
            <h4 style="color: #2d3748; margin-bottom: 10px;">Détail</h4>
            <table style="width: 100%; font-size: 0.9em;">
                {rows_html}
            </table>
            <div style="margin-top: 15px; padding: 10px; background-color: #edf2f7; text-align: right; font-weight: bold;">
                Total : {float(data.total_amount):.2f} CHF
            </div>
        </div>
        """

    def build_lines_table_detailed(self, data: InvoiceData) -> str:
        """Génère une table HTML exhaustive pour les lignes de facture (template Detailed).

        Args:
            data: Données de la facture

        Returns:
            str: HTML de la table des lignes (version exhaustive)
        """
        if not data.lines:
            return "<p style='color: #718096;'>Aucune course facturée pour cette période</p>"

        patient_header = (
            '<th style="padding: 14px; text-align: left; width: 180px;">Patient</th>'
            if data.show_patient_column
            else ""
        )
        colspan = "5" if data.show_patient_column else "4"

        rows_html = ""
        subtotal = 0.0
        for idx, line in enumerate(data.lines, 1):
            subtotal += line["amount"]
            patient_cell = (
                f'<td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["patient"]}</td>'
                if data.show_patient_column
                else ""
            )
            adj = line.get("adjustment_note")
            adj_html = ""
            if adj:
                adj_html = (
                    f'<br/><span style="font-size:0.8em;color:#718096;font-weight:400;">'
                    f"{html.escape(str(adj))}</span>"
                )
            rows_html += f"""
            <tr style="{"background-color: #f7fafc;" if idx % 2 == 0 else ""}">
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: center;">{idx}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{line["date"]}</td>
                {patient_cell}
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; color: #4a5568;">{line["departure"]}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; color: #4a5568;">{line["arrival"]}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: bold;">{line["amount"]:.2f} CHF{adj_html}</td>
            </tr>
            """

        tva_amount = 0.0
        if data.vat_applicable and data.vat_rate:
            tva_amount = float(data.total_amount) * float(data.vat_rate) / 100

        return f"""
        <div style="margin: 30px 0;">
            <h3 style="color: #2d3748; margin-bottom: 15px;">📋 Détail exhaustif des prestations</h3>
            <table style="width: 100%; border-collapse: collapse; background-color: white;">
                <thead>
                    <tr style="background-color: #2d3748; color: white;">
                        <th style="padding: 14px; text-align: center; width: 50px;">N°</th>
                        <th style="padding: 14px; text-align: left; width: 100px;">Date</th>
                        {patient_header}
                        <th style="padding: 14px; text-align: left;">Départ</th>
                        <th style="padding: 14px; text-align: left;">Arrivée</th>
                        <th style="padding: 14px; text-align: right; width: 120px;">Montant</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
                <tfoot>
                    <tr>
                        <td colspan="{colspan}" style="padding: 14px; text-align: right; border-top: 2px solid #2d3748; font-weight: bold;">Sous-total :</td>
                        <td style="padding: 14px; text-align: right; border-top: 2px solid #2d3748; font-weight: bold;">{subtotal:.2f} CHF</td>
                    </tr>
                    {f'<tr><td colspan="{colspan}" style="padding: 10px; text-align: right;">TVA ({data.vat_rate}%) :</td><td style="padding: 10px; text-align: right;">{tva_amount:.2f} CHF</td></tr>' if data.vat_applicable else ""}
                    <tr style="background-color: #4299e1; color: white;">
                        <td colspan="{colspan}" style="padding: 16px; text-align: right; font-size: 1.2em; font-weight: bold;">TOTAL À PAYER :</td>
                        <td style="padding: 16px; text-align: right; font-size: 1.2em; font-weight: bold;">{float(data.total_amount):.2f} CHF</td>
                    </tr>
                </tfoot>
            </table>
        </div>
        """


# Import nécessaire pour les types
from models import InvoiceLineType  # noqa: E402
