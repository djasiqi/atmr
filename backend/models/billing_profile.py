"""Modèle pour le profil de facturation centralisé.

Ce modèle centralise toutes les informations nécessaires pour la génération
de factures conformes aux standards suisses (QR-Bill).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class CompanyBillingProfile(db.Model):
    """Profil de facturation centralisé - Single Source of Truth.

    Ce modèle garantit la cohérence entre les données affichées sur la facture
    et celles encodées dans le QR-Bill. Toutes les adresses sont structurées
    selon les spécifications Swiss QR-Bill (ISO 20022).

    Attributes:
        company_id: ID de l'entreprise (relation 1-1)
        legal_name: Nom légal de l'entreprise (pour factures)
        brand_name: Nom commercial (optionnel)
        uid_ide: Numéro IDE/UID suisse (obligatoire)

        # Adresse structurée (QR-Bill Type S - Structured)
        street_name: Nom de rue (max 70 caractères)
        building_number: Numéro de bâtiment (max 16 caractères)
        postal_code: Code postal (4 chiffres pour CH)
        city: Ville (max 35 caractères)
        country_code: Code pays ISO (2 lettres, défaut: CH)

        # Contact facturation
        billing_email: Email de facturation (obligatoire)
        billing_phone: Téléphone de facturation (format +41...)

        # TVA
        vat_registered: Assujetti à la TVA (bool)
        vat_number: Numéro TVA (si applicable)
        vat_rate: Taux de TVA par défaut (Decimal, ex: 7.7)

        # IBAN & Paiement
        iban: IBAN standard (chiffré en base, 34 caractères max)
        qr_iban: QR-IBAN pour références QRR (optionnel)
        payment_reference_mode: Mode de référence (NONE/SCOR/QRR)
        creditor_reference_base: Base pour générer références QRR

        # Métadonnées
        is_address_validated: Adresse validée (bool)
        created_at: Date de création
        updated_at: Date de dernière modification
    """

    __tablename__ = "company_billing_profile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # === IDENTITÉ LÉGALE ===
    legal_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Nom légal de l'entreprise pour factures",
    )
    brand_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="Nom commercial (si différent du nom légal)",
    )
    uid_ide: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Numéro IDE/UID suisse (format: CHE-XXX.XXX.XXX)",
    )

    # === ADRESSE STRUCTURÉE (QR-Bill Type S compliant) ===
    street_name: Mapped[str] = mapped_column(
        String(70),
        nullable=False,
        comment="Nom de rue (sans numéro)",
    )
    building_number: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Numéro de bâtiment (peut contenir lettres: 12A)",
    )
    postal_code: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Code postal (4 chiffres pour Suisse)",
    )
    city: Mapped[str] = mapped_column(
        String(35),
        nullable=False,
        comment="Ville",
    )
    country_code: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
        default="CH",
        comment="Code pays ISO 3166-1 alpha-2",
    )

    # === CONTACT FACTURATION ===
    billing_email: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Email pour envoi factures",
    )
    billing_phone: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Téléphone facturation (format international recommandé)",
    )

    # === TVA ===
    vat_registered: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Entreprise assujettie à la TVA",
    )
    vat_number: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Numéro TVA (si assujetti)",
    )
    vat_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Taux TVA par défaut (ex: 7.7 pour 7.7%)",
    )

    # === IBAN & PAIEMENT ===
    # Note: IBAN chiffré en base via property (voir Company model)
    _iban_raw = Column(
        String(200),
        nullable=False,
        name="iban",
        comment="IBAN chiffré (format CHxx xxxx xxxx xxxx xxxx x)",
    )
    _qr_iban_raw = Column(
        String(200),
        nullable=True,
        name="qr_iban",
        comment="QR-IBAN chiffré (uniquement si références QRR)",
    )

    payment_reference_mode: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="SCOR",
        comment="Mode référence paiement: NONE, SCOR (ISO 11649), QRR (ESR)",
    )

    creditor_reference_base: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Base pour générer références QRR (si mode=QRR)",
    )

    # === PARAMÈTRES FACTURATION ===
    payment_terms_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=30,
        comment="Délai de paiement en jours",
    )
    overdue_fee: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        default=Decimal("15.00"),
        comment="Frais de retard (CHF)",
    )

    # === TEMPLATES (optionnel) ===
    legal_footer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Texte légal pied de page facture",
    )

    # === MÉTADONNÉES ===
    is_address_validated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Adresse validée (structure + existence)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=func.now(),
    )

    # === RELATIONS ===
    company = relationship("Company", back_populates="billing_profile")

    def __repr__(self):  # type: ignore[reportImplicitOverride]
        """Représentation string du profil de facturation."""
        return f"<BillingProfile(company_id={self.company_id}, legal_name='{self.legal_name}')>"

    # === PROPERTIES POUR IBAN (chiffrement/déchiffrement) ===
    # À implémenter comme dans Company model si nécessaire
    @property
    def iban(self) -> str | None:
        """Déchiffre et retourne l'IBAN."""
        raw = self._iban_raw
        if raw is None or (isinstance(raw, str) and not raw):
            return None
        # TODO: Implémenter déchiffrement (voir Company model)
        return str(raw) if not isinstance(raw, str) else raw

    @iban.setter
    def iban(self, value: str | None):
        """Chiffre et stocke l'IBAN."""
        if value is None:
            self._iban_raw = None
        else:
            # TODO: Implémenter chiffrement (voir Company model)
            self._iban_raw = value

    @property
    def qr_iban(self) -> str | None:
        """Déchiffre et retourne le QR-IBAN."""
        raw = self._qr_iban_raw
        if raw is None or (isinstance(raw, str) and not raw):
            return None
        # TODO: Implémenter déchiffrement
        return str(raw) if not isinstance(raw, str) else raw

    @qr_iban.setter
    def qr_iban(self, value: str | None):
        """Chiffre et stocke le QR-IBAN."""
        if value is None:
            self._qr_iban_raw = None
        else:
            # TODO: Implémenter chiffrement
            self._qr_iban_raw = value

    # === MÉTHODES UTILITAIRES ===
    def get_formatted_address(self) -> str:
        """Retourne l'adresse formatée pour affichage.

        Returns:
            Adresse sur 2 lignes:
            "Rue Verte 8\\n1205 Genève"
        """
        return (
            f"{self.street_name} {self.building_number}\n{self.postal_code} {self.city}"
        )

    def get_qr_address_structured(self) -> dict[str, str]:
        """Retourne l'adresse au format QR-Bill Type S (Structured).

        Returns:
            dict[str, str]: Adresse structurée conforme ISO 20022
        """
        return {
            "type": "S",  # Structured
            "name": self.legal_name,
            "street": self.street_name,
            "house_number": self.building_number,
            "postal_code": self.postal_code,
            "city": self.city,
            "country": self.country_code,
        }

    def get_vat_display_text(self) -> str:
        """Retourne le texte d'affichage du statut TVA.

        Returns:
            str: Texte formaté pour facture
        """
        if not self.vat_registered:
            return "Non assujetti à la TVA"

        if self.vat_number:
            return f"N° TVA : {self.vat_number}"

        if self.vat_rate:
            return f"TVA {self.vat_rate}% incluse"

        return "TVA applicable"

    def validate_for_invoicing(self) -> tuple[bool, list[str]]:
        """Valide que le profil est complet pour facturer.

        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Champs obligatoires
        required_fields = {
            "legal_name": self.legal_name,
            "uid_ide": self.uid_ide,
            "street_name": self.street_name,
            "building_number": self.building_number,
            "postal_code": self.postal_code,
            "city": self.city,
            "billing_email": self.billing_email,
            "billing_phone": self.billing_phone,
            "iban": self.iban,
        }

        for field_name, value in required_fields.items():
            if not value:
                errors.append(f"Champ obligatoire manquant: {field_name}")

        # Validation format code postal (Suisse)
        CH_POSTAL_CODE_LENGTH = 4
        if (
            self.country_code == "CH"
            and self.postal_code
            and (
                not self.postal_code.isdigit()
                or len(self.postal_code) != CH_POSTAL_CODE_LENGTH
            )
        ):
            errors.append("Code postal invalide (doit être 4 chiffres pour CH)")

        # Validation TVA
        if self.vat_registered and not self.vat_number and not self.vat_rate:
            errors.append("Numéro TVA ou taux TVA requis si assujetti")

        # Validation mode paiement
        valid_modes = ["NONE", "SCOR", "QRR"]
        if self.payment_reference_mode not in valid_modes:
            errors.append(f"Mode paiement invalide: {self.payment_reference_mode}")

        if self.payment_reference_mode == "QRR" and not self.qr_iban:
            errors.append("QR-IBAN requis pour mode de paiement QRR")

        return (len(errors) == 0, errors)
