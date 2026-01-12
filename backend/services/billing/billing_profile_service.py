"""Service de gestion des profils de facturation.

Ce service gère les opérations CRUD sur CompanyBillingProfile
et garantit la cohérence des données de facturation.
"""

# ruff: noqa: G004
import logging
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ext import db
from models import Company, CompanyBillingProfile

logger = logging.getLogger(__name__)


class BillingProfileService:
    """Service de gestion des profils de facturation."""

    @staticmethod
    def get_by_company_id(company_id: int) -> CompanyBillingProfile | None:
        """Récupère le profil de facturation d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            CompanyBillingProfile | None: Profil trouvé ou None
        """
        try:
            return CompanyBillingProfile.query.filter_by(company_id=company_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Erreur récupération profil (company_id={company_id}): {e}")
            return None

    @staticmethod
    def get_by_id(profile_id: int) -> CompanyBillingProfile | None:
        """Récupère un profil par son ID.

        Args:
            profile_id: ID du profil

        Returns:
            CompanyBillingProfile | None: Profil trouvé ou None
        """
        try:
            return CompanyBillingProfile.query.get(profile_id)
        except SQLAlchemyError as e:
            logger.error(f"Erreur récupération profil (ID={profile_id}): {e}")
            return None

    @staticmethod
    def create(
        company_id: int, data: dict[str, Any]
    ) -> tuple[CompanyBillingProfile | None, list[str]]:
        """Crée un nouveau profil de facturation.

        Args:
            company_id: ID de l'entreprise
            data: Données du profil

        Returns:
            tuple[CompanyBillingProfile | None, list[str]]: (Profil créé, Erreurs)
        """
        errors = []

        try:
            # Vérifier que l'entreprise existe
            company = Company.query.get(company_id)
            if not company:
                errors.append(f"Entreprise {company_id} introuvable")
                return (None, errors)

            # Vérifier qu'un profil n'existe pas déjà
            existing = BillingProfileService.get_by_company_id(company_id)
            if existing:
                errors.append(f"Un profil existe déjà pour l'entreprise {company_id}")
                return (None, errors)

            # Créer le profil
            profile = CompanyBillingProfile(
                company_id=company_id,
                legal_name=data.get("legal_name", company.name or "[Non configuré]"),
                brand_name=data.get("brand_name"),
                uid_ide=data.get("uid_ide", company.uid_ide or "[Non configuré]"),
                street_name=data.get("street_name", "[Non configuré]"),
                building_number=data.get("building_number", "0"),
                postal_code=data.get("postal_code", "[Non configuré]"),
                city=data.get("city", "[Non configuré]"),
                country_code=data.get("country_code", "CH"),
                billing_email=data.get("billing_email", "[Non configuré]"),
                billing_phone=data.get("billing_phone", "[Non configuré]"),
                vat_registered=data.get("vat_registered", False),
                vat_number=data.get("vat_number"),
                vat_rate=Decimal(str(data["vat_rate"]))
                if data.get("vat_rate")
                else None,
                iban=data.get("iban", "[Non configuré]"),
                qr_iban=data.get("qr_iban"),
                payment_reference_mode=data.get("payment_reference_mode", "SCOR"),
                creditor_reference_base=data.get("creditor_reference_base"),
                payment_terms_days=data.get("payment_terms_days", 30),
                overdue_fee=Decimal(str(data.get("overdue_fee", "15.00"))),
                legal_footer=data.get("legal_footer"),
                is_address_validated=data.get("is_address_validated", False),
            )

            # Valider avant création
            is_valid, validation_errors = profile.validate_for_invoicing()
            if not is_valid:
                logger.warning(
                    f"Profil créé mais avec des erreurs de validation : {validation_errors}"
                )
                # On crée quand même le profil, mais on log les warnings

            db.session.add(profile)
            db.session.commit()

            logger.info(
                f"✅ Profil créé (ID={profile.id}) pour company_id={company_id}"
            )
            return (profile, errors)

        except IntegrityError as e:
            db.session.rollback()
            logger.error(f"Erreur IntegrityError création profil: {e}")
            errors.append("Erreur d'intégrité (profil déjà existant ?)")
            return (None, errors)
        except Exception as e:
            db.session.rollback()
            logger.error(f"Erreur création profil: {e}")
            errors.append(f"Erreur lors de la création: {e!s}")
            return (None, errors)

    @staticmethod
    def update(
        profile_id: int, data: dict[str, Any]
    ) -> tuple[CompanyBillingProfile | None, list[str]]:
        """Met à jour un profil existant.

        Args:
            profile_id: ID du profil à mettre à jour
            data: Données à modifier

        Returns:
            tuple[CompanyBillingProfile | None, list[str]]: (Profil mis à jour, Erreurs)
        """
        errors = []

        try:
            profile = BillingProfileService.get_by_id(profile_id)
            if not profile:
                errors.append(f"Profil {profile_id} introuvable")
                return (None, errors)

            # Liste des champs modifiables
            updatable_fields = [
                "legal_name",
                "brand_name",
                "uid_ide",
                "street_name",
                "building_number",
                "postal_code",
                "city",
                "country_code",
                "billing_email",
                "billing_phone",
                "vat_registered",
                "vat_number",
                "vat_rate",
                "iban",
                "qr_iban",
                "payment_reference_mode",
                "creditor_reference_base",
                "payment_terms_days",
                "overdue_fee",
                "legal_footer",
                "is_address_validated",
            ]

            # Mettre à jour les champs fournis
            for field in updatable_fields:
                if field in data:
                    value = data[field]

                    # Conversion spéciale pour Decimal
                    if field in ["vat_rate", "overdue_fee"] and value is not None:
                        value = Decimal(str(value))

                    setattr(profile, field, value)

            # Valider avant mise à jour
            is_valid, validation_errors = profile.validate_for_invoicing()
            if not is_valid:
                logger.warning(
                    f"Profil mis à jour mais avec des erreurs de validation : {validation_errors}"
                )

            db.session.commit()

            logger.info(f"✅ Profil {profile_id} mis à jour")
            return (profile, errors)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Erreur mise à jour profil {profile_id}: {e}")
            errors.append(f"Erreur lors de la mise à jour: {e!s}")
            return (None, errors)

    @staticmethod
    def delete(profile_id: int) -> tuple[bool, list[str]]:
        """Supprime un profil.

        Args:
            profile_id: ID du profil à supprimer

        Returns:
            tuple[bool, list[str]]: (Succès, Erreurs)
        """
        errors = []

        try:
            profile = BillingProfileService.get_by_id(profile_id)
            if not profile:
                errors.append(f"Profil {profile_id} introuvable")
                return (False, errors)

            db.session.delete(profile)
            db.session.commit()

            logger.info(f"✅ Profil {profile_id} supprimé")
            return (True, errors)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Erreur suppression profil {profile_id}: {e}")
            errors.append(f"Erreur lors de la suppression: {e!s}")
            return (False, errors)

    @staticmethod
    def validate(profile_id: int) -> tuple[bool, list[str]]:
        """Valide un profil pour la facturation.

        Args:
            profile_id: ID du profil à valider

        Returns:
            tuple[bool, list[str]]: (Est valide, Liste des erreurs)
        """
        profile = BillingProfileService.get_by_id(profile_id)
        if not profile:
            return (False, [f"Profil {profile_id} introuvable"])

        return profile.validate_for_invoicing()

    @staticmethod
    def mark_as_validated(profile_id: int) -> tuple[bool, list[str]]:
        """Marque une adresse comme validée.

        Args:
            profile_id: ID du profil

        Returns:
            tuple[bool, list[str]]: (Succès, Erreurs)
        """
        errors = []

        try:
            profile = BillingProfileService.get_by_id(profile_id)
            if not profile:
                errors.append(f"Profil {profile_id} introuvable")
                return (False, errors)

            profile.is_address_validated = True
            db.session.commit()

            logger.info(f"✅ Profil {profile_id} marqué comme validé")
            return (True, errors)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Erreur validation profil {profile_id}: {e}")
            errors.append(f"Erreur lors de la validation: {e!s}")
            return (False, errors)

    @staticmethod
    def get_or_create_for_company(
        company_id: int,
    ) -> tuple[CompanyBillingProfile | None, bool, list[str]]:
        """Récupère ou crée un profil pour une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            tuple[CompanyBillingProfile | None, bool, list[str]]: (Profil, Créé?, Erreurs)
        """
        # Essayer de récupérer un profil existant
        profile = BillingProfileService.get_by_company_id(company_id)
        if profile:
            return (profile, False, [])

        # Sinon, en créer un avec des valeurs par défaut
        company = Company.query.get(company_id)
        if not company:
            return (None, False, [f"Entreprise {company_id} introuvable"])

        default_data = {
            "legal_name": company.name or "[Non configuré]",
            "uid_ide": company.uid_ide or "[Non configuré]",
            "billing_email": company.billing_email or "[Non configuré]",
            "billing_phone": company.contact_phone or "[Non configuré]",
        }

        profile, errors = BillingProfileService.create(company_id, default_data)
        if profile:
            logger.info(f"✅ Profil créé automatiquement pour company_id={company_id}")
            return (profile, True, errors)

        return (None, False, errors)
