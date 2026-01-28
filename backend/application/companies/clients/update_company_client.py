from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol


class _UserLike(Protocol):
    birth_date: Any
    gender: Any
    first_name: Any
    last_name: Any
    phone: Any


class _ClientLike(Protocol):
    user: _UserLike | None
    contact_email: Any
    contact_phone: Any
    billing_address: Any
    billing_lat: Any
    billing_lon: Any
    is_active: Any
    is_institution: Any
    institution_name: Any
    residence_facility: Any
    domicile_address: Any
    domicile_zip: Any
    domicile_city: Any
    domicile_lat: Any
    domicile_lon: Any
    preferential_rate: Any
    avs_number: Any
    door_code: Any
    floor: Any
    access_notes: Any
    gp_name: Any
    gp_phone: Any
    default_billed_to_type: Any
    default_billed_to_contact: Any


@dataclass(frozen=True, slots=True)
class UpdateCompanyClientResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class UpdateCompanyClientUseCase:
    """Use-case Application: mise à jour d'un client
    (coordonnées, facturation, statut, etc.)."""

    def execute(
        self, *, client: _ClientLike, data: dict[str, Any]
    ) -> UpdateCompanyClientResult:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "📝 [UpdateCompanyClientUseCase] Début mise à jour client ID=%s, données reçues: %s",
            getattr(client, "id", "N/A"),
            data,
        )

        # Champs Client
        if "contact_email" in data:
            client.contact_email = data["contact_email"] or None
            logger.info("📝 [UpdateCompanyClientUseCase] contact_email mis à jour: %s", client.contact_email)
        if "contact_phone" in data:
            client.contact_phone = data["contact_phone"] or None
            logger.info("📝 [UpdateCompanyClientUseCase] contact_phone mis à jour: %s", client.contact_phone)
        if "billing_address" in data:
            client.billing_address = data["billing_address"]
        if "billing_lat" in data:
            client.billing_lat = data["billing_lat"]
        if "billing_lon" in data:
            client.billing_lon = data["billing_lon"]
        if "is_active" in data:
            client.is_active = bool(data["is_active"])

        if "is_institution" in data:
            client.is_institution = bool(data["is_institution"])
            if bool(client.is_institution) and "institution_name" in data:
                client.institution_name = data["institution_name"]
            elif not bool(client.is_institution):
                client.institution_name = None

        if "residence_facility" in data:
            client.residence_facility = data["residence_facility"] or None

        if "domicile_address" in data:
            old_value = client.domicile_address
            client.domicile_address = data["domicile_address"] or None
            logger.info(
                "📝 [UpdateCompanyClientUseCase] domicile_address: %s -> %s",
                old_value,
                client.domicile_address,
            )
        if "domicile_zip" in data:
            old_value = client.domicile_zip
            client.domicile_zip = data["domicile_zip"] or None
            logger.info(
                "📝 [UpdateCompanyClientUseCase] domicile_zip: %s -> %s",
                old_value,
                client.domicile_zip,
            )
        if "domicile_city" in data:
            old_value = client.domicile_city
            client.domicile_city = data["domicile_city"] or None
            logger.info(
                "📝 [UpdateCompanyClientUseCase] domicile_city: %s -> %s",
                old_value,
                client.domicile_city,
            )
        if "domicile_lat" in data:
            old_value = client.domicile_lat
            client.domicile_lat = data["domicile_lat"]
            logger.info(
                "📝 [UpdateCompanyClientUseCase] domicile_lat: %s -> %s",
                old_value,
                client.domicile_lat,
            )
        if "domicile_lon" in data:
            old_value = client.domicile_lon
            client.domicile_lon = data["domicile_lon"]
            logger.info(
                "📝 [UpdateCompanyClientUseCase] domicile_lon: %s -> %s",
                old_value,
                client.domicile_lon,
            )

        if "preferential_rate" in data:
            rate_value = data["preferential_rate"]
            old_value = client.preferential_rate
            if rate_value == "" or rate_value is None:
                client.preferential_rate = None
                logger.info(
                    "📝 [UpdateCompanyClientUseCase] preferential_rate: %s -> None (vide)",
                    old_value,
                )
            else:
                try:
                    client.preferential_rate = Decimal(str(rate_value))
                    logger.info(
                        "📝 [UpdateCompanyClientUseCase] preferential_rate: %s -> %s",
                        old_value,
                        client.preferential_rate,
                    )
                except (ValueError, TypeError) as e:
                    logger.error("❌ [UpdateCompanyClientUseCase] Erreur tarif préférentiel: %s", e)
                    return UpdateCompanyClientResult(
                        ok=False,
                        error={"error": "Tarif préférentiel invalide"},
                        status_code=400,
                    )

        # ✅ Numéro AVS
        if "avs_number" in data:
            client.avs_number = data["avs_number"] or None

        # Accès logement
        if "door_code" in data:
            client.door_code = data["door_code"] or None
        if "floor" in data:
            client.floor = data["floor"] or None
        if "access_notes" in data:
            client.access_notes = data["access_notes"] or None

        # Médecin traitant
        if "gp_name" in data:
            client.gp_name = data["gp_name"] or None
        if "gp_phone" in data:
            client.gp_phone = data["gp_phone"] or None

        # Facturation par défaut
        if "default_billed_to_type" in data:
            v = (data["default_billed_to_type"] or "").strip().lower()
            client.default_billed_to_type = (
                v if v in ("patient", "clinic", "insurance") else "patient"
            )
        if "default_billed_to_contact" in data:
            client.default_billed_to_contact = data["default_billed_to_contact"] or None

        # Champs User (first_name, last_name, phone)
        user = getattr(client, "user", None)
        if user is not None:
            if "first_name" in data:
                user.first_name = (data["first_name"] or "").strip() or None
            if "last_name" in data:
                user.last_name = (data["last_name"] or "").strip() or None
            if "phone" in data:
                user.phone = (data["phone"] or "").strip() or None

        # Champ User.birth_date (format YYYY-MM-DD)
        if "birth_date" in data and getattr(client, "user", None):
            user = client.user
            assert user is not None
            birth_date_value = data["birth_date"]
            if birth_date_value:
                try:
                    user.birth_date = datetime.strptime(
                        str(birth_date_value), "%Y-%m-%d"
                    ).date()
                except (ValueError, TypeError):
                    return UpdateCompanyClientResult(
                        ok=False,
                        error={
                            "error": (
                                "Format de date de naissance invalide. "
                                "Utiliser YYYY-MM-DD."
                            )
                        },
                        status_code=400,
                    )
            else:
                user.birth_date = None

        # ✅ Champ User.gender
        if "gender" in data and getattr(client, "user", None):
            user = client.user
            assert user is not None
            gender_value = data["gender"]
            old_gender = user.gender
            logger.info(
                "📝 [UpdateCompanyClientUseCase] gender reçu: %s (ancien: %s)",
                gender_value,
                old_gender,
            )
            if gender_value:
                # Convertir en GenderEnum
                from models.enums import GenderEnum

                try:
                    if isinstance(gender_value, str):
                        # ✅ Accepter "male"/"female" (anglais) ET "homme"/"femme" (français)
                        # GenderEnum utilise: HOMME, FEMME, AUTRE
                        gender_str = gender_value.lower().strip()

                        # Mapping: anglais → français
                        if gender_str in {"male", "homme"}:
                            user.gender = GenderEnum.HOMME
                            logger.info(
                                "📝 [UpdateCompanyClientUseCase] gender: %s -> HOMME",
                                old_gender,
                            )
                        elif gender_str in {"female", "femme"}:
                            user.gender = GenderEnum.FEMME
                            logger.info(
                                "📝 [UpdateCompanyClientUseCase] gender: %s -> FEMME",
                                old_gender,
                            )
                        elif gender_str in {"autre", "other"}:
                            user.gender = GenderEnum.AUTRE
                            logger.info(
                                "📝 [UpdateCompanyClientUseCase] gender: %s -> AUTRE",
                                old_gender,
                            )
                        else:
                            # Essayer directement avec la valeur (si déjà au bon format)
                            try:
                                user.gender = GenderEnum(gender_value.upper())
                            except (ValueError, AttributeError):
                                return UpdateCompanyClientResult(
                                    ok=False,
                                    error={
                                        "error": (
                                            "Genre invalide. Utiliser 'male'/'homme', "
                                            "'female'/'femme' ou 'autre'/'other'."
                                        )
                                    },
                                    status_code=400,
                                )
                    else:
                        # Si c'est déjà un GenderEnum
                        user.gender = gender_value
                except (ValueError, TypeError, AttributeError) as e:
                    return UpdateCompanyClientResult(
                        ok=False,
                        error={"error": f"Genre invalide: {e!s}"},
                        status_code=400,
                    )
            else:
                user.gender = None
                logger.info(
                    "📝 [UpdateCompanyClientUseCase] gender: %s -> None (vide)",
                    old_gender,
                )

        # ✅ Synchroniser la Company clinique si le client est une institution
        try:
            if bool(getattr(client, "is_institution", False)):
                sync_fields = {
                    "domicile_address",
                    "domicile_zip",
                    "domicile_city",
                    "domicile_lat",
                    "domicile_lon",
                    "contact_email",
                    "contact_phone",
                    "preferential_rate",
                    "institution_name",
                }
                if sync_fields.intersection(data.keys()):
                    from models import Company, db  # import local pour éviter cycles

                    clinic_company = None
                    clinic_company_id = getattr(
                        client, "default_billed_to_company_id", None
                    )
                    if clinic_company_id:
                        clinic_company = Company.query.filter_by(
                            id=clinic_company_id
                        ).first()
                    if not clinic_company and client.institution_name:
                        clinic_company = Company.query.filter_by(
                            name=client.institution_name
                        ).first()
                        if clinic_company and not clinic_company_id:
                            client.default_billed_to_company_id = clinic_company.id
                    if clinic_company:
                        domicile_address = getattr(client, "domicile_address", None) or ""
                        domicile_zip = getattr(client, "domicile_zip", None) or ""
                        domicile_city = getattr(client, "domicile_city", None) or ""
                        postal_city = " ".join(
                            part for part in [domicile_zip, domicile_city] if part
                        )
                        full_address = (
                            f"{domicile_address}, {postal_city}".strip(", ")
                            if domicile_address or postal_city
                            else ""
                        )

                        if "institution_name" in data and client.institution_name:
                            clinic_company.name = client.institution_name
                        clinic_company.address = full_address or clinic_company.address
                        clinic_company.domicile_address_line1 = (
                            domicile_address or clinic_company.domicile_address_line1
                        )
                        clinic_company.domicile_zip = (
                            domicile_zip or clinic_company.domicile_zip
                        )
                        clinic_company.domicile_city = (
                            domicile_city or clinic_company.domicile_city
                        )

                        if "domicile_lat" in data:
                            clinic_company.latitude = client.domicile_lat
                        if "domicile_lon" in data:
                            clinic_company.longitude = client.domicile_lon
                        if "contact_email" in data:
                            clinic_company.contact_email = client.contact_email
                        if "contact_phone" in data:
                            clinic_company.contact_phone = client.contact_phone
                        if "preferential_rate" in data:
                            clinic_company.preferential_rate = client.preferential_rate

                        db.session.add(clinic_company)
                        logger.info(
                            "✅ [UpdateCompanyClientUseCase] Company clinique %s synchronisée",
                            clinic_company.id,
                        )
        except Exception as sync_error:
            logger.warning(
                "⚠️ [UpdateCompanyClientUseCase] Erreur sync company clinique: %s",
                sync_error,
            )

        logger.info(
            "✅ [UpdateCompanyClientUseCase] Mise à jour client ID=%s terminée avec succès",
            getattr(client, "id", "N/A"),
        )
        return UpdateCompanyClientResult(ok=True)
