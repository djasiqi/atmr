from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol


class _UserLike(Protocol):
    birth_date: Any
    gender: Any  # ✅ Ajout du genre


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
    avs_number: Any  # ✅ Ajout du numéro AVS


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
        logger.info("📝 [UpdateCompanyClientUseCase] Début mise à jour client ID=%s, données reçues: %s", 
                   getattr(client, 'id', 'N/A'), data)
        
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
            logger.info("📝 [UpdateCompanyClientUseCase] domicile_address: %s -> %s", old_value, client.domicile_address)
        if "domicile_zip" in data:
            old_value = client.domicile_zip
            client.domicile_zip = data["domicile_zip"] or None
            logger.info("📝 [UpdateCompanyClientUseCase] domicile_zip: %s -> %s", old_value, client.domicile_zip)
        if "domicile_city" in data:
            old_value = client.domicile_city
            client.domicile_city = data["domicile_city"] or None
            logger.info("📝 [UpdateCompanyClientUseCase] domicile_city: %s -> %s", old_value, client.domicile_city)
        if "domicile_lat" in data:
            old_value = client.domicile_lat
            client.domicile_lat = data["domicile_lat"]
            logger.info("📝 [UpdateCompanyClientUseCase] domicile_lat: %s -> %s", old_value, client.domicile_lat)
        if "domicile_lon" in data:
            old_value = client.domicile_lon
            client.domicile_lon = data["domicile_lon"]
            logger.info("📝 [UpdateCompanyClientUseCase] domicile_lon: %s -> %s", old_value, client.domicile_lon)

        if "preferential_rate" in data:
            rate_value = data["preferential_rate"]
            old_value = client.preferential_rate
            if rate_value == "" or rate_value is None:
                client.preferential_rate = None
                logger.info("📝 [UpdateCompanyClientUseCase] preferential_rate: %s -> None (vide)", old_value)
            else:
                try:
                    client.preferential_rate = Decimal(str(rate_value))
                    logger.info("📝 [UpdateCompanyClientUseCase] preferential_rate: %s -> %s", old_value, client.preferential_rate)
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
            logger.info("📝 [UpdateCompanyClientUseCase] gender reçu: %s (ancien: %s)", gender_value, old_gender)
            if gender_value:
                # Convertir en GenderEnum
                from models.enums import GenderEnum

                try:
                    if isinstance(gender_value, str):
                        # ✅ Accepter "male"/"female" (anglais) ET "homme"/"femme" (français)
                        # GenderEnum utilise: HOMME, FEMME, AUTRE
                        gender_str = gender_value.lower().strip()
                        
                        # Mapping: anglais → français
                        if gender_str == "male" or gender_str == "homme":
                            user.gender = GenderEnum.HOMME
                            logger.info("📝 [UpdateCompanyClientUseCase] gender: %s -> HOMME", old_gender)
                        elif gender_str == "female" or gender_str == "femme":
                            user.gender = GenderEnum.FEMME
                            logger.info("📝 [UpdateCompanyClientUseCase] gender: %s -> FEMME", old_gender)
                        elif gender_str == "autre" or gender_str == "other":
                            user.gender = GenderEnum.AUTRE
                            logger.info("📝 [UpdateCompanyClientUseCase] gender: %s -> AUTRE", old_gender)
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
                        error={"error": f"Genre invalide: {str(e)}"},
                        status_code=400,
                    )
            else:
                user.gender = None
                logger.info("📝 [UpdateCompanyClientUseCase] gender: %s -> None (vide)", old_gender)

        logger.info("✅ [UpdateCompanyClientUseCase] Mise à jour client ID=%s terminée avec succès", 
                   getattr(client, 'id', 'N/A'))
        return UpdateCompanyClientResult(ok=True)
