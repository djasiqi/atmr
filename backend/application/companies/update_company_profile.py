from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Protocol


class _CompanyLike(Protocol):
    id: int | None


@dataclass(frozen=True, slots=True)
class UpdateCompanyProfileResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    geocoded: bool = False
    geocoded_lat: float | None = None
    geocoded_lon: float | None = None
    billing_profile_synced: bool = False


class UpdateCompanyProfileUseCase:
    """Use-case Application: mise à jour du profil entreprise.

    - applique une liste blanche de champs
    - déclenche un géocodage (via fn injectée) si address fournie et coords absentes
    - ne commit pas (géré par la route / UoW)
    """

    _ALLOWED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "address",
            "latitude",
            "longitude",
            "contact_email",
            "contact_phone",
            "billing_email",
            "billing_notes",
            "iban",
            "uid_ide",
            "domicile_address_line1",
            "domicile_address_line2",
            "domicile_zip",
            "domicile_city",
            "domicile_country",
            "logo_url",
        }
    )

    def __init__(self, *, geocode_fn: Callable[[str], dict[str, Any] | None]) -> None:
        super().__init__()
        self._geocode = geocode_fn

    def execute(
        self, company: _CompanyLike, *, validated_data: dict[str, Any]
    ) -> UpdateCompanyProfileResult:
        address = validated_data.get("address")
        lat = validated_data.get("latitude")
        lon = validated_data.get("longitude")

        geocoded = False
        geo_lat: float | None = None
        geo_lon: float | None = None

        if address and (not lat or not lon):
            coords = self._geocode(str(address))
            if coords:
                geo_lat_val = coords.get("lat")
                geo_lon_val = coords.get("lon")
                try:
                    geo_lat = float(geo_lat_val) if geo_lat_val is not None else None
                    geo_lon = float(geo_lon_val) if geo_lon_val is not None else None
                except Exception:
                    geo_lat = None
                    geo_lon = None
                if geo_lat is not None and geo_lon is not None:
                    validated_data["latitude"] = geo_lat
                    validated_data["longitude"] = geo_lon
                    geocoded = True

        # ✅ Détecter si des champs domicile_* sont modifiés
        domicile_fields_modified = any(
            k in validated_data
            for k in [
                "domicile_address_line1",
                "domicile_zip",
                "domicile_city",
                "domicile_country",
            ]
        )

        for k, v in validated_data.items():
            if k in self._ALLOWED_FIELDS:
                setattr(company, k, v)

        # ✅ Synchroniser CompanyBillingProfile si champs domicile_* modifiés
        billing_profile_synced = False
        if domicile_fields_modified:
            billing_profile_synced = self._sync_billing_profile(company, validated_data)

        return UpdateCompanyProfileResult(
            ok=True,
            geocoded=geocoded,
            geocoded_lat=geo_lat,
            geocoded_lon=geo_lon,
            billing_profile_synced=billing_profile_synced,
        )

    def _sync_billing_profile(
        self, company: _CompanyLike, validated_data: dict[str, Any]
    ) -> bool:
        """Synchronise CompanyBillingProfile avec les données de domicile.

        Args:
            company: Instance de Company (avec id)
            validated_data: Données validées contenant les champs domicile_*

        Returns:
            bool: True si synchronisé, False sinon
        """
        try:
            from models import CompanyBillingProfile

            # Récupérer le profil existant
            profile = CompanyBillingProfile.query.filter_by(
                company_id=company.id
            ).first()

            if not profile:
                # Pas de profil, pas de synchronisation
                return False

            # ✅ Synchroniser les champs modifiés
            # Mapping : Company.domicile_* → CompanyBillingProfile.*
            sync_mapping = {
                "domicile_address_line1": "street_name",  # Note: simplifié
                "domicile_zip": "postal_code",
                "domicile_city": "city",
                "domicile_country": "country_code",
            }

            synced = False
            for company_field, profile_field in sync_mapping.items():
                if company_field in validated_data:
                    new_value = validated_data[company_field]

                    # Cas spécial pour street_name : stocker toute l'adresse
                    if profile_field == "street_name":
                        # domicile_address_line1 contient déjà "Rue Numéro" complet
                        # On stocke tout dans street_name et on vide building_number
                        address_str = str(new_value or "").strip()
                        profile.street_name = address_str
                        profile.building_number = ""  # ✅ Vider building_number
                    else:
                        setattr(profile, profile_field, new_value)

                    synced = True

            return synced
        except Exception:
            # En cas d'erreur, on ne bloque pas la mise à jour de Company
            return False
