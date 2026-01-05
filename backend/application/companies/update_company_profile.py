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

        for k, v in validated_data.items():
            if k in self._ALLOWED_FIELDS:
                setattr(company, k, v)

        return UpdateCompanyProfileResult(
            ok=True,
            geocoded=geocoded,
            geocoded_lat=geo_lat,
            geocoded_lon=geo_lon,
        )
