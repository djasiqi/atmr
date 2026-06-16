from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from typing import Tuple, Union
from zoneinfo import ZoneInfo

import pytz

# ---------------------------------------------------------------------------
# MODE NAÏF LOCAL (Europe/Zurich implicite) - AUCUNE CONVERSION UTC
# ---------------------------------------------------------------------------

# Fuseau business par défaut (surchargeable par variable d'env)
LOCAL_TZ = ZoneInfo(os.getenv("UD_DEFAULT_TZ", "Europe/Zurich"))


def _normalize_local_string(s: str) -> str:
    """Normalise quelques variations communes (sans TZ) :
    - remplace séparateur espace par 'T' si nécessaire
    - conserve les fractions de secondes si présentes.
    """
    s = s.strip()
    if " " in s and "T" not in s:
        s = s.replace(" ", "T")
    return s


def parse_local_naive(dt: Union[str, datetime, None]) -> datetime | None:
    """Retourne un datetime **naïf** (ou None) en interprétant
    toute entrée comme locale."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        # On retire toute tzinfo éventuelle
        return dt.replace(tzinfo=None)
    s = _normalize_local_string(str(dt))
    return datetime.fromisoformat(s)


def now_local() -> datetime:
    """Horloge unique : maintenant en **naïf local** (Europe/Zurich)."""
    # Get current UTC time
    now_utc = datetime.now(UTC)
    # Convert to Europe/Zurich
    now_zurich = now_utc.astimezone(pytz.timezone("Europe/Zurich"))
    # Return as naive datetime
    return now_zurich.replace(tzinfo=None)


def minutes_from_now_local(dt: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre maintenant (local naïf) et dt (naïf)."""
    if not dt:
        return 10**9
    try:
        target = parse_local_naive(dt)
    except Exception:
        return 10**9
    if target is None:
        return 10**9
    delta = target - now_local()
    return max(0, int(delta.total_seconds() // 60))


def minutes_between_local(
    a: Union[str, datetime, None], b: Union[str, datetime, None]
) -> int:
    """Minutes (>=0) entre a et b (tous naïfs)."""
    if not a or not b:
        return 0
    try:
        aa = parse_local_naive(a)
        bb = parse_local_naive(b)
    except Exception:
        return 0
    if aa is None or bb is None:
        return 0
    return max(0, int((aa - bb).total_seconds() // 60))


def sort_key_local(dt: Union[str, datetime, None]) -> datetime:
    """Clé de tri sûre (naïf local). None => +50 ans."""
    if not dt:
        return now_local() + timedelta(days=0.365 * 50)
    try:
        parsed = parse_local_naive(dt)
    except Exception:
        return now_local() + timedelta(days=0.365 * 50)
    if parsed is None:
        return now_local() + timedelta(days=0.365 * 50)
    return parsed


def split_date_time_local(
    dt: Union[str, datetime, None],
) -> Tuple[str | None, str | None]:
    """Retourne ('DD.MM.YYYY', 'HH:MM') au format suisse - pour l'affichage."""
    d = parse_local_naive(dt)
    if not d:
        return None, None
    # ✅ Format suisse : dd.MM.yyyy (au lieu de YYYY-MM-DD)
    return d.strftime("%d.%m.%Y"), d.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Bornes locales naïves pour un jour 'YYYY-MM-DD'
# ---------------------------------------------------------------------------


def day_local_bounds(day_str: str) -> tuple[datetime, datetime]:
    """Minuit inclus → minuit du jour suivant (naïf)."""
    y, m, d = map(int, day_str.split("-"))
    start = datetime(y, m, d, 0, 0, 0)
    end = start + timedelta(days=1)
    return start, end


def coerce_local_day(value: Union[str, date]) -> str:
    """Normalise en 'YYYY-MM-DD'."""
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip().replace("/", "-")
    y, m, d = map(int, s.split("-"))
    _ = date(y, m, d)
    return f"{y:04d}-{m:02d}-{d:02d}"


__all__ = [
    "coerce_local_day",
    "day_local_bounds",
    "minutes_between_local",
    "minutes_from_now_local",
    "now_local",
    "parse_iso8601",
    "parse_local_naive",
    "sort_key_local",
    "split_date_time_local",
]


def parse_iso8601(dt_string: str | None) -> datetime | None:
    """Parse une chaîne ISO8601 avec timezone en datetime aware.

    Supporte les formats:
    - 2026-02-04T14:30:00Z
    - 2026-02-04T14:30:00+01:00
    - 2026-02-04T14:30:00.123456+01:00
    - 2026-02-04 14:30:00 (sans TZ, interprété comme LOCAL_TZ)

    Args:
        dt_string: Chaîne ISO8601 ou None

    Returns:
        datetime aware ou None
    """
    if not dt_string:
        return None

    try:
        s = dt_string.strip()

        # Normaliser espace en T
        if " " in s and "T" not in s:
            s = s.replace(" ", "T")

        # Parser ISO8601
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))

        # Si naïf, assumer LOCAL_TZ
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)

        return dt

    except (ValueError, TypeError):
        return None


# RÉTRO-COMPAT UTC : **ici, on corrige pour du vrai UTC aware**
# ---------------------------------------------------------------------------
def _ensure_dt(obj: Union[str, datetime, None]) -> datetime | None:
    if obj is None:
        return None
    if isinstance(obj, datetime):
        return obj
    # chaîne → essaye ISO, interprété comme local naïf si sans TZ
    return parse_local_naive(obj)


def to_utc(dt: Union[str, datetime, None]) -> datetime | None:
    """Normalise en **UTC aware**.
    - Si dt est naïf: supposé en LOCAL_TZ → converti en UTC.
    - Si dt est aware: astimezone(UTC).
    """
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(UTC)


def to_utc_from_db(dt: Union[str, datetime, None]) -> datetime | None:
    """Datetime venant de la DB (souvent aware si timezone=True).
    Ramène toujours en **UTC aware**.
    """
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(UTC)


def ensure_aware_utc(dt: Union[str, datetime, None]) -> datetime | None:
    """Alias explicite vers to_utc (UTC aware)."""
    return to_utc(dt)


def now_utc() -> datetime:
    """Maintenant en **UTC aware** (pour comparer avec des DateTime timezone=True)."""
    return datetime.now(UTC)


def minutes_from_now(dt: Union[str, datetime, None]) -> int:
    """Minutes (>=0) entre maintenant (UTC) et dt (normalisé UTC)."""
    d = to_utc(dt)
    if d is None:
        return 10**9
    delta = d - now_utc()
    return max(0, int(delta.total_seconds() // 60))


def minutes_between(
    a: Union[str, datetime, None], b: Union[str, datetime, None]
) -> int:
    """Minutes (>=0) entre a et b (tous deux normalisés UTC)."""
    aa, bb = to_utc(a), to_utc(b)
    if aa is None or bb is None:
        return 0
    return max(0, int((aa - bb).total_seconds() // 60))


def sort_key_utc(dt: Union[str, datetime, None]) -> datetime:
    """Clé de tri sûre en **UTC aware**."""
    d = to_utc(dt)
    return d if d is not None else (now_utc() + timedelta(days=0.365 * 50))


def to_geneva_local(dt: Union[str, datetime, None]) -> datetime | None:
    """Retourne un datetime **aware** en fuseau LOCAL_TZ."""
    d = _ensure_dt(dt)
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    return d.astimezone(LOCAL_TZ)


def geneva_naive_midnight_from_date_ymd(ymd: str) -> datetime | None:
    """Construit minuit (datetime naïf fuseau métier) pour une date ``YYYY-MM-DD``.

    Utilisé notamment pour un retour aller-retour dont la **date** est connue
    mais l'heure reste à confirmer (sentinelle 00:00 + ``time_confirmed=False``).
    """
    s = str(ymd).strip()
    if not s:
        return None
    try:
        d = date.fromisoformat(s)
    except ValueError:
        return None
    return datetime(d.year, d.month, d.day, 0, 0, 0)


def is_return_time_pending(dt: datetime | None) -> bool:
    """True si l'heure de retour est la sentinelle « à définir » (00:00:00).

    Toute logique métier (dispatch, notifications, planification) doit traiter
    cette valeur comme « heure non définie », jamais comme minuit réel.
    """
    if dt is None:
        return True
    st = api_scheduled_iso_to_naive_geneva(dt)
    if st is None:
        return False
    return st.hour == 0 and st.minute == 0 and st.second == 0


PROPOSED_PICKUP_MAX_DAYS = 365


def validate_proposed_pickup_time(
    value: Union[str, datetime],
) -> tuple[datetime | None, str | None]:
    """Valide et normalise un horaire proposé (ISO8601 → naive Geneva).

    Règles:
    - format ISO8601 valide obligatoire
    - timezone normalisée en UTC pour les comparaisons
    - date passée refusée
    - date > maintenant + 365 jours refusée

    Returns:
        (datetime_naive_geneva, error_message)
    """
    parsed = parse_iso8601(value) if isinstance(value, str) else to_utc(value)
    if parsed is None:
        return None, "Format d'horaire invalide (ISO8601 attendu)"

    utc_dt = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    now_utc = datetime.now(UTC)
    if utc_dt <= now_utc:
        return None, "L'horaire proposé doit être dans le futur"
    if utc_dt > now_utc + timedelta(days=PROPOSED_PICKUP_MAX_DAYS):
        return None, (
            f"L'horaire proposé ne peut pas dépasser {PROPOSED_PICKUP_MAX_DAYS} jours"
        )

    naive_geneva = api_scheduled_iso_to_naive_geneva(utc_dt)
    if naive_geneva is None:
        return None, "Format d'horaire invalide (ISO8601 attendu)"
    return naive_geneva, None


def naive_geneva_to_db_aware(value: Union[str, datetime, None]) -> datetime | None:
    """Convertit une heure murale Genève en datetime aware pour colonnes timestamptz.

    Les datetime naïfs écrits tels quels en timestamptz sont interprétés comme UTC
    par PostgreSQL (session UTC), ce qui décale l'affichage de +1 à +2 h.
    """
    naive = api_scheduled_iso_to_naive_geneva(value)
    if naive is None:
        return None
    return naive.replace(tzinfo=LOCAL_TZ)


def mission_scheduled_to_api_iso(value: Union[str, datetime, None]) -> str | None:
    """Sérialise un horaire mission DB/API en ISO naïf Genève (sans suffixe Z).

    Aligné sur ``split_date_time_local`` côté entreprise : les colonnes timestamptz
    conservent l'heure murale saisie ; on retire tzinfo sans conversion UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        naive = parse_local_naive(value)
        return naive.strftime("%Y-%m-%dT%H:%M:%S") if naive else None
    naive = api_scheduled_iso_to_naive_geneva(value)
    if naive is None:
        return None
    return naive.strftime("%Y-%m-%dT%H:%M:%S")


def api_scheduled_iso_to_naive_geneva(
    value: Union[str, datetime, None],
) -> datetime | None:
    """Convertit une date/heure de réservation issue du portail (ISO, souvent UTC « Z »).

    - Chaîne avec fuseau (Z ou offset) : interprétée comme instant absolu puis
      ramenée à l'horloge Europe/Zurich **sans** tzinfo (convention métier DB).
    - Chaîne sans fuseau : ``parse_iso8601`` suppose déjà LOCAL_TZ, puis idem.
    - ``datetime`` aware : conversion vers l'horloge locale naïve.
    - ``datetime`` naïf : considéré déjà en heure locale métier.

    Évite l'ancien piège ``replace(tzinfo=None)`` sur un UTC aware (perte de
    1-2 h vs l'heure affichée côté client) et l'ambiguïté JSON ``T...`` sans « Z »
    interprété comme local par les navigateurs.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            loc = to_geneva_local(value)
            return loc.replace(tzinfo=None) if loc else None
        return value.replace(tzinfo=None)
    s = str(value).strip()
    parsed = parse_iso8601(s)
    if parsed is not None:
        loc = to_geneva_local(parsed)
        return loc.replace(tzinfo=None) if loc else None
    return parse_local_naive(s)


def normalize_mission_wall_clock(
    value: Union[str, datetime, None],
) -> datetime | None:
    """Point d'entrée UNIQUE pour toute écriture d'horaire mission.

    Entrée : ISO ``YYYY-MM-DDTHH:MM:SS`` (avec ou sans tz) ou ``datetime``.
    Sortie : ``datetime`` NAIF représentant l'heure murale Genève.

    IMPORTANT : la sortie dépend de la VALEUR RÉELLE de l'entrée, pas du libellé.
    - Entrée naïve 12:30  -> 12:30 (considérée déjà heure murale Genève)
    - Entrée offset 12:30+02:00 -> 12:30 (déjà Genève été)
    - Entrée 10:30Z (UTC) -> 12:30 Genève été (instant absolu converti en mural)

    RÈGLE D'ARCHITECTURE : point d'entrée unique des ÉCRITURES mission.
    ``parse_iso8601()`` (aware) est interdit pour les écritures mission.
    Réservé à la validation/comparaison et aux flux non-mission.
    """
    return api_scheduled_iso_to_naive_geneva(value)


def format_geneva(dt: Union[str, datetime, None]) -> Tuple[str | None, str | None]:
    """Retourne ('YYYY-MM-DD','HH:MM') en **LOCAL_TZ** (aware),
    pratique pour l'affichage."""
    d = to_geneva_local(dt)
    if not d:
        return None, None
    return d.strftime("%Y-%m-%d"), d.strftime("%H:%M")


def iso_utc_z(dt: Union[str, datetime, None]) -> str | None:
    """ISO 8601 en UTC, suffixé 'Z' (ex: 2025-09-21T10:30:00Z)."""
    d = to_utc(dt)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ") if d else None


# Exposer également ces noms pour ne pas casser les imports
__all__ += [
    "api_scheduled_iso_to_naive_geneva",
    "ensure_aware_utc",
    "format_geneva",
    "geneva_naive_midnight_from_date_ymd",
    "iso_utc_z",
    "mission_scheduled_to_api_iso",
    "minutes_between",
    "minutes_from_now",
    "naive_geneva_to_db_aware",
    "normalize_mission_wall_clock",
    "now_utc",
    "sort_key_utc",
    "to_geneva_local",
    "to_utc",
    "to_utc_from_db",
]
