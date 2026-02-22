# application/institutions/institution_settings_service.py
"""Service centralisé pour les paramètres institution.

Fournit:
- get_or_create_settings(institution_id) : lazy-create 1:1
- get_offer_timeouts(institution_id)     : (same_day, default) en minutes
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytz
from sqlalchemy.exc import IntegrityError

from ext import db
from models.institution_settings import (
    DEFAULT_TIMEOUT_DEFAULT_MINUTES,
    DEFAULT_TIMEOUT_SAME_DAY_MINUTES,
    DEFAULT_TIMEZONE,
    InstitutionSettings,
)

logger = logging.getLogger(__name__)

# Timezone par défaut pour calcul same-day
_DEFAULT_TZ = pytz.timezone(DEFAULT_TIMEZONE)


def get_or_create_settings(institution_id: int) -> InstitutionSettings:
    """Retourne les settings d'une institution, les crée si absents (lazy-create).

    Thread-safe grâce à un INSERT ... ON CONFLICT DO NOTHING implicite:
    si deux requêtes concurrentes tentent de créer, la seconde lèvera
    IntegrityError et on re-query.
    """
    settings = InstitutionSettings.query.filter_by(
        institution_id=institution_id
    ).first()

    if settings is not None:
        return settings

    # Lazy-create
    settings = InstitutionSettings(institution_id=institution_id)  # type: ignore[call-arg]
    db.session.add(settings)
    try:
        db.session.flush()
        logger.info(
            "[InstitutionSettings] Auto-created settings for institution %s",
            institution_id,
        )
    except IntegrityError:
        # Race condition: une autre transaction a créé entre-temps (unique constraint)
        db.session.rollback()
        settings = InstitutionSettings.query.filter_by(
            institution_id=institution_id
        ).first()
        if settings is None:
            raise  # Vrai problème, pas un conflit de concurrence

    return settings


def get_offer_timeouts(institution_id: int) -> tuple[int, int]:
    """Retourne (same_day_minutes, default_minutes) depuis les settings.

    Fallback sur les constantes par défaut si settings absent.
    Ne crée PAS les settings (lecture seule, utilisé dans les tasks Celery).
    """
    settings = InstitutionSettings.query.filter_by(
        institution_id=institution_id
    ).first()

    if settings is None:
        return DEFAULT_TIMEOUT_SAME_DAY_MINUTES, DEFAULT_TIMEOUT_DEFAULT_MINUTES

    return settings.timeout_same_day_minutes, settings.timeout_default_minutes


def calculate_timeout(
    institution_id: int,
    scheduled_time: datetime | None,
) -> int:
    """Calcule le timeout en minutes selon l'institution et la date prévue.

    - Si transport prévu aujourd'hui (timezone institution) : same_day timeout
    - Sinon : default timeout
    """
    same_day, default = get_offer_timeouts(institution_id)

    if scheduled_time is None:
        return default

    # Récupérer la timezone de l'institution
    settings = InstitutionSettings.query.filter_by(
        institution_id=institution_id
    ).first()
    tz_name = settings.timezone if settings else DEFAULT_TIMEZONE

    try:
        tz = pytz.timezone(tz_name)
    except pytz.exceptions.UnknownTimeZoneError:
        # Ne devrait jamais arriver si le PUT valide via schema,
        # sauf corruption directe en DB.
        logger.warning(
            "[InstitutionSettings] Timezone invalide '%s' pour institution %s, fallback sur %s",
            tz_name,
            institution_id,
            DEFAULT_TIMEZONE,
        )
        tz = _DEFAULT_TZ

    now_local = datetime.now(tz)
    scheduled_local = scheduled_time.astimezone(tz)

    if now_local.date() == scheduled_local.date():
        return same_day
    return default
