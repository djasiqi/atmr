"""Droits du module temps de travail.

Le control plane entreprise est encore en mode shadow : ces clés sont prêtes
à être appliquées, mais aujourd'hui seul le rôle ``company`` est exigé.
L'isolation réelle reste ``company_id``.
"""

from __future__ import annotations

from models.enums import UserRole

WORK_TIME_VIEW = "company.work_time.view"
WORK_TIME_MANAGE = "company.work_time.manage"
WORK_TIME_CONFIGURE = "company.work_time.configure"


def _is_company(role: object) -> bool:
    value = getattr(role, "value", role)
    return str(value) == UserRole.company.value


def can_view(role: object) -> bool:
    """Lecture du rapport. Clé future : ``company.work_time.view``."""
    return _is_company(role)


def can_manage(role: object) -> bool:
    """Corrections, saisies, clôture. Clé future : ``company.work_time.manage``."""
    return _is_company(role)


def can_configure(role: object) -> bool:
    """Politiques de rémunération. Clé future : ``company.work_time.configure``."""
    return _is_company(role)
