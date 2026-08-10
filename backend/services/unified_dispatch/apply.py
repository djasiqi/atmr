"""Alias de compatibilité vers ``optimization.assignment_applier``.

Réexporte l'intégralité du module pour que les patches de tests sur
``services.unified_dispatch.apply.*`` ciblent le même namespace que le code.
"""

from __future__ import annotations

from services.unified_dispatch.optimization.assignment_applier import *  # noqa: F401,F403
