"""Compat shim.

Migration physique progressive:
- Le code a été déplacé vers `infrastructure.dispatch.unified_dispatch_engine`.
- Ce module conserve l'API historique (`services.unified_dispatch.engine.*`) pour
  éviter de casser les imports existants (routes, tests, scripts).

Important pour les tests:
- Certains tests patchent `services.unified_dispatch.engine.data.*`.
  On conserve l'attribut `data` pointant vers le module `services.unified_dispatch.data`
  afin que ces patchs continuent d'affecter l'implémentation.
"""

from __future__ import annotations

from infrastructure.dispatch import unified_dispatch_engine as _impl
from services.unified_dispatch import data
from services.unified_dispatch import settings as ud_settings

# Ré-exports explicites (compatibilité API + évite F401/unused-import).
_acquire_day_lock = _impl._acquire_day_lock
_analyze_unassigned_reasons = _impl._analyze_unassigned_reasons
_apply_and_emit = _impl._apply_and_emit
_filter_problem = _impl._filter_problem
_release_day_lock = _impl._release_day_lock
_safe_int = _impl._safe_int
_serialize_assignment = _impl._serialize_assignment
_serialize_booking = _impl._serialize_booking
_serialize_driver = _impl._serialize_driver
_to_date_ymd = _impl._to_date_ymd
run = _impl.run
tracer = _impl.tracer

__all__ = [
    "_acquire_day_lock",
    "_analyze_unassigned_reasons",
    "_apply_and_emit",
    "_filter_problem",
    "_release_day_lock",
    "_safe_int",
    "_serialize_assignment",
    "_serialize_booking",
    "_serialize_driver",
    "_to_date_ymd",
    "data",
    "run",
    "tracer",
    "ud_settings",
]
