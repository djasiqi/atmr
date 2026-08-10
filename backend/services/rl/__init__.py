"""Alias de compatibilité — ancien package ``services.rl`` → ``services.ml.rl``.

Les imports ``import services.rl.<module>`` et les patches de tests
continuent de résoudre vers ``services.ml.rl`` via le ``__path__``.
"""

from __future__ import annotations

import services.ml.rl as _ml_rl

# Délègue la résolution des sous-modules à services.ml.rl
__path__ = list(_ml_rl.__path__)  # type: ignore[misc, name-defined]

__all__: list[str] = []
