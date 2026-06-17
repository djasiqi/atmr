"""Compatibilité sérialisation : propriété dict + appel legacy ``.serialize()``."""

from __future__ import annotations

from typing import Any


class SerializeResult(dict[str, Any]):
    """Dict API tolérant un appel ``()`` superflu (scripts / docs legacy).

    ``Booking.serialize`` est une propriété qui renvoie un dict. Certains scripts
    ou outils appellent encore ``booking.serialize()`` comme une méthode, ce qui
    provoquait ``TypeError: 'dict' object is not callable``.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> SerializeResult:
        if args or kwargs:
            msg = "serialize() n'accepte aucun argument."
            raise TypeError(msg)
        return self


def as_serialize_result(payload: dict[str, Any]) -> SerializeResult:
    """Enveloppe un dict de sérialisation pour compatibilité property + ()."""
    return SerializeResult(payload)
