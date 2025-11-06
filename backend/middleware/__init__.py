"""Middleware pour observabilité et métriques."""

from middleware.metrics import prom_middleware

__all__ = ["prom_middleware"]

