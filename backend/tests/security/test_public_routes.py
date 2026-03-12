"""Test de liste des routes publiques — Plan remédiation LIRIE.

Vérifie qu'aucune route dangereuse n'est exposée.
Si une PR ajoute un endpoint dangereux, le test casse.
"""

import pytest


def test_no_dangerous_public_routes(app):
    """Aucune route interdite ne doit exister dans l'application."""
    routes = [rule.rule for rule in app.url_map.iter_rules()]

    forbidden = [
        "/config",
        "/debug",
        "/internal",
        "/admin/dev",
    ]

    for route in forbidden:
        assert route not in routes, (
            f"Route interdite '{route}' ne doit pas exister. "
            f"Routes similaires: {[r for r in routes if route in r]}"
        )
