"""Fixtures partagées pour les tests de routes."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.routes.admin_route_fixtures"]


@pytest.fixture
def admin_tenant_client(client, admin_headers):
    """Client avec en-têtes admin pour routes /platform/*."""

    class H:
        def __init__(self, c, headers):
            self._c = c
            self._h = headers

        def _merge(self, kw):
            h = dict(self._h)
            h.update(kw.pop("headers", None) or {})
            kw["headers"] = h
            return kw

        def get(self, path, **kw):
            return self._c.get(path, **self._merge(kw))

        def post(self, path, **kw):
            return self._c.post(path, **self._merge(kw))

    return H(client, admin_headers)
