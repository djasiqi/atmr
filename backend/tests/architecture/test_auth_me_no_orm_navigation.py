"""Garde-fou léger : la route UserInfo ne doit pas naviguer sur des relations ORM."""

from __future__ import annotations

import inspect

import routes.auth as auth_module


def test_userinfo_get_source_has_no_user_dot_driver():
    src = inspect.getsource(auth_module.UserInfo.get)
    assert "user.driver" not in src
    assert "user.clients" not in src
    assert "GetBootstrapSessionUseCase" in src
