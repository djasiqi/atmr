"""Tests format_user_actor_display_name."""

from types import SimpleNamespace

from shared.user_display import (
    format_user_actor_display_name,
    is_placeholder_actor_display_name,
)


def test_uses_full_name_when_available():
    user = SimpleNamespace(
        id=91597,
        first_name="Anouk",
        last_name="THIERY",
        username="athiery",
        email="anouk@example.com",
    )
    assert format_user_actor_display_name(user=user) == "Anouk THIERY"


def test_uses_username_with_user_id_when_names_missing():
    user = SimpleNamespace(
        id=91597,
        first_name=None,
        last_name="",
        username="athiery",
        email="anouk@example.com",
    )
    assert (
        format_user_actor_display_name(user=user, user_id=91597)
        == "athiery (User #91597)"
    )


def test_uses_email_when_name_and_username_missing():
    user = SimpleNamespace(
        id=12,
        first_name="",
        last_name="",
        username=None,
        email="ops@clinic.ch",
    )
    assert format_user_actor_display_name(user=user) == "ops@clinic.ch"


def test_falls_back_to_user_id_placeholder():
    user = SimpleNamespace(
        id=91597,
        first_name="",
        last_name="",
        username=None,
        email=None,
    )
    assert format_user_actor_display_name(user=user, user_id=91597) == "User #91597"


def test_user_missing_without_db_lookup():
    assert (
        format_user_actor_display_name(
            user_id=91597,
            user=None,
            allow_db_lookup=False,
        )
        == "User #91597"
    )


def test_ignores_placeholder_fallback_when_user_resolvable():
    user = SimpleNamespace(
        id=91597,
        first_name="Anouk",
        last_name="THIERY",
        username="athiery",
        email=None,
    )
    assert (
        format_user_actor_display_name(
            user=user,
            user_id=91597,
            fallback="User #91597",
        )
        == "Anouk THIERY"
    )


def test_placeholder_detection():
    assert is_placeholder_actor_display_name(None) is True
    assert is_placeholder_actor_display_name("") is True
    assert is_placeholder_actor_display_name("User #91597") is True
    assert is_placeholder_actor_display_name("Anouk THIERY") is False
    assert is_placeholder_actor_display_name("athiery (User #91597)") is False
