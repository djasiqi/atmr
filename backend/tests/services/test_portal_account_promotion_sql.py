"""Prédicats AUTH-SMS-02 : preview et migration doivent rester identiques."""

from __future__ import annotations

from services.auth.portal_account_promotion_sql import (
    COUNT_PORTAL_PROMOTION_USERS_SQL,
    LIST_PORTAL_PROMOTION_USERS_SQL,
    PORTAL_PROMOTION_USER_PREDICATE,
    PROMOTE_PORTAL_USERS_SQL,
)


def test_count_list_et_update_partagent_les_predicats():
    assert PORTAL_PROMOTION_USER_PREDICATE in COUNT_PORTAL_PROMOTION_USERS_SQL
    assert PORTAL_PROMOTION_USER_PREDICATE in LIST_PORTAL_PROMOTION_USERS_SQL
    assert PORTAL_PROMOTION_USER_PREDICATE in PROMOTE_PORTAL_USERS_SQL


def test_sql_promotion_ne_lit_ni_necrit_phone_verified_at():
    # Le preview doit tourner avant flask db upgrade (colonne encore absente).
    assert "phone_verified_at" not in COUNT_PORTAL_PROMOTION_USERS_SQL
    assert "phone_verified_at" not in LIST_PORTAL_PROMOTION_USERS_SQL
    assert "phone_verified_at" not in PROMOTE_PORTAL_USERS_SQL
    assert "phone_verified_at" not in PORTAL_PROMOTION_USER_PREDICATE
