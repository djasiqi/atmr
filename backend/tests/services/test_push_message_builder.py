# backend/tests/services/test_push_message_builder.py
"""Tests unitaires du PushMessageBuilder (messages push métier)."""

from __future__ import annotations

import re

import pytest

from services.notifications.push_message_builder import (
    EVENT_ASSIGNED,
    EVENT_CANCELLED,
    EVENT_COMPLETED,
    EVENT_REASSIGNED,
    EVENT_STATUS_UPDATED,
    build_push_message,
)


# ---------- 1) Assignation ----------


def test_build_push_message_assigned_driver():
    """Assignation chauffeur : titre + body avec nom client + heure + lieu."""
    ctx = {
        "id": 3253,
        "client_name": "Drin Jasiqi",
        "time_formatted": "13:00",
        "dropoff_location": "HUG",
    }
    out = build_push_message(EVENT_ASSIGNED, ctx, "driver", discrete_mode=False)
    assert out["title"] == "Assignation : Nouvelle course assignée"
    assert "Drin Jasiqi" in out["body"]
    assert "13:00" in out["body"]
    assert "HUG" in out["body"]
    assert out["data"]["booking_id"] == 3253
    assert out["data"]["event"] == EVENT_ASSIGNED
    assert out["data"]["client_display_name"] == "Drin Jasiqi"
    assert out["data"]["deep_link"] == "atmr://booking/3253"
    assert out["data"]["deepLink"] == "atmr://booking/3253"


def test_build_push_message_assigned_company():
    """Assignation entreprise : titre + body avec client + deep_link enterprise."""
    ctx = {
        "id": 3253,
        "client_name": "Drin Jasiqi",
        "time_formatted": "13:00",
        "dropoff_location": "HUG",
    }
    out = build_push_message(EVENT_ASSIGNED, ctx, "company", discrete_mode=False)
    assert out["title"] == "Assignation : Nouvelle course assignée"
    assert "Drin Jasiqi" in out["body"]
    assert out["data"]["deep_link"] == "atmr://enterprise/rides/3253"
    assert out["data"]["deepLink"] == "atmr://enterprise/rides/3253"


# ---------- 2) Statut en_route ----------


def test_build_push_message_status_en_route():
    """Statut en_route : « Driss est en route pour Drin Jasiqi • Départ: Ernest-Pictet 9 »."""
    ctx = {
        "id": 3253,
        "client_name": "Drin Jasiqi",
        "pickup_location": "Ernest-Pictet 9",
        "dropoff_location": "HUG",
    }
    actor = {"first_name": "Driss", "last_name": "K.", "username": "drissk"}
    out = build_push_message(
        EVENT_STATUS_UPDATED,
        ctx,
        "company",
        actor=actor,
        status="en_route",
        changes_preview="Départ: Ernest-Pictet 9",
        discrete_mode=False,
    )
    assert out["title"] == "Statut : Mise à jour de course"
    assert "Driss" in out["body"]
    assert "Drin Jasiqi" in out["body"]
    assert "Ernest-Pictet" in out["body"] or "Départ" in out["body"]
    assert out["data"]["status"] == "en_route"
    assert out["data"]["booking_id"] == 3253


def test_build_push_message_status_en_route_actor_unknown():
    """Statut en_route sans acteur → « Un chauffeur est en route pour … »."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi"}
    out = build_push_message(
        EVENT_STATUS_UPDATED,
        ctx,
        "company",
        actor=None,
        status="en_route",
        discrete_mode=False,
    )
    assert "Un chauffeur" in out["body"]
    assert "Drin Jasiqi" in out["body"]


# ---------- 3) Terminé (COMPLETED) ----------


def test_build_push_message_completed():
    """Terminé : « Course terminée • Drin Jasiqi • 45 CHF »."""
    ctx = {
        "id": 3253,
        "client_name": "Drin Jasiqi",
        "amount": 45.0,
    }
    out = build_push_message(EVENT_COMPLETED, ctx, "company", discrete_mode=False)
    assert out["title"] == "Terminé : Course terminée"
    assert "Drin Jasiqi" in out["body"]
    assert "45 CHF" in out["body"]
    assert out["data"]["event"] == EVENT_COMPLETED
    assert out["data"]["booking_id"] == 3253


# ---------- 4) Mode discret (pas de nom) ----------


def test_build_push_message_discrete_assigned():
    """Mode discret assignation : pas de nom client dans le body."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi", "time_formatted": "13:00"}
    out = build_push_message(EVENT_ASSIGNED, ctx, "driver", discrete_mode=True)
    assert out["title"] == "Assignation : Nouvelle course assignée"
    assert "Drin Jasiqi" not in out["body"]
    assert "Nouvelle course assignée" in out["body"] or "Ouvrez" in out["body"]
    assert "client_display_name" not in out["data"] or out["data"].get("client_display_name") is None


def test_build_push_message_discrete_status_updated():
    """Mode discret mise à jour : body générique."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi"}
    out = build_push_message(
        EVENT_STATUS_UPDATED,
        ctx,
        "company",
        status="en_route",
        discrete_mode=True,
    )
    assert "Drin Jasiqi" not in out["body"]
    assert "Mise à jour" in out["body"] or "Ouvrez" in out["body"]


def test_build_push_message_discrete_cancelled():
    """Mode discret annulation : « Une course a été annulée »."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi"}
    out = build_push_message(EVENT_CANCELLED, ctx, "driver", discrete_mode=True)
    assert "Drin Jasiqi" not in out["body"]
    assert "annulée" in out["body"].lower()


def test_build_push_message_discrete_no_client_in_data():
    """P0 mode discret : data ne contient pas client_display_name (lockscreen)."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi", "time_formatted": "13:00"}
    out = build_push_message(EVENT_ASSIGNED, ctx, "company", discrete_mode=True)
    assert "client_display_name" not in out["data"] or out["data"].get("client_display_name") is None
    assert "Drin Jasiqi" not in out["body"]


# ---------- 5) Réassignation ----------


def test_build_push_message_reassigned():
    """Réassignation : deep_link = atmr://bookings."""
    ctx = {"id": 3253, "client_name": "Drin Jasiqi"}
    out = build_push_message(EVENT_REASSIGNED, ctx, "driver", discrete_mode=False)
    assert out["title"] == "Course réassignée"
    assert "Drin Jasiqi" in out["body"]
    assert out["data"]["deep_link"] == "atmr://bookings"
    assert out["data"]["deepLink"] == "atmr://bookings"


# ---------- 6) Booking model-like (dict avec client_name) ----------


def test_build_push_message_accepts_booking_id_key():
    """Accepte booking_id en clé alternative à id dans le dict."""
    ctx = {"booking_id": 999, "client_name": "Test", "dropoff_location": "HUG"}
    out = build_push_message(EVENT_ASSIGNED, ctx, "driver", discrete_mode=False)
    assert out["data"]["booking_id"] == 999
    assert "atmr://booking/999" in (out["data"]["deep_link"], out["data"]["deepLink"])


# ---------- 7) P0 : Jamais de body "ID-only" ----------

# Ancien fallback technique : "Course #123 assignée …" (et variantes Transport #N, Booking N…)
_ID_HASH_PATTERN = re.compile(r"#\d+")
_ID_STANDALONE_PATTERN = re.compile(r"\b\d{3,}\b")  # IDs typiquement 3+ chiffres

# Marqueurs métier acceptables en mode detailed quand pas de client_name (ex. "Client", "Ouvrez", …)
_NON_ID_MARKERS = ("Client", "Ouvrez", "assignée", "Nouvelle course", "application", "détails")


def _body_contains_raw_id(body: str) -> bool:
    """True si le body contient un ID brut (#N ou N avec 3+ chiffres)."""
    return bool(_ID_HASH_PATTERN.search(body) or _ID_STANDALONE_PATTERN.search(body))


def test_build_push_message_never_id_only_body():
    """P0 : le body ne doit jamais contenir d'ID brut (#N ou N 3+ chiffres).

    - discrete_mode=True : body ne doit contenir ni #\\d+ ni \\b\\d{3,}\\b.
    - discrete_mode=False : idem, et si le contexte a client_name alors le body doit
      le contenir ; si contexte minimal, le body doit contenir un marqueur métier non-ID.
    """
    for role in ("driver", "company"):
        for discrete in (True, False):
            ctx = {"id": 3253}  # minimal, pas de client_name
            out = build_push_message(EVENT_ASSIGNED, ctx, role, discrete_mode=discrete)
            assert not _body_contains_raw_id(
                out["body"]
            ), f"Body ne doit pas contenir d'ID brut pour role={role} discrete={discrete}: {out['body']!r}"

    # discrete_mode=False avec client_name : body doit contenir le nom client
    for role in ("driver", "company"):
        ctx = {"id": 3253, "client_name": "Drin Jasiqi", "time_formatted": "13:00"}
        out = build_push_message(EVENT_ASSIGNED, ctx, role, discrete_mode=False)
        assert not _body_contains_raw_id(out["body"]), f"Pas d'ID brut: {out['body']!r}"
        assert "Drin Jasiqi" in out["body"], (
            f"Mode detailed avec client_name : le body doit contenir le nom client pour role={role}: {out['body']!r}"
        )

    # discrete_mode=False avec contexte minimal : body doit contenir un marqueur métier non-ID
    for role in ("driver", "company"):
        ctx = {"id": 3253}
        out = build_push_message(EVENT_ASSIGNED, ctx, role, discrete_mode=False)
        assert not _body_contains_raw_id(out["body"]), f"Pas d'ID brut: {out['body']!r}"
        has_marker = any(m in out["body"] for m in _NON_ID_MARKERS)
        assert has_marker, (
            f"Mode detailed sans client_name : le body doit contenir un marqueur métier (ex. Client, Ouvrez…) pour role={role}: {out['body']!r}"
        )
