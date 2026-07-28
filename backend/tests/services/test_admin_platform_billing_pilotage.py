"""Tests unitaires — pilotage billing plateforme (args détail entreprise)."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.admin_platform_billing_pilotage import parse_pilotage_detail_args


def test_parse_pilotage_detail_args_excludes_company_id():
    args = MagicMock()
    args.get.side_effect = lambda key, default=None, type_=None: {
        "page": 2,
        "per_page": 40,
        "company_id": None,
        "created_from": None,
        "created_to": None,
        "scheduled_from": None,
        "scheduled_to": None,
        "institution_id": None,
        "institution_q": None,
        "company_q": None,
        "q": None,
        "status": None,
        "cancelled_only": None,
        "exclude_cancelled": None,
        "with_transfer": None,
        "unassigned": None,
        "incomplete_data": None,
        "needs_investigation": None,
    }.get(key, default)

    parsed = parse_pilotage_detail_args(args)

    assert "company_id" not in parsed
    assert parsed["page"] == 2
    assert parsed["per_page"] == 40
