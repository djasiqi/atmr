"""Marshmallow schema for validating company cancellation policy JSON."""

from __future__ import annotations

from typing import Any

from marshmallow import (
    Schema,
    ValidationError,
    fields,
    post_load,
    validate,
    validates_schema,
)

from application.bookings.cancellation_rules import CANCELLATION_REASON_LABELS

VALID_STATUSES = {"EN_ROUTE"}


class _TierSchema(Schema):
    id = fields.String(required=True, validate=validate.Length(min=1, max=50))
    type = fields.String(required=True, validate=validate.OneOf(["time", "status"]))
    hours_before = fields.Float(load_default=None)
    status = fields.String(load_default=None)
    percent = fields.Integer(required=True, validate=validate.Range(min=0, max=100))
    label = fields.String(load_default=None, validate=validate.Length(max=80))

    @validates_schema
    def _validate_tier(self, data: dict[str, Any], **_kwargs) -> None:
        if data["type"] == "time":
            if data.get("hours_before") is None or data["hours_before"] <= 0:
                raise ValidationError(
                    "hours_before must be > 0 for time tiers", field_name="hours_before"
                )
        elif data["type"] == "status" and data.get("status") not in VALID_STATUSES:
            raise ValidationError(
                f"status must be one of {VALID_STATUSES}", field_name="status"
            )


class _ReasonOverrideSchema(Schema):
    billable = fields.Boolean(required=True)


class CancellationPolicySchema(Schema):
    enabled = fields.Boolean(required=True)
    basis = fields.String(
        load_default="booking_amount",
        validate=validate.OneOf(["booking_amount"]),
    )
    apply_when_driver_assigned_only = fields.Boolean(load_default=True)
    tiers = fields.List(fields.Nested(_TierSchema), load_default=[])
    min_fee_chf = fields.Float(load_default=0, validate=validate.Range(min=0))
    max_fee_chf = fields.Float(load_default=None, allow_none=True)
    reason_overrides = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(_ReasonOverrideSchema),
        load_default={},
    )

    @validates_schema
    def _validate_policy(self, data: dict[str, Any], **_kwargs) -> None:
        tier_ids = [t["id"] for t in data.get("tiers", [])]
        if len(tier_ids) != len(set(tier_ids)):
            raise ValidationError("Tier IDs must be unique", field_name="tiers")

        max_fee = data.get("max_fee_chf")
        min_fee = data.get("min_fee_chf") or 0
        if max_fee is not None and max_fee < min_fee:
            raise ValidationError(
                "max_fee_chf must be >= min_fee_chf", field_name="max_fee_chf"
            )

        for key in data.get("reason_overrides", {}):
            if key not in CANCELLATION_REASON_LABELS:
                raise ValidationError(
                    f"Unknown reason code: {key}", field_name="reason_overrides"
                )

    @post_load
    def _sort_tiers(self, data: dict[str, Any], **_kwargs) -> dict[str, Any]:
        """Sort time tiers ASC by hours_before (invariant for compute algorithm)."""
        tiers = data.get("tiers", [])
        time_tiers = sorted(
            [t for t in tiers if t["type"] == "time"],
            key=lambda t: t["hours_before"],
        )
        status_tiers = [t for t in tiers if t["type"] != "time"]
        data["tiers"] = time_tiers + status_tiers
        return data
