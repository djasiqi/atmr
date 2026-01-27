"""Schemas Marshmallow pour le contrôle facturation (P5)."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    ValidationError,
    fields,
    validate,
    validates_schema,
)


class BillingReviewListQuerySchema(Schema):
    """Schema pour valider les paramètres de requête GET /billing/monthly-review."""

    company_id = fields.Integer(required=True, validate=validate.Range(min=1))
    year = fields.Integer(required=True, validate=validate.Range(min=2000, max=2100))
    month = fields.Integer(
        required=True, validate=validate.Range(min=1, max=12)
    )
    status = fields.String(
        validate=validate.OneOf(["draft", "needs_review", "ready", "locked"]),
        allow_none=True,
    )
    billing_party_id = fields.Integer(validate=validate.Range(min=1), allow_none=True)
    clinic_id = fields.Integer(validate=validate.Range(min=1), allow_none=True)


class SetPayerRequestSchema(Schema):
    """Schema pour valider POST /billing/bookings/{id}/set-payer."""

    billed_to_type = fields.String(
        required=True,
        validate=validate.OneOf(["patient", "company", "billing_party", "clinic"]),
    )
    billing_party_id = fields.Integer(
        validate=validate.Range(min=1), allow_none=True, required=False
    )
    billed_to_company_id = fields.Integer(
        validate=validate.Range(min=1), allow_none=True, required=False
    )
    reason = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        error_messages={"required": "Le motif est obligatoire pour modifier le payeur"},
    )

    @validates_schema
    def validate_payer_fields(self, data, **kwargs):  # noqa: ARG002
        """Valide que les champs appropriés sont fournis selon billed_to_type."""
        billed_to_type = data.get("billed_to_type")
        billing_party_id = data.get("billing_party_id")
        billed_to_company_id = data.get("billed_to_company_id")

        if billed_to_type == "billing_party":
            if not billing_party_id:
                raise ValidationError(
                    "billing_party_id est requis lorsque billed_to_type='billing_party'"
                )
        elif billed_to_type == "company":
            if not billed_to_company_id:
                raise ValidationError(
                    "billed_to_company_id est requis lorsque billed_to_type='company'"
                )
        elif billed_to_type == "clinic":
            # clinic: requis en principe; si manquant, la route peut inférer depuis booking.billed_to_company_id
            pass
        elif billed_to_type == "patient":
            # Pour patient, on peut avoir billing_party_id=None et billed_to_company_id=None
            pass


class LockBookingRequestSchema(Schema):
    """Schema pour valider POST /billing/bookings/{id}/lock."""

    reason = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        error_messages={"required": "Le motif est obligatoire pour verrouiller"},
    )


class UnlockBookingRequestSchema(Schema):
    """Schema pour valider POST /billing/bookings/{id}/unlock."""

    reason = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        error_messages={"required": "Le motif est obligatoire pour déverrouiller"},
    )


class BatchSetPayerRequestSchema(Schema):
    """Schema pour valider POST /billing/bookings/batch-set-payer."""

    booking_ids = fields.List(
        fields.Integer(validate=validate.Range(min=1)),
        required=True,
        validate=validate.Length(min=1, max=100),
        error_messages={
            "required": "Au moins un booking_id est requis",
            "invalid": "booking_ids doit être une liste d'entiers",
        },
    )
    billed_to_type = fields.String(
        required=True,
        validate=validate.OneOf(["patient", "company", "billing_party", "clinic"]),
    )
    billing_party_id = fields.Integer(
        validate=validate.Range(min=1), allow_none=True, required=False
    )
    billed_to_company_id = fields.Integer(
        validate=validate.Range(min=1), allow_none=True, required=False
    )
    reason = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        error_messages={"required": "Le motif est obligatoire pour modifier le payeur"},
    )

    @validates_schema
    def validate_payer_fields(self, data, **kwargs):  # noqa: ARG002
        """Valide que les champs appropriés sont fournis selon billed_to_type."""
        billed_to_type = data.get("billed_to_type")
        billing_party_id = data.get("billing_party_id")
        billed_to_company_id = data.get("billed_to_company_id")

        if billed_to_type == "billing_party":
            if not billing_party_id:
                raise ValidationError(
                    "billing_party_id est requis lorsque billed_to_type='billing_party'"
                )
        elif billed_to_type in ("company", "clinic"):
            if not billed_to_company_id:
                raise ValidationError(
                    "billed_to_company_id est requis lorsque billed_to_type='company' ou 'clinic'"
                )
        elif billed_to_type == "patient":
            # Pour patient, on peut avoir billing_party_id=None et billed_to_company_id=None
            pass
