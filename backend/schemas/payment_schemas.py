"""✅ Schemas Marshmallow pour validation des endpoints payments."""

from marshmallow import Schema, fields, validate


class PaymentCreateSchema(Schema):
    """Schema pour création de paiement (POST /api/payments)."""

    amount = fields.Float(
        required=True,
        validate=validate.Range(min=0.01, error="Le montant doit être > 0"),
    )
    method = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=50, error="Méthode de paiement requise"),
    )
    booking_id = fields.Int(validate=validate.Range(min=1), allow_none=True)
    reference = fields.Str(validate=validate.Length(max=100), allow_none=True)


class PaymentStatusUpdateSchema(Schema):
    """Schema pour mise à jour du statut d'un paiement (PUT /api/payments/<id>)."""

    status = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["pending", "completed", "failed"],
            error="Statut invalide. Valeurs possibles: pending, completed, failed",
        ),
    )
