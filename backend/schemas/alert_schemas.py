"""✅ Schemas Marshmallow pour validation des endpoints d'alertes."""

from marshmallow import Schema, fields, validate


class ClearAlertHistorySchema(Schema):
    """Schema pour nettoyage de l'historique des alertes (POST /api/proactive-alerts/clear-history)."""

    booking_id = fields.Str(
        validate=validate.Length(max=100),
        allow_none=True,
        load_default=None,
    )

    class Meta:  # type: ignore
        unknown = "EXCLUDE"  # Rejeter les champs inconnus pour sécurité
