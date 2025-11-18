"""✅ Schemas Marshmallow pour validation des endpoints admin."""

from marshmallow import Schema, fields, validate


class UserRoleUpdateSchema(Schema):
    """Schema pour mise à jour du rôle d'un utilisateur (PUT /api/admin/users/<id>/role)."""

    role = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["admin", "client", "driver", "company"],
            error="Rôle invalide. Valeurs possibles: admin, client, driver, company",
        ),
    )
    company_id = fields.Int(validate=validate.Range(min=1, error="company_id doit être > 0"), allow_none=True)
    company_name = fields.Str(validate=validate.Length(min=1, max=200), allow_none=True)


class AutonomousActionReviewSchema(Schema):
    """Schema pour review d'une action autonome (POST /api/admin/autonomous-actions/<id>/review)."""

    notes = fields.Str(
        validate=validate.Length(max=1000, error="notes doit faire max 1000 caractères"),
        allow_none=True,
        load_default=None,
    )
