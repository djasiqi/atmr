"""✅ Schemas Marshmallow pour validation des endpoints admin."""

from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields,
    validate,
)

from models.enums import InstitutionRole
from schemas.query_schemas import (
    DateRangeQuerySchema,
    FilterQuerySchema,
    PaginationQuerySchema,
)


class UserRoleUpdateSchema(Schema):
    """Schema pour mise à jour / preview du rôle (PUT/POST role-transition)."""

    role = fields.Str(
        required=True,
        validate=validate.OneOf(
            ["admin", "client", "driver", "company", "institution"],
            error=(
                "Rôle invalide. Valeurs possibles: admin, client, driver, "
                "company, institution"
            ),
        ),
    )
    reason = fields.Str(
        required=False,
        load_default=None,
        validate=validate.Length(min=5, max=500),
        allow_none=True,
    )
    expected_current_role = fields.Str(
        required=False, load_default=None, allow_none=True
    )
    expected_company_id = fields.Int(
        validate=validate.Range(min=1),
        allow_none=True,
        load_default=None,
    )
    expected_institution_id = fields.Int(
        validate=validate.Range(min=1),
        allow_none=True,
        load_default=None,
    )
    expected_institution_role = fields.Str(
        validate=validate.OneOf(InstitutionRole.choices()),
        allow_none=True,
        load_default=None,
    )
    company_id = fields.Int(
        validate=validate.Range(min=1, error="company_id doit être > 0"),
        allow_none=True,
        load_default=None,
    )
    company_name = fields.Str(
        validate=validate.Length(min=1, max=200), allow_none=True, load_default=None
    )
    institution_id = fields.Int(
        validate=validate.Range(min=1, error="institution_id doit être > 0"),
        allow_none=True,
        load_default=None,
    )
    institution_role = fields.Str(
        validate=validate.OneOf(
            InstitutionRole.choices(),
            error=(
                "Rôle institution invalide. Valeurs: "
                + ", ".join(InstitutionRole.choices())
            ),
        ),
        allow_none=True,
        load_default=None,
    )
    preview_id = fields.Str(allow_none=True, load_default=None)
    transition_id = fields.Str(allow_none=True, load_default=None)


class UserRoleApplySchema(UserRoleUpdateSchema):
    """Apply : reason + expected_current_role obligatoires."""

    reason = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
    )
    expected_current_role = fields.Str(required=True)


class AdminReasonSchema(Schema):
    """Motif admin générique (revoke sessions, etc.)."""

    reason = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
    )


class AdminResetPasswordSchema(AdminReasonSchema):
    """POST /admin/users/<id>/reset-password."""


class AdminDriverStatusSchema(Schema):
    """PUT /admin/users/<id>/driver-status."""

    is_active = fields.Bool(required=True)
    reason = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
    )
    expected_is_active = fields.Bool(required=False, load_default=None, allow_none=True)


class AdminCompanyApprovalSchema(Schema):
    """PUT /admin/companies/<id>/approval."""

    is_approved = fields.Bool(required=True)
    reason = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
    )
    expected_is_approved = fields.Bool(
        required=False, load_default=None, allow_none=True
    )


class AdminCompanyDispatchSchema(Schema):
    """PUT /admin/companies/<id>/dispatch-status."""

    dispatch_enabled = fields.Bool(required=True)
    reason = fields.Str(
        required=True,
        validate=validate.Length(min=5, max=500),
    )
    expected_dispatch_enabled = fields.Bool(
        required=False, load_default=None, allow_none=True
    )


class AutonomousActionReviewSchema(Schema):
    """Schema pour review d'une action autonome
    (POST /api/admin/autonomous-actions/<id>/review)."""

    notes = fields.Str(
        validate=validate.Length(
            max=1000, error="notes doit faire max 1000 caractères"
        ),
        allow_none=True,
        load_default=None,
    )


class AutonomousActionsListQuerySchema(
    PaginationQuerySchema, DateRangeQuerySchema, FilterQuerySchema
):
    """Schema pour validation query params GET /api/admin/autonomous-actions.

    Combine pagination, date range et filtres communs,
    avec filtres spécifiques aux actions autonomes.
    """

    action_type = fields.Str(
        load_default=None,
        validate=validate.Length(
            max=50, error="action_type doit faire max 50 caractères"
        ),
        allow_none=True,
    )
    success = fields.Str(
        load_default=None,
        validate=validate.OneOf(
            ["true", "false", "1", "0", "yes", "no"],
            error="success doit être: true, false, 1, 0, yes ou no",
        ),
        allow_none=True,
    )
    reviewed = fields.Str(
        load_default=None,
        validate=validate.OneOf(
            ["true", "false", "1", "0", "yes", "no"],
            error="reviewed doit être: true, false, 1, 0, yes ou no",
        ),
        allow_none=True,
    )

    class Meta:
        unknown = "INCLUDE"  # Permettre des champs supplémentaires pour compatibilité
