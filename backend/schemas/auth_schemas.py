"""✅ Schemas Marshmallow pour validation des endpoints d'authentification."""

from marshmallow import Schema, fields, validate

from schemas.validation_utils import (
    EMAIL_VALIDATOR,
    PASSWORD_VALIDATOR,
    USERNAME_VALIDATOR,
)


class LoginSchema(Schema):
    """Schema pour validation login (POST /api/auth/login)."""

    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)


class RegisterSchema(Schema):
    """Schema pour validation inscription (POST /api/auth/register)."""

    username = fields.Str(required=True, validate=USERNAME_VALIDATOR)
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)

    # Champs optionnels
    first_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    last_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    phone = fields.Str(load_default=None, validate=validate.Length(max=20))
    address = fields.Str(load_default=None, validate=validate.Length(max=500))
    birth_date = fields.Date(load_default=None)
    gender = fields.Str(load_default=None, validate=validate.OneOf(["male", "female", "other"]))
    profile_image = fields.Str(load_default=None)


class RefreshTokenSchema(Schema):
    """Schema pour refresh token (POST /api/auth/refresh-token).

    Note: Le refresh token est dans le header Authorization, pas dans le body.
    Ce schema peut être utilisé pour validation future si nécessaire.
    """

    pass  # Pas de body pour refresh token


class ChangePasswordSchema(Schema):
    """Schema pour changement de mot de passe."""

    current_password = fields.Str(required=True)
    new_password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)
    confirm_password = fields.Str(required=True)
