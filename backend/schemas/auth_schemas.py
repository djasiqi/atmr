"""✅ Schemas Marshmallow pour validation des endpoints d'authentification."""

from marshmallow import (
    Schema,
    fields,
    validate,
)

from schemas.validation_utils import (
    EMAIL_VALIDATOR,
    PASSWORD_VALIDATOR,
    USERNAME_VALIDATOR,
)


class LoginSchema(Schema):
    """Schema pour validation login (POST /api/auth/login)."""

    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)
    # Optionnel: si True (côté web), demande un refresh token long-lived (ex: 30j)
    # avec cookie persistant. Si False/absent: refresh token court (ex: 1h) et
    # cookie de session (max_age=None). Ignoré pour les clients mobiles, qui
    # conservent leurs propres TTL.
    remember_me = fields.Bool(load_default=False)


class RegisterSchema(Schema):
    """Schema pour validation inscription (POST /api/auth/register)."""

    username = fields.Str(required=True, validate=USERNAME_VALIDATOR)
    email = fields.Email(required=True, validate=EMAIL_VALIDATOR)
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)

    # Champs optionnels
    first_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    last_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    phone = fields.Str(required=True, validate=validate.Length(min=7, max=20))
    address = fields.Str(load_default=None, validate=validate.Length(max=500))
    birth_date = fields.Date(load_default=None)
    gender = fields.Str(
        load_default=None, validate=validate.OneOf(["male", "female", "other"])
    )
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


class VerifyEmailActivationSchema(Schema):
    token = fields.Str(required=True, validate=validate.Length(min=16, max=2048))


class VerifySmsActivationSchema(Schema):
    activation_session_id = fields.Str(
        required=True, validate=validate.Length(min=8, max=64)
    )
    code = fields.Str(required=True, validate=validate.Regexp(r"^\d{6}$"))


class FinalizeActivationSchema(Schema):
    activation_session_id = fields.Str(
        required=True, validate=validate.Length(min=8, max=64)
    )


class ResendActivationSchema(Schema):
    activation_session_id = fields.Str(
        required=True, validate=validate.Length(min=8, max=64)
    )


class UpdateActivationPhoneSchema(Schema):
    activation_session_id = fields.Str(
        required=True, validate=validate.Length(min=8, max=64)
    )
    phone = fields.Str(required=True, validate=validate.Length(min=7, max=20))


class PasswordlessOtpRequestSchema(Schema):
    channel = fields.Str(required=True, validate=validate.OneOf(["email", "phone"]))
    identifier = fields.Str(required=True, validate=validate.Length(min=3, max=255))


class PasswordlessOtpVerifySchema(Schema):
    otp_session_id = fields.Str(required=True, validate=validate.Length(min=8, max=128))
    code = fields.Str(required=True, validate=validate.Regexp(r"^\d{6}$"))
