"""✅ Schemas Marshmallow pour validation des endpoints d'authentification."""

from marshmallow import (
    Schema,
    ValidationError,
    fields,
    pre_load,
    validate,
    validates_schema,
)

from schemas.validation_utils import (
    EMAIL_VALIDATOR,
    PASSWORD_VALIDATOR,
    USERNAME_VALIDATOR,
)


class LoginSchema(Schema):
    """Schema pour validation login (POST /api/auth/login)."""

    # Email ou identifiant institution composé (slug/username)
    email = fields.Str(required=True, validate=validate.Length(min=3, max=255))
    password = fields.Str(required=True, validate=PASSWORD_VALIDATOR)
    # Optionnel: si True (côté web), demande un refresh token long-lived (ex: 30j)
    # avec cookie persistant. Si False/absent: refresh token court (ex: 1h) et
    # cookie de session (max_age=None). Ignoré pour les clients mobiles, qui
    # conservent leurs propres TTL.
    remember_me = fields.Bool(load_default=False)


class RegisterSchema(Schema):
    """Schema pour validation inscription (POST /api/auth/register).

    Email et téléphone sont optionnels individuellement, mais au moins l'un
    des deux doit être fourni.
    """

    username = fields.Str(required=True, validate=USERNAME_VALIDATOR)
    email = fields.Email(
        required=False, allow_none=True, load_default=None, validate=EMAIL_VALIDATOR
    )
    # Requis uniquement si email fourni (inscription téléphone seule → OTP)
    password = fields.Str(
        required=False, allow_none=True, load_default=None, validate=PASSWORD_VALIDATOR
    )

    # Champs optionnels
    first_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    last_name = fields.Str(load_default=None, validate=validate.Length(max=100))
    phone = fields.Str(
        required=False,
        allow_none=True,
        load_default=None,
        validate=validate.Length(min=7, max=20),
    )
    address = fields.Str(load_default=None, validate=validate.Length(max=500))
    birth_date = fields.Date(load_default=None)
    gender = fields.Str(
        load_default=None, validate=validate.OneOf(["male", "female", "other"])
    )
    profile_image = fields.Str(load_default=None)

    @pre_load
    def _normalize_optional_contact(self, data, **kwargs):
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in ("email", "phone", "address", "password"):
            if key in out and isinstance(out[key], str) and not out[key].strip():
                out[key] = None
        return out

    @validates_schema
    def _require_email_or_phone(self, data, **kwargs):
        email = (data.get("email") or "").strip() if data.get("email") else ""
        phone = (data.get("phone") or "").strip() if data.get("phone") else ""
        password = (data.get("password") or "").strip() if data.get("password") else ""
        if not email and not phone:
            raise ValidationError(
                "Indiquez une adresse email ou un numéro de téléphone.",
                field_name="email",
            )
        if email and not password:
            raise ValidationError(
                "Le mot de passe est obligatoire lorsque vous indiquez un email.",
                field_name="password",
            )


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
