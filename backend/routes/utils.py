import base64
import io
import re
from datetime import datetime

import qrcode  # pyright: ignore[reportMissingModuleSource]
from flask import jsonify, request  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]
from qrcode.constants import (  # pyright: ignore[reportMissingModuleSource]
    ERROR_CORRECT_L,
)

from shared.error_handlers import APIErrorHandler

utils_ns = Namespace("utils", description="Endpoints utilitaires")

# Constantes pour éviter les valeurs magiques
MIN_PASSWORD_LENGTH = 8
MAX_QR_DATA_LENGTH = 4096
PASSWORD_VALIDATION_MESSAGE = (
    "Le mot de passe doit contenir au moins 12 caractères, "
    "une majuscule, une minuscule, un chiffre et un caractère spécial."
)

# -------------------------
# Helpers internes
# -------------------------


def _qr_png_bytes(data: str) -> bytes:
    qr = qrcode.QRCode(
        version=1,
        error_correction=ERROR_CORRECT_L,  # ✅ constante importée
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    with io.BytesIO() as buf:
        # ✅ Arg positionnel au lieu de format="PNG" pour calmer Pylance
        img.save(buf, "PNG")
        return buf.getvalue()


def generate_qr_code(data: str) -> str:
    """Retourne une image PNG encodée base64 (data URL-ready sans préfixe)."""
    return base64.b64encode(_qr_png_bytes(data)).decode("utf-8")


def is_valid_email(email: str) -> bool:
    return re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email or "") is not None


def is_valid_phone(phone: str) -> bool:
    return re.match(r"^\+?[0-9]+$", phone or "") is not None


def validate_password(password: str) -> bool:
    """✅ S3: Valide un mot de passe selon les critères de sécurité renforcés.

    Utilise PasswordPolicyService pour une validation stricte.

    Critères:
    - Au moins 12 caractères (configurable via MIN_PASSWORD_LENGTH)
    - Au moins une majuscule
    - Au moins une minuscule
    - Au moins un chiffre
    - Au moins un caractère spécial

    Args:
        password: Mot de passe à valider

    Returns:
        True si le mot de passe est valide, False sinon
    """
    try:
        from security.password_policy import PasswordPolicyService

        # Utiliser le service de politique de mot de passe
        PasswordPolicyService.validate_password(
            password, user_id=None, check_history=False
        )
        return True
    except Exception:
        # En cas d'erreur, retourner False (compatibilité avec l'ancienne API)
        return False


def validate_password_or_raise(password: str, _user=None) -> None:
    """✅ S3: Valide un mot de passe et lève une ValueError si invalide.

    Utilisé pour satisfaire Semgrep en validant explicitement avant set_password.
    Utilise PasswordPolicyService pour une validation stricte.

    Args:
        password: Mot de passe à valider
        _user: Utilisateur (optionnel, pour vérification historique si fourni)

    Raises:
        ValueError: Si le mot de passe ne respecte pas les critères de sécurité
    """
    # Importer le service de politique de mot de passe
    try:
        from security.password_policy import (
            PasswordPolicyError,
            PasswordPolicyService,
        )
    except ImportError:
        # Fallback vers l'ancienne validation si le service n'est pas disponible
        if not validate_password(password):
            raise ValueError(PASSWORD_VALIDATION_MESSAGE) from None
        return

    # Utiliser le service de politique de mot de passe
    try:
        user_id = _user.id if _user and hasattr(_user, "id") else None
        PasswordPolicyService.validate_password(
            password, user_id=user_id, check_history=(user_id is not None)
        )
    except PasswordPolicyError as e:
        # Convertir PasswordPolicyError en ValueError
        raise ValueError(str(e)) from e
    except Exception as e:
        # Si l'exception a un attribut message, l'utiliser
        if hasattr(e, "message"):
            raise ValueError(str(e.message)) from e
        # Sinon, fallback vers l'ancienne validation
        if not validate_password(password):
            raise ValueError(PASSWORD_VALIDATION_MESSAGE) from e


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def handle_error(e: Exception):
    error_response, status_code = APIErrorHandler.handle_exception(e, None)
    return jsonify(error_response), status_code


# -------------------------
# API
# -------------------------


@utils_ns.route("/generate_qr")
class GenerateQR(Resource):
    def post(self):
        """Génère un QR code PNG (base64) à partir des données fournies.
        Attendu JSON : { "data": "votre texte ici" }.
        """
        payload = request.get_json(silent=True) or {}
        data = (payload.get("data") or "").strip()
        if not data:
            return APIErrorHandler.handle_validation_error(
                "Aucune donnée fournie.",
                logger_instance=None,
            )
        if len(data) > MAX_QR_DATA_LENGTH:
            return {
                "error": (
                    f"Données trop volumineuses (max {MAX_QR_DATA_LENGTH} caractères)."
                )
            }, 413

        try:
            b64_png = generate_qr_code(data)
            # (optionnel) tu peux ajouter le préfixe
            # "data:image/png;base64," côté client
            return {"qr_code": b64_png}, 200
        except Exception as e:
            return handle_error(e)
