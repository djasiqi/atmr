import base64
import io
import re
from datetime import datetime

import qrcode
from flask import jsonify, request
from flask_restx import Namespace, Resource

# ✅ évite "constants is not a known attribute"
from qrcode.constants import ERROR_CORRECT_L

utils_ns = Namespace("utils", description="Endpoints utilitaires")

# Constantes pour éviter les valeurs magiques
MIN_PASSWORD_LENGTH = 8
MAX_QR_DATA_LENGTH = 4096

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
    return len(password) >= MIN_PASSWORD_LENGTH


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def handle_error(e: Exception):
    return jsonify({"error": str(e)}), 500


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
            return {"error": "Aucune donnée fournie."}, 400
        if len(data) > MAX_QR_DATA_LENGTH:
            return {"error": f"Données trop volumineuses (max {MAX_QR_DATA_LENGTH} caractères)."}, 413

        try:
            b64_png = generate_qr_code(data)
            # (optionnel) tu peux ajouter le préfixe "data:image/png;base64," côté client
            return {"qr_code": b64_png}, 200
        except Exception as e:
            return handle_error(e)
