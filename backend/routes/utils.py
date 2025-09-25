from flask import jsonify, request
from flask_restx import Namespace, Resource
import qrcode
import io
import base64
import re
from datetime import datetime

# NOTE:
# - Ne PAS réimporter to_utc ici. Si tu en as besoin côté routes, importe depuis shared:
#   from shared.time_utils import to_utc

utils_ns = Namespace('utils', description="Endpoints utilitaires")

# -------------------------
# Helpers internes
# -------------------------
def _qr_png_bytes(data: str) -> bytes:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        return buf.getvalue()

def generate_qr_code(data: str) -> str:
    """Retourne une image PNG encodée base64 (data URL-ready sans préfixe)."""
    return base64.b64encode(_qr_png_bytes(data)).decode("utf-8")

def is_valid_email(email: str) -> bool:
    return re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email or "") is not None

def is_valid_phone(phone: str) -> bool:
    return re.match(r"^\+?[0-9]+$", phone or "") is not None

def validate_password(password: str) -> bool:
    return isinstance(password, str) and len(password) >= 8

def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def handle_error(e: Exception):
    return jsonify({"error": str(e)}), 500

# -------------------------
# API
# -------------------------
@utils_ns.route('/generate_qr')
class GenerateQR(Resource):
    def post(self):
        """
        Génère un QR code PNG (base64) à partir des données fournies.
        Attendu JSON : { "data": "votre texte ici" }
        """
        payload = request.get_json(silent=True) or {}
        data = (payload.get("data") or "").strip()
        if not data:
            return {"error": "Aucune donnée fournie."}, 400
        if len(data) > 4096:
            return {"error": "Données trop volumineuses (max 4096 caractères)."}, 413

        try:
            b64_png = generate_qr_code(data)
            return {"qr_code": b64_png}, 200
        except Exception as e:
            return handle_error(e)
