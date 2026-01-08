from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]
from werkzeug.utils import secure_filename  # pyright: ignore[reportMissingImports]

from models import UserRole
from repositories.company_repository import CompanyRepository
from repositories.message_repository import MessageRepository
from repositories.user_repository import UserRepository
from services.documents.clamav import scan_bytes
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# Initialisation des repositories
user_repo = UserRepository()
message_repo = MessageRepository()
company_repo = CompanyRepository()

messages_ns = Namespace("messages", description="Messagerie entreprise")

# Constantes pour l'upload de fichiers
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_PDF_EXT = {"pdf"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_PDF_EXT
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/gif"}
ALLOWED_PDF_MIME = {"application/pdf"}
ALLOWED_MIME = ALLOWED_IMAGE_MIME | ALLOWED_PDF_MIME
MAX_FILE_SIZE_MB = 10  # 10 Mo max par fichier
MAX_FILES_PER_MESSAGE = 1  # Limite: 1 fichier par message


def _allowed_file(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXT


def _is_image(filename: str) -> bool:
    """Vérifie si le fichier est une image."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXT


def _is_pdf(filename: str) -> bool:
    """Vérifie si le fichier est un PDF."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_PDF_EXT


def _validate_file_upload(
    file, filename: str, file_bytes: bytes
) -> tuple[dict[str, Any] | None, int]:
    """
    Valide un fichier uploadé.
    Retourne (error_dict, status_code) en cas d'erreur, ou (None, 0) si valide.
    """
    # Validation extension
    if not filename or not _allowed_file(filename):
        return (
            {
                "error": (
                    f"Extension non autorisée. "
                    f"Autorisées: {', '.join(sorted(ALLOWED_EXT))}."
                )
            },
            400,
        )

    # Validation taille
    size_bytes = len(file_bytes)
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        error_response, status_code = APIErrorHandler.handle_validation_error(
            f"Fichier trop volumineux (max {MAX_FILE_SIZE_MB} Mo).",
            logger_instance=logger,
        )
        return (error_response, status_code)

    # Validation MIME type
    mime_type = file.content_type or ""
    if mime_type not in ALLOWED_MIME:
        return (
            {
                "error": (
                    f"Type MIME non autorisé: {mime_type}. "
                    f"Autorisés: {', '.join(sorted(ALLOWED_MIME))}."
                )
            },
            400,
        )

    # Validation type de fichier
    is_image_file = _is_image(filename) and mime_type in ALLOWED_IMAGE_MIME
    is_pdf_file = _is_pdf(filename) and mime_type in ALLOWED_PDF_MIME

    if not (is_image_file or is_pdf_file):
        return (
            {"error": "Type de fichier non reconnu (doit être une image ou un PDF)."},
            400,
        )

    # Scan antivirus ClamAV
    is_safe, error_msg = scan_bytes(file_bytes)
    if not is_safe:
        logger.warning("🦠 Fichier rejeté par ClamAV: %s - %s", filename, error_msg)
        error_response, status_code = APIErrorHandler.handle_validation_error(
            error_msg or "Fichier infecté - upload refusé",
            logger_instance=logger,
        )
        return (error_response, status_code)

    return (None, 0)


@messages_ns.route("/<int:company_id>")
class MessagesList(Resource):
    @jwt_required()
    def get(self, company_id: int):
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        user_public_id = get_jwt_identity()

        # 🔍 Chargement de l'utilisateur + relations (avec cast pour Pylance)
        user = user_repo.find_by_public_id_with_driver_and_company(user_public_id)
        if not user:
            logger.error(
                "❌ Utilisateur introuvable pour public_id: %s", user_public_id
            )
            result = {"error": "Utilisateur introuvable"}
            status_code = 404
        else:
            # 🔐 Contrôle d'accès
            if user.role == UserRole.driver:
                if (
                    not getattr(user, "driver", None)
                    or user.driver.company_id != company_id
                ):
                    result = {"error": "Accès refusé au chat de cette entreprise"}
                    status_code = 403
            elif user.role == UserRole.company:
                if not getattr(user, "company", None) or user.company.id != company_id:
                    result = {"error": "Accès refusé à cette entreprise"}
                    status_code = 403
            else:
                result = {"error": "Rôle non autorisé"}
                status_code = 403

            if result is None:
                # 📦 Lecture des params de pagination
                try:
                    limit = max(1, int(request.args.get("limit", 20)))
                    before = request.args.get("before", None)
                except ValueError:
                    result = {"error": "Paramètres invalides"}
                    status_code = 400
                else:
                    # 🔎 Construction de la requête
                    dt_before = None
                    if before:
                        try:
                            # support basique ISO8601 avec 'Z'
                            before_str = before.rstrip("Z")
                            dt_before = datetime.fromisoformat(before_str)
                        except ValueError:
                            result = {"error": "Timestamp invalide"}
                            status_code = 400

                    if result is None:
                        # 🔄 Récupération des messages (avec relations préchargées)
                        messages = (
                            message_repo.find_models_by_company_with_eager_loading(
                                company_id=company_id,
                                before_timestamp=dt_before,
                                limit=limit,
                            )
                        )

                        # ↩️ On remet en ordre ascendant
                        messages.reverse()

                        # Précharger l'entreprise (évite une requête par message)
                        company = company_repo.find_model_by_id(company_id)
                        company_name = (
                            company.name
                            if company and getattr(company, "name", None)
                            else "Entreprise"
                        )

                        # 🔧 Sérialisation (s'aligne sur Message.serialize
                        # pour cohérence API)
                        results: list[dict[str, Any]] = []
                        for m in messages:
                            try:
                                base = m.serialize if hasattr(m, "serialize") else {}
                            except Exception:
                                base = {}
                            if not base:
                                # Fallback minimal si serialize indisponible
                                base = {
                                    "id": m.id,
                                    "company_id": m.company_id,
                                    "sender_id": getattr(m, "sender_id", None),
                                    "receiver_id": getattr(m, "receiver_id", None),
                                    "sender_role": getattr(m, "sender_role", None),
                                    "content": getattr(m, "content", None),
                                    "timestamp": m.timestamp.isoformat()
                                    if getattr(m, "timestamp", None)
                                    else None,
                                }
                                # enrichir noms
                                base["sender_name"] = (
                                    company_name
                                    if getattr(m, "sender_role", None)
                                    in ("COMPANY", "company")
                                    else (
                                        getattr(
                                            getattr(m, "sender", None),
                                            "first_name",
                                            None,
                                        )
                                    )
                                )
                                base["receiver_name"] = getattr(
                                    getattr(m, "receiver", None), "first_name", None
                                )

                            results.append(base)

                        logger.info(
                            "📨 %s messages (limit=%s, before=%s) pour company_id=%s",
                            len(results),
                            limit,
                            before,
                            company_id,
                        )
                        result = results

        return result, status_code


@messages_ns.route("/upload")
class MessageUpload(Resource):
    @jwt_required()
    def post(self):
        """
        Upload d'un fichier (image ou PDF) pour un message de chat.

        Accepte:
        - Images: PNG, JPG, JPEG, GIF, WEBP
        - PDF: PDF

        Retourne:
        - url: URL publique du fichier
        - filename: Nom du fichier
        - size_bytes: Taille en octets
        - file_type: "image" ou "pdf"
        """
        user_public_id = get_jwt_identity()
        user = user_repo.find_by_public_id_first(user_public_id)

        # Validation utilisateur et rôle
        error_response = None
        if not user:
            error_response = ({"error": "Utilisateur introuvable"}, 404)
        elif user.role not in (UserRole.driver, UserRole.company):
            error_response = ({"error": "Rôle non autorisé pour le chat"}, 403)

        if error_response:
            return error_response

        # Validation fichiers
        files = request.files.getlist("file")
        if len(files) > MAX_FILES_PER_MESSAGE:
            return {
                "error": (
                    f"Trop de fichiers. Maximum {MAX_FILES_PER_MESSAGE} "
                    f"fichier(s) par message."
                )
            }, 400

        if not files or not files[0] or not files[0].filename:
            return {
                "error": "Aucun fichier fourni. Le champ doit s'appeler 'file'."
            }, 400

        file = files[0]
        filename = file.filename or ""

        # Lire le fichier
        file.stream.seek(0)
        file_bytes = file.read()
        file.stream.seek(0)
        size_bytes = len(file_bytes)

        # Validation complète du fichier
        error_response, status_code = _validate_file_upload(file, filename, file_bytes)
        if error_response:
            return error_response, status_code

        # Déterminer le type de fichier
        mime_type = file.content_type or ""
        is_image_file = _is_image(filename) and mime_type in ALLOWED_IMAGE_MIME
        is_pdf_file = _is_pdf(filename) and mime_type in ALLOWED_PDF_MIME

        # Créer le dossier de stockage
        upload_root = current_app.config.get(
            "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
        )
        chat_dir = Path(upload_root) / "chat"
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Générer un nom de fichier unique (timestamp + nom original sécurisé)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        ext = (file.filename or "").rsplit(".", 1)[1].lower()
        safe_name = secure_filename(file.filename or "file")
        base_name = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
        fname = f"{timestamp}_{base_name}.{ext}"
        fpath = chat_dir / fname

        # Sauvegarder le fichier
        file.save(fpath)

        # Construire l'URL publique
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        public_url = f"{public_base}/chat/{fname}"

        # Retourner la réponse
        response = {
            "url": public_url,
            "filename": file.filename,
            "size_bytes": size_bytes,
        }

        if is_image_file:
            response["file_type"] = "image"
        elif is_pdf_file:
            response["file_type"] = "pdf"

        logger.info(
            "📎 Fichier uploadé: %s (%s bytes) -> %s par user %s",
            file.filename,
            size_bytes,
            public_url,
            user_public_id,
        )

        return response, 200
