from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
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
ALLOWED_AUDIO_EXT = {"m4a", "mp3", "wav", "aac", "caf", "3gp", "webm", "ogg"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_PDF_EXT | ALLOWED_AUDIO_EXT
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/gif"}
ALLOWED_PDF_MIME = {"application/pdf"}
ALLOWED_AUDIO_MIME = {
    "audio/m4a",
    "audio/mp4",
    "audio/x-m4a",
    "audio/mp4a-latm",
    "audio/aac",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/x-caf",
    "audio/3gpp",
    "audio/3gpp2",
    "audio/webm",
    "audio/ogg",
}
ALLOWED_MIME = ALLOWED_IMAGE_MIME | ALLOWED_PDF_MIME | ALLOWED_AUDIO_MIME

_AUDIO_MIME_BY_EXT = {
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "caf": "audio/x-caf",
    "3gp": "audio/3gpp",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
}
_IMAGE_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}
_PDF_MIME_BY_EXT = {"pdf": "application/pdf"}


def _infer_mime_from_filename(filename: str) -> str | None:
    """Déduit un MIME autorisé depuis l'extension (React Native envoie souvent un MIME vide)."""
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in _AUDIO_MIME_BY_EXT:
        return _AUDIO_MIME_BY_EXT[ext]
    if ext in _IMAGE_MIME_BY_EXT:
        return _IMAGE_MIME_BY_EXT[ext]
    if ext in _PDF_MIME_BY_EXT:
        return _PDF_MIME_BY_EXT[ext]
    return None


def _resolve_upload_mime(file, filename: str) -> str:
    raw = (getattr(file, "content_type", None) or "").strip().lower()
    if raw in ALLOWED_MIME:
        return raw
    # Clients mobiles : content-type vide / octet-stream / generic
    if raw in ("", "application/octet-stream", "binary/octet-stream"):
        inferred = _infer_mime_from_filename(filename)
        if inferred:
            return inferred
    inferred = _infer_mime_from_filename(filename)
    if inferred and (not raw or raw.startswith("audio/") or raw.startswith("image/")):
        return inferred
    return raw
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


def _is_audio(filename: str) -> bool:
    """Vérifie si le fichier est un message vocal."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXT


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

    # Validation MIME type (inférence extension si le client n'envoie pas de Content-Type)
    mime_type = _resolve_upload_mime(file, filename)
    if mime_type not in ALLOWED_MIME:
        return (
            {
                "error": (
                    f"Type MIME non autorisé: {mime_type or '(vide)'}. "
                    f"Autorisés: {', '.join(sorted(ALLOWED_MIME))}."
                )
            },
            400,
        )

    # Validation type de fichier
    is_image_file = _is_image(filename) and mime_type in ALLOWED_IMAGE_MIME
    is_pdf_file = _is_pdf(filename) and mime_type in ALLOWED_PDF_MIME
    is_audio_file = _is_audio(filename) and mime_type in ALLOWED_AUDIO_MIME

    if not (is_image_file or is_pdf_file or is_audio_file):
        return (
            {"error": ("Type de fichier non reconnu (image, PDF ou message vocal).")},
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
                    limit = max(1, min(100, int(request.args.get("limit", 20))))
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
                            if dt_before.tzinfo is None:
                                dt_before = dt_before.replace(tzinfo=UTC)
                            now_utc = datetime.now(UTC)
                            if dt_before < now_utc - timedelta(days=365):
                                result = {
                                    "error": "Timestamp trop ancien (max 365 jours)"
                                }
                                status_code = 400
                            elif dt_before > now_utc + timedelta(days=1):
                                result = {"error": "Timestamp futur invalide"}
                                status_code = 400
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


@messages_ns.route("/<int:message_id>/attachment")
class MessageAttachmentDownload(Resource):
    """Téléchargement pièce jointe message via lookup DB (Lot 0 SEC-06)."""

    @jwt_required()
    def get(self, message_id: int):
        from werkzeug.exceptions import NotFound as WzNotFound

        from models import Conversation, Message, User
        from services.messaging.permission_service import MessagingPermissionService
        from shared.upload_path_resolver import serve_stored_upload

        user = User.query.filter_by(public_id=get_jwt_identity()).first()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404

        message = Message.query.get(message_id)
        if not message:
            return {"error": "Message introuvable"}, 404

        conversation = None
        conversation_id = getattr(message, "conversation_id", None)
        if conversation_id:
            conversation = Conversation.query.get(conversation_id)

        participant = None
        if conversation:
            participant = MessagingPermissionService.participant_for(
                conversation.id, user.id
            )

        if not MessagingPermissionService.can_read_message(
            user, message, conversation, participant
        ):
            return {"error": "Accès non autorisé"}, 403

        stored_url = getattr(message, "pdf_url", None) or getattr(
            message, "image_url", None
        )
        if not stored_url:
            return {"error": "Aucune pièce jointe"}, 404

        try:
            return serve_stored_upload(stored_url)
        except WzNotFound:
            return {"error": "Fichier introuvable"}, 404


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
        mime_type = _resolve_upload_mime(file, filename)
        is_image_file = _is_image(filename) and mime_type in ALLOWED_IMAGE_MIME
        is_pdf_file = _is_pdf(filename) and mime_type in ALLOWED_PDF_MIME
        is_audio_file = _is_audio(filename) and mime_type in ALLOWED_AUDIO_MIME

        # Créer le dossier de stockage
        from shared.upload_write import ensure_writable_dir, write_upload_bytes

        upload_root = current_app.config.get(
            "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
        )
        chat_dir = Path(upload_root) / "chat"
        try:
            ensure_writable_dir(chat_dir)
        except OSError:
            chat_dir.mkdir(parents=True, exist_ok=True)

        # Générer un nom de fichier unique (timestamp + nom original sécurisé)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        ext = (file.filename or "").rsplit(".", 1)[1].lower()
        safe_name = secure_filename(file.filename or "file")
        base_name = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
        fname = f"{timestamp}_{base_name}.{ext}"
        fpath = chat_dir / fname

        # Sauvegarder le fichier (best-effort permissions volumes Docker)
        try:
            write_upload_bytes(fpath, file_bytes)
        except PermissionError:
            logger.exception("Upload chat: permission denied path=%s", fpath)
            return {
                "error": (
                    "Impossible d'enregistrer le fichier (permissions uploads). "
                    "Contactez l'administrateur."
                )
            }, 500

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
        elif is_audio_file:
            response["file_type"] = "audio"

        logger.info(
            "📎 Fichier uploadé: %s (%s bytes) -> %s par user %s",
            file.filename,
            size_bytes,
            public_url,
            user_public_id,
        )

        return response, 200
