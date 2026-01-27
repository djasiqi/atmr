"""✅ Routes API pour TransportVoucher (P3) - Bons de transport."""

import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app, request
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename

from ext import db, role_required
from models import BillingParty, Booking, Client, TransportVoucher, TransportVoucherFile
from models.enums import TransportVoucherStatus, TransportVoucherType, UserRole
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from routes.companies import get_company_from_token
from schemas.transport_voucher_schemas import (
    TransportVoucherCreateSchema,
    TransportVoucherRejectSchema,
    TransportVoucherUpdateSchema,
    TransportVoucherValidateSchema,
)
from schemas.validation_utils import handle_validation_error, validate_request
from services.documents.clamav import scan_bytes
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import get_current_user_via_use_case
from shared.time_utils import now_utc, to_utc

logger = logging.getLogger(__name__)

# Constantes pour l'upload de fichiers
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp", "pdf"}
ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "image/gif",
    "application/pdf",
}
MAX_FILE_SIZE_MB = 10  # 10 Mo max par fichier
MAX_FILES_PER_VOUCHER = 10  # Limite: 10 fichiers par bon


def _allowed_file(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXT

transport_vouchers_ns = Namespace(
    "transport-vouchers",
    description="Gestion des bons de transport (justificatifs facturation)",
)

# ✅ Modèles d'erreur standardisés
api_error_model = create_api_error_model(transport_vouchers_ns)
validation_error_model = create_validation_error_model(transport_vouchers_ns)
not_found_error_model = create_not_found_error_model(transport_vouchers_ns)
permission_error_model = create_permission_error_model(transport_vouchers_ns)

# Modèles Swagger pour documentation
transport_voucher_model = transport_vouchers_ns.model(
    "TransportVoucher",
    {
        "id": fields.Integer(description="ID du bon"),
        "company_id": fields.Integer(description="ID de l'entreprise"),
        "client_id": fields.Integer(description="ID du client"),
        "booking_id": fields.Integer(description="ID de la course (optionnel)"),
        "billing_party_id": fields.Integer(description="ID du payeur (optionnel)"),
        "type": fields.String(description="Type de bon", enum=["clinic", "insurance", "other"]),
        "status": fields.String(
            description="Statut",
            enum=["draft", "submitted", "validated", "rejected", "expired"],
        ),
        "valid_from": fields.DateTime(description="Date de début de validité"),
        "valid_to": fields.DateTime(description="Date de fin de validité"),
        "external_ref": fields.String(description="Référence externe (n° dossier)"),
        "notes": fields.String(description="Notes"),
        "validated_by_user_id": fields.Integer(description="ID utilisateur validateur"),
        "validated_at": fields.DateTime(description="Date de validation"),
        "created_by_user_id": fields.Integer(description="ID utilisateur créateur"),
        "created_at": fields.DateTime(description="Date de création"),
        "updated_at": fields.DateTime(description="Date de mise à jour"),
    },
)


def _serialize_transport_voucher(voucher: TransportVoucher) -> dict[str, Any]:
    """Sérialise un TransportVoucher en dictionnaire."""
    billing_party = None
    if voucher.billing_party_id:
        billing_party = BillingParty.query.filter_by(id=voucher.billing_party_id).first()

    return {
        "id": voucher.id,
        "company_id": voucher.company_id,
        "client_id": voucher.client_id,
        "booking_id": voucher.booking_id,
        "billing_party_id": voucher.billing_party_id,
        "billing_party_name": billing_party.legal_name if billing_party else None,
        "type": voucher.type.value if hasattr(voucher.type, "value") else str(voucher.type),
        "status": voucher.status.value if hasattr(voucher.status, "value") else str(voucher.status),
        "valid_from": voucher.valid_from.isoformat() if voucher.valid_from else None,
        "valid_to": voucher.valid_to.isoformat() if voucher.valid_to else None,
        "external_ref": voucher.external_ref,
        "notes": voucher.notes,
        "validated_by_user_id": voucher.validated_by_user_id,
        "validated_at": voucher.validated_at.isoformat() if voucher.validated_at else None,
        "created_by_user_id": voucher.created_by_user_id,
        "created_at": voucher.created_at.isoformat() if voucher.created_at else None,
        "updated_at": voucher.updated_at.isoformat() if voucher.updated_at else None,
        "files": [
            {
                "id": f.id,
                "file_url": f.file_url,
                "filename": f.filename,
                "mime_type": f.mime_type,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in voucher.files
        ],
    }


@transport_vouchers_ns.route("")
class TransportVouchersList(Resource):
    """Liste et création de bons de transport."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    @transport_vouchers_ns.doc(
        params={
            "client_id": "Filtrer par client",
            "booking_id": "Filtrer par course",
            "status": "Filtrer par statut (draft, submitted, validated, rejected, expired)",
            "type": "Filtrer par type (clinic, insurance, other)",
        }
    )
    def get(self):
        """Lister les bons de transport avec filtres."""
        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        # Filtres
        client_id = request.args.get("client_id", type=int)
        booking_id = request.args.get("booking_id", type=int)
        status = request.args.get("status")
        voucher_type = request.args.get("type")

        query = TransportVoucher.query.filter_by(company_id=company.id)

        if client_id:
            query = query.filter_by(client_id=client_id)
        if booking_id:
            query = query.filter_by(booking_id=booking_id)
        if status:
            query = query.filter_by(status=status)
        if voucher_type:
            query = query.filter_by(type=voucher_type)

        # Eager loading des fichiers pour éviter N+1 queries
        vouchers = (
            query.options(joinedload(TransportVoucher.files))
            .order_by(TransportVoucher.created_at.desc())
            .all()
        )

        return {
            "success": True,
            "data": [_serialize_transport_voucher(v) for v in vouchers],
        }, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    @transport_vouchers_ns.expect(transport_voucher_model)
    @transport_vouchers_ns.response(201, "Bon créé avec succès", transport_voucher_model)
    @transport_vouchers_ns.response(400, "Erreur de validation", validation_error_model)
    def post(self):  # noqa: PLR0911
        """Créer un bon de transport."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        data = request.get_json() or {}
        try:
            validated = validate_request(TransportVoucherCreateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Vérifier que le client appartient à l'entreprise
        client_id = int(validated["client_id"])
        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        # Vérifier booking si fourni
        booking_id = validated.get("booking_id")
        if booking_id:
            booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
            if not booking:
                return APIErrorHandler.handle_not_found("Booking", booking_id, logger)

        # Vérifier billing_party si fourni
        billing_party_id = validated.get("billing_party_id")
        if billing_party_id:
            billing_party = BillingParty.query.filter_by(
                id=billing_party_id, company_id=company.id
            ).first()
            if not billing_party:
                return APIErrorHandler.handle_not_found("BillingParty", billing_party_id, logger)

        user = get_current_user_via_use_case()

        voucher = TransportVoucher()
        voucher.company_id = company.id
        voucher.client_id = client_id
        voucher.booking_id = booking_id
        voucher.billing_party_id = billing_party_id
        voucher.type = TransportVoucherType(validated["type"])
        voucher.status = TransportVoucherStatus(validated.get("status", "draft"))
        voucher.valid_from = to_utc(validated.get("valid_from"))
        voucher.valid_to = to_utc(validated.get("valid_to"))
        voucher.external_ref = validated.get("external_ref")
        voucher.notes = validated.get("notes")
        voucher.created_by_user_id = getattr(user, "id", None) if user else None
        voucher.created_at = now_utc()
        voucher.updated_at = now_utc()

        try:
            db.session.add(voucher)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur création TransportVoucher: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_transport_voucher(voucher)}, 201


@transport_vouchers_ns.route("/<int:voucher_id>")
class TransportVoucherById(Resource):
    """Opérations sur un bon spécifique."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, voucher_id: int):
        """Récupérer un bon par son ID."""
        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = (
            TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id)
            .options(joinedload(TransportVoucher.files))
            .first()
        )
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        return {"success": True, "data": _serialize_transport_voucher(voucher)}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def patch(self, voucher_id: int):  # noqa: PLR0911
        """Modifier un bon."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        data = request.get_json() or {}
        try:
            validated = validate_request(TransportVoucherUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Mise à jour des champs
        if "booking_id" in validated:
            booking_id = validated.get("booking_id")
            if booking_id:
                booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
                if not booking:
                    return APIErrorHandler.handle_not_found("Booking", booking_id, logger)
            voucher.booking_id = booking_id

        if "billing_party_id" in validated:
            billing_party_id = validated.get("billing_party_id")
            if billing_party_id:
                billing_party = BillingParty.query.filter_by(
                    id=billing_party_id, company_id=company.id
                ).first()
                if not billing_party:
                    return APIErrorHandler.handle_not_found("BillingParty", billing_party_id, logger)
            voucher.billing_party_id = billing_party_id

        if validated.get("type"):
            voucher.type = TransportVoucherType(validated["type"])

        if validated.get("status"):
            voucher.status = TransportVoucherStatus(validated["status"])

        if "valid_from" in validated:
            voucher.valid_from = to_utc(validated.get("valid_from"))

        if "valid_to" in validated:
            voucher.valid_to = to_utc(validated.get("valid_to"))

        if "external_ref" in validated:
            voucher.external_ref = validated.get("external_ref")

        if "notes" in validated:
            voucher.notes = validated.get("notes")

        voucher.updated_at = now_utc()

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur update TransportVoucher: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_transport_voucher(voucher)}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def delete(self, voucher_id: int):
        """Supprimer un bon (uniquement si draft)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        # Ne permettre la suppression que si draft
        if voucher.status != TransportVoucherStatus.DRAFT:
            return APIErrorHandler.handle_validation_error(
                "Impossible de supprimer un bon non-draft. Utilisez 'reject' pour rejeter.",
                logger_instance=logger,
            )

        try:
            db.session.delete(voucher)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur delete TransportVoucher: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "message": "Bon supprimé avec succès"}, 200


@transport_vouchers_ns.route("/<int:voucher_id>/validate")
class TransportVoucherValidate(Resource):
    """Validation d'un bon (backoffice)."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    @transport_vouchers_ns.expect(
        transport_vouchers_ns.model(
            "ValidateRequest",
            {
                "billing_party_id": fields.Integer(description="ID du payeur (optionnel)"),
                "notes": fields.String(description="Notes de validation"),
            },
        )
    )
    def post(self, voucher_id: int):  # noqa: PLR0911
        """Valider un bon (passe en status=validated)."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        # Vérifier que le bon peut être validé
        if voucher.status == TransportVoucherStatus.VALIDATED:
            return APIErrorHandler.handle_validation_error(
                "Le bon est déjà validé",
                logger_instance=logger,
            )

        if voucher.status == TransportVoucherStatus.REJECTED:
            return APIErrorHandler.handle_validation_error(
                "Impossible de valider un bon rejeté",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(TransportVoucherValidateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Vérifier billing_party si fourni
        billing_party_id = validated.get("billing_party_id")
        if billing_party_id:
            billing_party = BillingParty.query.filter_by(
                id=billing_party_id, company_id=company.id
            ).first()
            if not billing_party:
                return APIErrorHandler.handle_not_found("BillingParty", billing_party_id, logger)
            voucher.billing_party_id = billing_party_id

        user = get_current_user_via_use_case()

        # Mettre à jour le statut
        voucher.status = TransportVoucherStatus.VALIDATED
        voucher.validated_by_user_id = getattr(user, "id", None) if user else None
        voucher.validated_at = now_utc()
        if validated.get("notes"):
            voucher.notes = (voucher.notes or "") + f"\n[Validation] {validated['notes']}"
        voucher.updated_at = now_utc()

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur validation TransportVoucher: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_transport_voucher(voucher)}, 200


@transport_vouchers_ns.route("/<int:voucher_id>/reject")
class TransportVoucherReject(Resource):
    """Rejet d'un bon (backoffice)."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    @transport_vouchers_ns.expect(
        transport_vouchers_ns.model(
            "RejectRequest",
            {
                "reason": fields.String(required=True, description="Raison du rejet"),
                "notes": fields.String(description="Notes supplémentaires"),
            },
        )
    )
    def post(self, voucher_id: int):
        """Rejeter un bon (passe en status=rejected)."""
        from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        data = request.get_json() or {}
        try:
            validated = validate_request(TransportVoucherRejectSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Mettre à jour le statut
        voucher.status = TransportVoucherStatus.REJECTED
        reason = validated["reason"]
        notes = validated.get("notes")
        rejection_note = f"[Rejet] {reason}"
        if notes:
            rejection_note += f"\n{notes}"
        voucher.notes = (voucher.notes or "") + f"\n{rejection_note}"
        voucher.updated_at = now_utc()

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur rejet TransportVoucher: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_transport_voucher(voucher)}, 200


@transport_vouchers_ns.route("/<int:voucher_id>/files")
class TransportVoucherFiles(Resource):
    """Gestion des fichiers attachés à un bon."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, voucher_id: int):  # noqa: PLR0911  # Many returns due to validation checks
        """Upload un fichier pour un bon de transport."""
        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        # Validation fichiers
        files = request.files.getlist("file")
        if len(files) > MAX_FILES_PER_VOUCHER:
            return APIErrorHandler.handle_validation_error(
                f"Trop de fichiers. Maximum {MAX_FILES_PER_VOUCHER} fichier(s) par bon.",
                logger_instance=logger,
            )

        if not files or not files[0] or not files[0].filename:
            return APIErrorHandler.handle_validation_error(
                "Aucun fichier fourni. Le champ doit s'appeler 'file'.",
                logger_instance=logger,
            )

        file = files[0]
        filename = file.filename or ""

        # Validation extension
        if not _allowed_file(filename):
            return APIErrorHandler.handle_validation_error(
                f"Extension non autorisée. Autorisées: {', '.join(sorted(ALLOWED_EXT))}.",
                logger_instance=logger,
            )

        # Lire le fichier
        file.stream.seek(0)
        file_bytes = file.read()
        file.stream.seek(0)
        size_bytes = len(file_bytes)

        # Validation taille
        if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
            return APIErrorHandler.handle_validation_error(
                f"Fichier trop volumineux (max {MAX_FILE_SIZE_MB} Mo).",
                logger_instance=logger,
            )

        # Validation MIME type
        mime_type = file.content_type or ""
        if mime_type and mime_type not in ALLOWED_MIME:
            return APIErrorHandler.handle_validation_error(
                f"Type MIME non autorisé: {mime_type}. Autorisés: {', '.join(sorted(ALLOWED_MIME))}.",
                logger_instance=logger,
            )

        # Scan antivirus ClamAV
        is_safe, error_msg = scan_bytes(file_bytes)
        if not is_safe:
            logger.warning("🦠 Fichier rejeté par ClamAV: %s - %s", filename, error_msg)
            return APIErrorHandler.handle_validation_error(
                error_msg or "Fichier infecté - upload refusé",
                logger_instance=logger,
            )

        # Créer le dossier de stockage
        upload_root = current_app.config.get(
            "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
        )
        vouchers_dir = Path(upload_root) / "transport_vouchers"
        vouchers_dir.mkdir(parents=True, exist_ok=True)

        # Générer un nom de fichier unique
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        ext = filename.rsplit(".", 1)[1].lower()
        safe_name = secure_filename(filename)
        base_name = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
        fname = f"voucher_{voucher_id}_{timestamp}_{base_name}.{ext}"
        fpath = vouchers_dir / fname

        # Sauvegarder le fichier
        file.save(fpath)

        # Construire l'URL publique
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        public_url = f"{public_base}/transport_vouchers/{fname}"

        # Créer l'entrée en base
        voucher_file = TransportVoucherFile()
        voucher_file.voucher_id = voucher.id
        voucher_file.file_url = public_url
        voucher_file.filename = filename
        voucher_file.mime_type = mime_type
        voucher_file.created_at = now_utc()

        try:
            db.session.add(voucher_file)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            # Supprimer le fichier en cas d'erreur
            with contextlib.suppress(Exception):
                fpath.unlink()
            logger.exception("Erreur création TransportVoucherFile: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        logger.info(
            "📎 Fichier uploadé pour bon %s: %s (%s bytes) -> %s",
            voucher_id,
            filename,
            size_bytes,
            public_url,
        )

        return {
            "success": True,
            "data": {
                "id": voucher_file.id,
                "file_url": voucher_file.file_url,
                "filename": voucher_file.filename,
                "mime_type": voucher_file.mime_type,
                "created_at": voucher_file.created_at.isoformat() if voucher_file.created_at else None,
            },
        }, 201

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def delete(self, voucher_id: int):  # noqa: PLR0911
        """Supprime un fichier attaché à un bon."""
        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        voucher = TransportVoucher.query.filter_by(id=voucher_id, company_id=company.id).first()
        if not voucher:
            return APIErrorHandler.handle_not_found("TransportVoucher", voucher_id, logger)

        file_id = request.args.get("file_id", type=int)
        if not file_id:
            return APIErrorHandler.handle_validation_error(
                "file_id est requis",
                logger_instance=logger,
            )

        voucher_file = TransportVoucherFile.query.filter_by(
            id=file_id, voucher_id=voucher.id
        ).first()
        if not voucher_file:
            return APIErrorHandler.handle_not_found("TransportVoucherFile", file_id, logger)

        # Supprimer le fichier physique
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(voucher_file.file_url)
            file_path = Path(current_app.config.get("UPLOADS_DIR", "/app/uploads")) / parsed_url.path.lstrip("/")
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning("Impossible de supprimer le fichier physique: %s", e)

        try:
            db.session.delete(voucher_file)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur suppression TransportVoucherFile: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "message": "Fichier supprimé avec succès"}, 200
