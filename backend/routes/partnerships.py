# backend/routes/partnerships.py
"""Routes pour les partenariats (endpoints simplifiés pour le frontend)."""

import logging
from pathlib import Path
from typing import Any

from flask import request
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import joinedload

from ext import db, role_required
from models.enums import PartnershipStatus, UserRole
from models.partnership import Partnership
from routes.companies import _get_current_company_via_use_case
from services.booking.transfers import BookingTransferService
from shared.error_handlers import APIErrorHandler
from shared.response_helpers import success_response

logger = logging.getLogger(__name__)

# Créer le namespace pour les partenariats
partnerships_ns = Namespace("partnerships", description="Partenariats")


@partnerships_ns.route("/for-transfer")
class PartnershipsForTransfer(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les partenariats disponibles pour le transfert de courses.

        Retourne uniquement les partenariats où l'entreprise est propriétaire (owner_company_id)
        et où le partenariat est actif (status=ACCEPTED).
        """
        try:
            logger.info(
                "[PartnershipsForTransfer] Endpoint called for company (will be determined)"
            )
            company, error_response, status_code = _get_current_company_via_use_case()
            logger.info(
                "[PartnershipsForTransfer] Company retrieved: id=%s, name=%s",
                company.id if company else None,
                company.name if company else None,
            )
            if error_response or not company:
                logger.warning(
                    "[PartnershipsForTransfer] Company not found or error: %s",
                    error_response,
                )
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            # #region agent log
            try:
                log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
                with log_path.open("a", encoding="utf-8") as f:
                    import json

                    f.write(
                        json.dumps(
                            {
                                "location": "partnerships.py:PartnershipsForTransfer.get",
                                "message": "Starting query",
                                "data": {
                                    "company_id": company.id,
                                    "company_name": company.name,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # Récupérer tous les partenariats où l'entreprise est propriétaire (pour debug)
            all_owner_partnerships = (
                db.session.query(Partnership)
                .filter(Partnership.owner_company_id == company.id)
                .all()
            )

            # Vérifier aussi les partenariats où l'entreprise est partenaire (pour debug)
            all_partner_partnerships = (
                db.session.query(Partnership)
                .filter(Partnership.partner_company_id == company.id)
                .all()
            )

            logger.info(
                (
                    "[PartnershipsForTransfer] Company %s: Found %s partnerships where company is owner, "
                    "%s partnerships where company is partner"
                ),
                company.id,
                len(all_owner_partnerships),
                len(all_partner_partnerships),
            )
            for p in all_owner_partnerships:
                logger.info(
                    (
                        "[PartnershipsForTransfer] Partnership %s (company is owner): status=%s, is_active=%s, "
                        "owner_company_id=%s, partner_company_id=%s"
                    ),
                    p.id,
                    p.status.value if hasattr(p.status, "value") else str(p.status),
                    p.is_active,
                    p.owner_company_id,
                    p.partner_company_id,
                )
            for p in all_partner_partnerships:
                logger.info(
                    (
                        "[PartnershipsForTransfer] Partnership %s (company is partner): status=%s, is_active=%s, "
                        "owner_company_id=%s, partner_company_id=%s"
                    ),
                    p.id,
                    p.status.value if hasattr(p.status, "value") else str(p.status),
                    p.is_active,
                    p.owner_company_id,
                    p.partner_company_id,
                )

            # #region agent log
            try:
                # Essayer plusieurs chemins possibles
                possible_paths = [
                    Path(r"c:\Users\jasiq\atmr\.cursor\debug.log"),
                    Path(".cursor/debug.log"),
                    Path("../.cursor/debug.log"),
                    Path.cwd() / ".cursor" / "debug.log",
                ]
                log_path = None
                for p in possible_paths:
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        log_path = p
                        break
                    except Exception:
                        continue

                if log_path:
                    with log_path.open("a", encoding="utf-8") as f:
                        import json

                        f.write(
                            json.dumps(
                                {
                                    "location": "partnerships.py:PartnershipsForTransfer.get",
                                    "message": "All owner partnerships found",
                                    "data": {
                                        "count": len(all_owner_partnerships),
                                        "partnerships": [
                                            {
                                                "id": p.id,
                                                "status": (
                                                    p.status.value
                                                    if hasattr(p.status, "value")
                                                    else str(p.status)
                                                ),
                                                "is_active": p.is_active,
                                                "owner_company_id": p.owner_company_id,
                                                "partner_company_id": p.partner_company_id,
                                            }
                                            for p in all_owner_partnerships
                                        ],
                                    },
                                    "timestamp": int(__import__("time").time() * 1000),
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "B",
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
            # #endregion

            # Récupérer les partenariats où l'entreprise est propriétaire OU partenaire
            # et où le partenariat est actif
            # Note: Une entreprise peut transférer des courses si elle est propriétaire OU partenaire
            # (car elle peut vouloir transférer des courses qu'elle a reçues d'un partenaire)
            partnerships = (
                db.session.query(Partnership)
                .options(
                    joinedload(Partnership.owner_company),
                    joinedload(Partnership.partner_company),
                )
                .filter(
                    (
                        (Partnership.owner_company_id == company.id)
                        | (Partnership.partner_company_id == company.id)
                    ),
                    Partnership.status == PartnershipStatus.ACCEPTED,
                    Partnership.is_active == True,  # noqa: E712
                )
                .all()
            )

            logger.info(
                (
                    "[PartnershipsForTransfer] After filtering (ACCEPTED + is_active=True): "
                    "found %s partnerships"
                ),
                len(partnerships),
            )

            # #region agent log
            try:
                # Essayer plusieurs chemins possibles
                possible_paths = [
                    Path(r"c:\Users\jasiq\atmr\.cursor\debug.log"),
                    Path(".cursor/debug.log"),
                    Path("../.cursor/debug.log"),
                    Path.cwd() / ".cursor" / "debug.log",
                ]
                log_path = None
                for p in possible_paths:
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        log_path = p
                        break
                    except Exception:
                        continue

                if log_path:
                    with log_path.open("a", encoding="utf-8") as f:
                        import json

                        f.write(
                            json.dumps(
                                {
                                    "location": "partnerships.py:PartnershipsForTransfer.get",
                                    "message": "Filtered partnerships",
                                    "data": {
                                        "count": len(partnerships),
                                        "partnerships": [
                                            {
                                                "id": p.id,
                                                "status": (
                                                    p.status.value
                                                    if hasattr(p.status, "value")
                                                    else str(p.status)
                                                ),
                                                "is_active": p.is_active,
                                            }
                                            for p in partnerships
                                        ],
                                    },
                                    "timestamp": int(__import__("time").time() * 1000),
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "C",
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
            # #endregion

            # Sérialiser les partenariats
            result = []
            for p in partnerships:
                # Déterminer quelle entreprise est le partenaire (pas l'entreprise actuelle)
                if p.owner_company_id == company.id:
                    # L'entreprise actuelle est propriétaire, le partenaire est partner_company
                    partner_company_id = p.partner_company_id
                    partner_company_name = (
                        p.partner_company.name if p.partner_company else None
                    )
                    is_owner = True
                else:
                    # L'entreprise actuelle est partenaire, le partenaire est owner_company
                    partner_company_id = p.owner_company_id
                    partner_company_name = (
                        p.owner_company.name if p.owner_company else None
                    )
                    is_owner = False

                logger.info(
                    (
                        "[PartnershipsForTransfer] Serializing partnership %s: "
                        "company_id=%s, is_owner=%s, partner_company_id=%s, partner_company_name=%s"
                    ),
                    p.id,
                    company.id,
                    is_owner,
                    partner_company_id,
                    partner_company_name,
                )

                p_dict = {
                    "id": p.id,
                    "partner_company_id": partner_company_id,
                    "partner_company_name": partner_company_name,
                    "status": (
                        p.status.value if hasattr(p.status, "value") else str(p.status)
                    ),
                    "is_active": p.is_active,
                    "default_transfer_model": (
                        p.default_transfer_model.value
                        if p.default_transfer_model
                        and hasattr(p.default_transfer_model, "value")
                        else (
                            str(p.default_transfer_model)
                            if p.default_transfer_model
                            else None
                        )
                    ),
                }
                result.append(p_dict)

            logger.info(
                (
                    "[PartnershipsForTransfer] Company %s: Found %s partnerships for transfer. "
                    "Result: %s"
                ),
                company.id,
                len(result),
                result,
            )
            response = success_response(data=result)
            logger.info(
                "[PartnershipsForTransfer] Response content-length: %s bytes",
                len(str(response[0])),
            )
            return response
        except Exception as e:
            logger.exception("[PartnershipsForTransfer] Error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


def _validate_transfer_request(
    company, booking_id: int
) -> tuple[dict[str, Any] | None, int | None]:
    """Valide la requête de transfert et vérifie les permissions.

    Returns:
        (error_response, status_code) ou (None, None) si OK
    """
    from models.booking import Booking

    booking = Booking.query.get(booking_id)
    if not booking:
        # APIErrorHandler.handle_not_found retourne déjà (dict, int)
        error_response, _ = APIErrorHandler.handle_not_found(
            "Booking", booking_id, logger
        )
        return error_response, 404

    # ✅ P0-2: Vérifier que la company authentifiée est propriétaire de la course
    if booking.company_id != company.id:
        logger.warning(
            "[PartnershipTransfers] ⛔ Company %s attempted to transfer booking %s owned by company %s",
            company.id,
            booking_id,
            booking.company_id,
        )
        # APIErrorHandler.handle_validation_error retourne déjà (dict, int)
        error_response, _ = APIErrorHandler.handle_validation_error(
            "Vous ne pouvez transférer que les courses de votre entreprise",
            field="booking_id",
            logger_instance=logger,
        )
        return error_response, 403

    return None, None


@partnerships_ns.route("/<int:partnership_id>/transfers")
class PartnershipTransfers(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, partnership_id: int):
        """Propose un transfert de course à un partenaire.

        Args:
            partnership_id: ID du partenariat
            booking_id: ID de la course à transférer (dans le body)
            transfer_model: Modèle de transfert optionnel (dans le body)
        """
        try:
            logger.info(
                "[PartnershipTransfers] POST /partnerships/%s/transfers called",
                partnership_id,
            )
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                logger.warning(
                    "[PartnershipTransfers] Company not found or error: %s",
                    error_response,
                )
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            data = request.get_json(silent=True) or {}
            booking_id_raw = data.get("booking_id")
            transfer_model_str = data.get("transfer_model")

            # #region agent log
            try:
                import json

                log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "partnerships.py:407",
                                "message": "Received data",
                                "data": {
                                    "booking_id_raw": str(booking_id_raw),
                                    "booking_id_type": type(booking_id_raw).__name__,
                                    "partnership_id": partnership_id,
                                    "partnership_id_type": type(
                                        partnership_id
                                    ).__name__,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # ✅ Valider et convertir booking_id en entier pour éviter l'erreur SQL "integer = character varying"
            if not booking_id_raw:
                raise ValueError("booking_id is required")

            try:
                booking_id = int(booking_id_raw)
                # #region agent log
                try:
                    import json

                    log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "partnerships.py:415",
                                    "message": "booking_id converted successfully",
                                    "data": {
                                        "booking_id": booking_id,
                                        "booking_id_type": type(booking_id).__name__,
                                    },
                                    "timestamp": int(__import__("time").time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
            except (ValueError, TypeError) as e:
                # #region agent log
                try:
                    import json

                    log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "partnerships.py:416",
                                    "message": "booking_id conversion FAILED",
                                    "data": {
                                        "booking_id_raw": str(booking_id_raw),
                                        "error": str(e),
                                    },
                                    "timestamp": int(__import__("time").time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
                logger.warning(
                    "[PartnershipTransfers] Invalid booking_id: %s (type: %s)",
                    booking_id_raw,
                    type(booking_id_raw).__name__,
                )
                raise ValueError("booking_id must be a valid integer") from e

            # Convertir transfer_model de string à enum si nécessaire
            transfer_model = None
            if transfer_model_str:
                from models.enums import TransferModel

                try:
                    transfer_model = TransferModel(transfer_model_str)
                except (ValueError, TypeError):
                    logger.warning(
                        "[PartnershipTransfers] Invalid transfer_model: %s, using partnership default",
                        transfer_model_str,
                    )

            logger.info(
                (
                    "[PartnershipTransfers] Company %s proposing transfer: "
                    "partnership_id=%s, booking_id=%s, transfer_model=%s"
                ),
                company.id,
                partnership_id,
                booking_id,
                transfer_model,
            )

            # ✅ P0-2: Valider la requête et vérifier les permissions
            error_response, status_code = _validate_transfer_request(
                company, booking_id
            )
            # #region agent log
            try:
                import json

                log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H4",
                                "location": "partnerships.py:449",
                                "message": "After validation",
                                "data": {
                                    "has_error": bool(error_response),
                                    "error_response": str(error_response)
                                    if error_response
                                    else None,
                                    "status_code": status_code,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            if error_response:
                # Lever une ValueError pour traitement uniforme dans le bloc except
                raise ValueError(error_response.get("error", "Validation failed"))

            # #region agent log
            try:
                import json

                log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "partnerships.py:457",
                                "message": "Before propose_transfer call",
                                "data": {
                                    "booking_id": booking_id,
                                    "booking_id_type": type(booking_id).__name__,
                                    "partnership_id": partnership_id,
                                    "partnership_id_type": type(
                                        partnership_id
                                    ).__name__,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # Utiliser le service pour créer le transfert
            transfer = BookingTransferService.propose_transfer(
                booking_id=booking_id,
                partnership_id=partnership_id,
                transfer_model=transfer_model,
            )

            logger.info(
                "[PartnershipTransfers] Transfer created successfully: transfer_id=%s",
                transfer.id,
            )

            return success_response(
                data=transfer.to_dict(), message="Transfert proposé avec succès"
            )
        except ValueError as e:
            # ✅ Erreurs de validation métier (ValueError)
            logger.warning("[PartnershipTransfers] Validation error: %s", e)
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            ), 400
        except Exception as e:
            # ✅ Erreurs inattendues ou erreurs système
            logger.exception("[PartnershipTransfers] Unexpected error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@partnerships_ns.route("/transfers")
class TransfersList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère la liste des transferts (entrants et sortants).

        Query params:
            - partnership_id (optionnel): Filtrer par partenariat
            - status (optionnel): Filtrer par statut (PENDING, ACCEPTED, REJECTED)
        """
        try:
            logger.info("[TransfersList] GET /partnerships/transfers called")
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                logger.warning(
                    "[TransfersList] Company not found or error: %s", error_response
                )
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            # Récupérer les filtres
            partnership_id = request.args.get("partnership_id", type=int)
            status_filter = request.args.get("status")

            logger.info(
                "[TransfersList] Company %s requesting transfers: partnership_id=%s, status=%s",
                company.id,
                partnership_id,
                status_filter,
            )

            # Construire la requête de base
            from models.booking_transfer import BookingTransfer
            from models.partnership import Partnership

            query = db.session.query(BookingTransfer).options(
                joinedload(BookingTransfer.partnership),
                joinedload(BookingTransfer.booking),
            )

            # Filtrer les transferts liés à l'entreprise (entrants OU sortants)
            query = query.join(Partnership).filter(
                (Partnership.owner_company_id == company.id)
                | (Partnership.partner_company_id == company.id)
            )

            # Appliquer les filtres
            if partnership_id:
                query = query.filter(BookingTransfer.partnership_id == partnership_id)

            if status_filter:
                from models.enums import TransferStatus

                try:
                    status_enum = TransferStatus(status_filter)
                    query = query.filter(BookingTransfer.status == status_enum)
                except ValueError:
                    logger.warning(
                        "[TransfersList] Invalid status filter: %s", status_filter
                    )

            # Ordonner par date de proposition (plus récent d'abord)
            query = query.order_by(BookingTransfer.requested_at.desc())

            transfers = query.all()

            logger.info(
                "[TransfersList] Company %s: Found %s transfers",
                company.id,
                len(transfers),
            )

            # Sérialiser les transferts
            result = [transfer.to_dict() for transfer in transfers]

            return success_response(data=result)
        except Exception as e:
            logger.exception("[TransfersList] Error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@partnerships_ns.route("/transfers/<int:transfer_id>/accept")
class TransferAccept(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, transfer_id: int):  # noqa: PLR0911
        """Accepte un transfert de course proposé par une entreprise partenaire.

        Args:
            transfer_id: ID du transfert à accepter
        """
        try:
            logger.info(
                "[TransferAccept] POST /partnerships/transfers/%s/accept called",
                transfer_id,
            )
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                logger.warning(
                    "[TransferAccept] Company not found or error: %s", error_response
                )
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            # Récupérer le transfert
            from models.booking_transfer import BookingTransfer

            transfer = BookingTransfer.query.get(transfer_id)
            if not transfer:
                return APIErrorHandler.handle_not_found(
                    "Transfer", transfer_id, logger
                ), 404

            # Vérifier que l'entreprise est le destinataire du transfert
            partnership = transfer.partnership
            if not partnership:
                return APIErrorHandler.handle_validation_error(
                    "Partenariat introuvable",
                    field="partnership_id",
                    logger_instance=logger,
                ), 400

            # L'entreprise doit être le partenaire qui reçoit le transfert
            # Si owner_company_id a proposé, partner_company_id accepte et vice-versa
            is_recipient = False
            if (
                transfer.booking
                and transfer.booking.company_id == partnership.owner_company_id
            ):
                # La course vient du owner, donc partner doit accepter
                is_recipient = partnership.partner_company_id == company.id
            else:
                # La course vient du partner, donc owner doit accepter
                is_recipient = partnership.owner_company_id == company.id

            if not is_recipient:
                logger.warning(
                    "[TransferAccept] Company %s attempted to accept transfer %s but is not the recipient",
                    company.id,
                    transfer_id,
                )
                return APIErrorHandler.handle_validation_error(
                    "Vous n'êtes pas le destinataire de ce transfert",
                    field="transfer_id",
                    logger_instance=logger,
                ), 403

            # Utiliser le service pour accepter le transfert
            accepted_transfer = BookingTransferService.accept_transfer(
                transfer_id, executing_company_id=company.id
            )

            logger.info(
                "[TransferAccept] Transfer %s accepted successfully by company %s",
                transfer_id,
                company.id,
            )

            return success_response(
                data=accepted_transfer.to_dict(),
                message="Transfert accepté avec succès",
            )
        except ValueError as e:
            logger.warning("[TransferAccept] Validation error: %s", e)
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            ), 400
        except Exception as e:
            logger.exception("[TransferAccept] Error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@partnerships_ns.route("/transfers/<int:transfer_id>/reject")
class TransferReject(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, transfer_id: int):  # noqa: PLR0911
        """Refuse un transfert de course proposé par une entreprise partenaire.

        Args:
            transfer_id: ID du transfert à refuser
            reason (optionnel): Raison du refus (dans le body)
        """
        try:
            logger.info(
                "[TransferReject] POST /partnerships/transfers/%s/reject called",
                transfer_id,
            )
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                logger.warning(
                    "[TransferReject] Company not found or error: %s", error_response
                )
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            data = request.get_json(silent=True) or {}
            reason = data.get("reason")

            # Récupérer le transfert
            from models.booking_transfer import BookingTransfer

            transfer = BookingTransfer.query.get(transfer_id)
            if not transfer:
                return APIErrorHandler.handle_not_found(
                    "Transfer", transfer_id, logger
                ), 404

            # Vérifier que l'entreprise est le destinataire du transfert
            partnership = transfer.partnership
            if not partnership:
                return APIErrorHandler.handle_validation_error(
                    "Partenariat introuvable",
                    field="partnership_id",
                    logger_instance=logger,
                ), 400

            # L'entreprise doit être le partenaire qui reçoit le transfert
            is_recipient = False
            if (
                transfer.booking
                and transfer.booking.company_id == partnership.owner_company_id
            ):
                # La course vient du owner, donc partner doit accepter/refuser
                is_recipient = partnership.partner_company_id == company.id
            else:
                # La course vient du partner, donc owner doit accepter/refuser
                is_recipient = partnership.owner_company_id == company.id

            if not is_recipient:
                logger.warning(
                    "[TransferReject] Company %s attempted to reject transfer %s but is not the recipient",
                    company.id,
                    transfer_id,
                )
                return APIErrorHandler.handle_validation_error(
                    "Vous n'êtes pas le destinataire de ce transfert",
                    field="transfer_id",
                    logger_instance=logger,
                ), 403

            # Utiliser le service pour refuser le transfert
            # Note: Le paramètre 'reason' n'est pas supporté par le service actuellement
            # Il pourrait être ajouté dans une future version pour stocker la raison du refus
            rejected_transfer = BookingTransferService.reject_transfer(
                transfer_id, executing_company_id=company.id
            )

            logger.info(
                "[TransferReject] Transfer %s rejected successfully by company %s (reason: %s)",
                transfer_id,
                company.id,
                reason or "non spécifiée",
            )

            return success_response(
                data=rejected_transfer.to_dict(),
                message="Transfert refusé avec succès",
            )
        except ValueError as e:
            logger.warning("[TransferReject] Validation error: %s", e)
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            ), 400
        except Exception as e:
            logger.exception("[TransferReject] Error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
