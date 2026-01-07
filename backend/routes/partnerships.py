# backend/routes/partnerships.py
"""Routes pour les partenariats (endpoints simplifiés pour le frontend)."""

import logging
from pathlib import Path
from typing import Any

from flask import request  # pyright: ignore[reportMissingImports]
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
            booking_id = data.get("booking_id")
            transfer_model_str = data.get("transfer_model")

            if not booking_id:
                return APIErrorHandler.handle_validation_error(
                    "booking_id is required",
                    field="booking_id",
                    logger_instance=logger,
                ), 400

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
            if error_response:
                return error_response, status_code

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
            logger.warning("[PartnershipTransfers] Validation error: %s", e)
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            ), 400
        except Exception as e:
            logger.exception("[PartnershipTransfers] Error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
