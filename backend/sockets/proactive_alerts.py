#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
"""Socket.IO pour alertes proactives temps réel.

Ce module gère les connexions WebSocket pour:
- Diffusion d'alertes de retard en temps réel
- Notifications d'explicabilité RL
- Monitoring live desdécisions
- Gestion des subscriptions par entreprise

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Set

from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms

from services.proactive_alerts import ProactiveAlertsService
from sockets.websocket_ack import get_ack_manager, register_ack_handlers

logger = logging.getLogger(__name__)

# Instance globale du service
alerts_service = ProactiveAlertsService()

# Stockage des connexions actives par entreprise
active_connections: Dict[str, Set[str]] = {}


def register_proactive_alerts_sockets(socketio: SocketIO):
    """Enregistre les handlers Socket.IO pour les alertes proactives."""

    @socketio.on("connect")
    def handle_connect():
        """Gère la connexion d'un client."""
        try:
            client_id = getattr(request, "sid", "unknown")
            logger.info("[ProactiveAlerts] Client connecté: %s", client_id)

            # Envoyer confirmation de connexion
            emit("connection_established", {
                "status": "connected",
                "client_id": client_id,
                "timestamp": datetime.now(UTC).isoformat()
            })

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur connexion: %s", e)
            emit("connection_error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        """Gère la déconnexion d'un client."""
        try:
            client_id = getattr(request, "sid", "unknown")

            # Retirer de toutes les rooms
            current_rooms = rooms()
            for room in current_rooms:
                if room != client_id:  # Ne pas quitter sa propre room
                    leave_room(room)

            # Nettoyer les connexions actives
            for company_id, connections in active_connections.items():
                connections.discard(client_id)
                if not connections:
                    del active_connections[company_id]

            logger.info("[ProactiveAlerts] Client déconnecté: %s", client_id)

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur déconnexion: %s", e)

    @socketio.on("subscribe_alerts")
    def handle_subscribe_alerts(data: Dict[str, Any]):
        """Gère la subscription aux alertes pour une entreprise.
        
        Data:
            {
                "company_id": "company_123",
                "alert_types": ["delay_risk", "rl_explanation"],
                "filters": {
                    "risk_levels": ["high", "medium"],
                    "booking_ids": ["123", "124"]
                }
            }
        """
        try:
            client_id = getattr(request, "sid", "unknown")
            company_id = data.get("company_id")
            alert_types = data.get("alert_types", ["delay_risk"])
            filters = data.get("filters", {})

            if not company_id:
                emit("subscription_error", {
                    "error": "company_id requis"
                })
                return

            # Rejoindre la room de l'entreprise
            room_name = f"company_{company_id}"
            join_room(room_name)

            # Enregistrer la connexion
            if company_id not in active_connections:
                active_connections[company_id] = set()
            active_connections[company_id].add(client_id)

            # Confirmer la subscription
            emit("subscription_confirmed", {
                "company_id": company_id,
                "room": room_name,
                "alert_types": alert_types,
                "filters": filters,
                "timestamp": datetime.now(UTC).isoformat()
            })

            logger.info(
                "[ProactiveAlerts] Client %s subscribé aux alertes pour company %s",
                client_id, company_id
            )

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur subscription: %s", e)
            emit("subscription_error", {"error": str(e)})

    @socketio.on("unsubscribe_alerts")
    def handle_unsubscribe_alerts(data: Dict[str, Any]):
        """Gère la désubscription aux alertes.
        
        Data:
            {
                "company_id": "company_123"
            }
        """
        try:
            client_id = getattr(request, "sid", "unknown")
            company_id = data.get("company_id")

            if company_id:
                room_name = f"company_{company_id}"
                leave_room(room_name)

                # Retirer de la liste des connexions actives
                if company_id in active_connections:
                    active_connections[company_id].discard(client_id)
                    if not active_connections[company_id]:
                        del active_connections[company_id]

                emit("unsubscription_confirmed", {
                    "company_id": company_id,
                    "timestamp": datetime.now(UTC).isoformat()
                })

                logger.info(
                    "[ProactiveAlerts] Client %s désubscrit des alertes company %s",
                    client_id, company_id
                )
            else:
                # Désubscription complète
                current_rooms = rooms()
                for room in current_rooms:
                    if room.startswith("company_") and room != client_id:
                        leave_room(room)

                # Nettoyer toutes les connexions
                for company_id, connections in list(active_connections.items()):
                    connections.discard(client_id)
                    if not connections:
                        del active_connections[company_id]

                emit("unsubscription_confirmed", {
                    "message": "Désubscription complète",
                    "timestamp": datetime.now(UTC).isoformat()
                })

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur désubscription: %s", e)
            emit("unsubscription_error", {"error": str(e)})

    @socketio.on("request_explanation")
    def handle_request_explanation(data: Dict[str, Any]):
        """Gère les demandes d'explication RL en temps réel.
        
        Data:
            {
                "booking_id": "123",
                "driver_id": "456",
                "rl_decision": {...}
            }
        """
        try:
            booking_id = data.get("booking_id")
            driver_id = data.get("driver_id")
            rl_decision = data.get("rl_decision", {})

            if not all([booking_id, driver_id]):
                emit("explanation_error", {
                    "error": "booking_id et driver_id requis"
                })
                return

            # Générer l'explication
            explanation = alerts_service.get_explanation_for_decision(
                booking_id=str(booking_id),
                driver_id=str(driver_id),
                rl_decision=rl_decision
            )

            # Envoyer l'explication
            emit("explanation_response", {
                "booking_id": booking_id,
                "driver_id": driver_id,
                "explanation": explanation,
                "timestamp": datetime.now(UTC).isoformat()
            })

            logger.info(
                "[ProactiveAlerts] Explication envoyée - Booking %s, Driver %s",
                booking_id, driver_id
            )

        except Exception as e:
            logger.error("[ProactiveAlerts] Erreur demande explication: %s", e)
            emit("explanation_error", {"error": str(e)})

    @socketio.on("ping")
    def handle_ping():
        """Gère les pings de keep-alive."""
        emit("pong", {
            "timestamp": datetime.now(UTC).isoformat()
        })

    # ✅ C3: Enregistrer handlers ACK
    register_ack_handlers(socketio)
    
    @socketio.on("message_ack")
    def handle_message_ack(data: Dict[str, Any]):
        """✅ C3: Handler pour ACK côté serveur."""
        message_id = data.get("message_id")
        if message_id:
            manager = get_ack_manager()
            manager.on_ack_received(message_id)
    
    logger.info("[ProactiveAlerts] Handlers Socket.IO enregistrés (incluant ACK C3)")
    
    # Référencer les handlers pour indiquer qu'ils sont utilisés par Socket.IO
    _handlers = (handle_connect, handle_disconnect, handle_subscribe_alerts, 
                 handle_unsubscribe_alerts, handle_request_explanation, handle_ping,
                 handle_message_ack)


def broadcast_delay_alert(
    company_id: str,
    analysis_result: Dict[str, Any],
    socketio: SocketIO
) -> bool:
    """Diffuse une alerte de retard à tous les clients d'une entreprise.
    
    Args:
        company_id: ID de l'entreprise
        analysis_result: Résultat de l'analyse de risque
        socketio: Instance SocketIO
        
    Returns:
        True si diffusion réussie

    """
    try:
        room_name = f"company_{company_id}"

        # Construire le message d'alerte
        alert_message = {
            "type": "delay_risk_alert",
            "data": analysis_result,
            "timestamp": datetime.now(UTC).isoformat(),
            "priority": analysis_result.get("risk_level", "unknown")
        }

        # Diffuser à la room de l'entreprise
        socketio.emit("delay_alert", alert_message, room=room_name)  # type: ignore[call-arg]

        logger.info(
            "[ProactiveAlerts] Alerte diffusée - Company %s, Room %s, Risque %s",
            company_id, room_name, analysis_result.get("risk_level")
        )

        return True

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur diffusion alerte: %s", e)
        return False


def broadcast_rl_explanation(
    company_id: str,
    explanation: Dict[str, Any],
    socketio: SocketIO
) -> bool:
    """Diffuse une explication RL à tous les clients d'une entreprise.
    
    Args:
        company_id: ID de l'entreprise
        explanation: Explication RL
        socketio: Instance SocketIO
        
    Returns:
        True si diffusion réussie

    """
    try:
        room_name = f"company_{company_id}"

        # Construire le message d'explication
        explanation_message = {
            "type": "rl_explanation",
            "data": explanation,
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Diffuser à la room de l'entreprise
        socketio.emit("rl_explanation", explanation_message, room=room_name)  # type: ignore[call-arg]

        logger.info(
            "[ProactiveAlerts] Explication RL diffusée - Company %s, Booking %s",
            company_id, explanation.get("booking_id")
        )

        return True

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur diffusion explication: %s", e)
        return False


def broadcast_system_status(
    company_id: str,
    status_data: Dict[str, Any],
    socketio: SocketIO
) -> bool:
    """Diffuse le statut système à tous les clients d'une entreprise.
    
    Args:
        company_id: ID de l'entreprise
        status_data: Données de statut
        socketio: Instance SocketIO
        
    Returns:
        True si diffusion réussie

    """
    try:
        room_name = f"company_{company_id}"

        # Construire le message de statut
        status_message = {
            "type": "system_status",
            "data": status_data,
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Diffuser à la room de l'entreprise
        socketio.emit("system_status", status_message, room=room_name)  # type: ignore[call-arg]

        logger.info(
            "[ProactiveAlerts] Statut système diffusé - Company %s", company_id
        )

        return True

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur diffusion statut: %s", e)
        return False


def get_active_connections_stats() -> Dict[str, Any]:
    """Retourne les statistiques des connexions actives.
    
    Returns:
        Statistiques des connexions

    """
    try:
        total_connections = sum(len(connections) for connections in active_connections.values())

        return {
            "total_companies": len(active_connections),
            "total_connections": total_connections,
            "companies": {
                company_id: len(connections)
                for company_id, connections in active_connections.items()
            },
            "timestamp": datetime.now(UTC).isoformat()
        }


    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur stats connexions: %s", e)
        return {
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }


def cleanup_inactive_connections(socketio: SocketIO) -> int:
    """Nettoie les connexions inactives.
    
    Args:
        socketio: Instance SocketIO
        
    Returns:
        Nombre de connexions nettoyées

    """
    try:
        cleaned_count = 0

        # Vérifier chaque entreprise
        for company_id, connections in list(active_connections.items()):
            active_connections_copy = connections.copy()

            for client_id in active_connections_copy:
                # Vérifier si la connexion est toujours active
                if hasattr(socketio, "server") and hasattr(socketio.server, "manager") and not socketio.server.manager.is_connected(client_id):
                    connections.discard(client_id)
                    cleaned_count += 1

            # Supprimer les entreprises sans connexions
            if not connections:
                del active_connections[company_id]

        if cleaned_count > 0:
            logger.info(
                "[ProactiveAlerts] %s connexions inactives nettoyées", cleaned_count
            )

        return cleaned_count

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur nettoyage connexions: %s", e)
        return 0


# Fonction utilitaire pour intégrer avec le service d'alertes
def integrate_with_alerts_service(socketio: SocketIO):
    """Intègre le service d'alertes avec Socket.IO.
    
    Cette fonction peut être appelée depuis le service d'alertes
    pour diffuser automatiquement les alertes via WebSocket.
    """

    def enhanced_send_alert(original_send_alert):
        """Wrapper pour diffuser les alertes via Socket.IO."""
        def wrapper(analysis_result: Dict[str, Any], company_id: str, force_send: bool = False):
            # Appeler la méthode originale
            success = original_send_alert(analysis_result, company_id, force_send)

            # Si l'alerte a été envoyée, la diffuser via Socket.IO
            if success:
                broadcast_delay_alert(company_id, analysis_result, socketio)

            return success

        return wrapper

    # Appliquer le wrapper au service d'alertes
    alerts_service.send_proactive_alert = enhanced_send_alert(
        alerts_service.send_proactive_alert
    )

    logger.info("[ProactiveAlerts] Service d'alertes intégré avec Socket.IO")
