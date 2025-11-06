"""✅ D2: Log d'audit append-only pour décisions.

Objectif: Conformité juridique + audit RGPD.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from sqlalchemy import BigInteger, DateTime, Integer, String, Text

from ext import db

logger = logging.getLogger(__name__)


class AuditLog(db.Model):
    """✅ D2: Modèle d'audit append-only (pas de UPDATE/DELETE).
    
    Note: Les enregistrements ne sont jamais modifiés/supprimés.
    """
    
    __tablename__ = "audit_logs"
    
    # Identifiant unique (auto-incrémental)
    id = db.Column(BigInteger, primary_key=True)
    
    # Timestamp de création (immutable)
    created_at = db.Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    
    # Utilisateur/Système
    user_id = db.Column(Integer, nullable=True)  # NULL pour actions système
    user_type = db.Column(String(50), nullable=False)  # 'admin', 'operator', 'system', etc.
    
    # Action effectuée
    action_type = db.Column(String(100), nullable=False)  # 'dispatch', 'booking_update', 'driver_assign', etc.
    action_category = db.Column(String(50), nullable=False)  # 'dispatch', 'security', 'data_access', etc.
    
    # Détails de l'action (JSON)
    action_details = db.Column(Text, nullable=False)  # JSON string
    
    # Résultat
    result_status = db.Column(String(50), nullable=False)  # 'success', 'failure', 'partial'
    result_message = db.Column(Text, nullable=True)
    
    # Contexte
    company_id = db.Column(Integer, nullable=True)
    booking_id = db.Column(Integer, nullable=True)
    driver_id = db.Column(Integer, nullable=True)
    
    # Sécurité
    ip_address = db.Column(String(45), nullable=True)  # IPv4 ou IPv6
    user_agent = db.Column(Text, nullable=True)
    
    # Métadonnées
    additional_metadata = db.Column(Text, nullable=True)  # JSON string supplémentaire
    
    def __repr__(self) -> str:  # type: ignore[override]
        return f"<AuditLog {self.id}: {self.action_type} by {self.user_type} at {self.created_at}>"


class AuditLogger:
    """✅ D2: Logger pour audit append-only."""
    
    @staticmethod
    def log_action(
        action_type: str,
        action_category: str,
        user_id: Optional[int] = None,
        user_type: str = "system",
        result_status: str = "success",
        result_message: Optional[str] = None,
        action_details: Optional[Dict[str, Any]] = None,
        company_id: Optional[int] = None,
        booking_id: Optional[int] = None,
        driver_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """✅ D2: Enregistre une action d'audit (insert-only).
        
        Args:
            action_type: Type d'action (ex: 'dispatch_complete')
            action_category: Catégorie (ex: 'dispatch', 'security')
            user_id: ID utilisateur (optionnel pour actions système)
            user_type: Type d'utilisateur ('admin', 'operator', 'system')
            result_status: Statut ('success', 'failure', 'partial')
            result_message: Message de résultat
            action_details: Détails JSON de l'action
            company_id: ID entreprise (optionnel)
            booking_id: ID booking (optionnel)
            driver_id: ID driver (optionnel)
            ip_address: Adresse IP (optionnel)
            user_agent: User-Agent (optionnel)
            metadata: Métadonnées supplémentaires (optionnel)
            
        Returns:
            AuditLog créé
        """
        import json
        
        # SQLAlchemy crée dynamiquement les paramètres via métaclasse
        audit_log = AuditLog()
        audit_log.user_id = user_id
        audit_log.user_type = user_type
        audit_log.action_type = action_type
        audit_log.action_category = action_category
        audit_log.action_details = json.dumps(action_details or {})
        audit_log.result_status = result_status
        audit_log.result_message = result_message
        audit_log.company_id = company_id
        audit_log.booking_id = booking_id
        audit_log.driver_id = driver_id
        audit_log.ip_address = ip_address
        audit_log.user_agent = user_agent
        audit_log.additional_metadata = json.dumps(metadata or {})
        audit_log.created_at = datetime.now(UTC)
        
        try:
            db.session.add(audit_log)
            db.session.commit()
            logger.debug("[D2] Audit log créé: %s", audit_log)
        except Exception as e:
            logger.error("[D2] Échec création audit log: %s", e)
            db.session.rollback()
            raise
        
        return audit_log
    
    @staticmethod
    def log_dispatch_action(
        dispatch_run_id: int,
        company_id: int,
        assignments_count: int,
        unassigned_count: int,
        mode: str,
        user_id: Optional[int] = None,
        result_status: str = "success",
    ) -> AuditLog:
        """✅ D2: Log spécifique pour dispatch."""
        return AuditLogger.log_action(
            action_type="dispatch_complete",
            action_category="dispatch",
            user_id=user_id,
            user_type="system",
            result_status=result_status,
            result_message=f"Dispatch run {dispatch_run_id}: {assignments_count} assigned, {unassigned_count} unassigned",
            action_details={
                "dispatch_run_id": dispatch_run_id,
                "mode": mode,
                "assignments_count": assignments_count,
                "unassigned_count": unassigned_count,
            },
            company_id=company_id,
        )
    
    @staticmethod
    def log_data_access(
        user_id: int,
        user_type: str,
        data_type: str,
        data_id: int,
        company_id: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        """✅ D2: Log d'accès aux données sensibles."""
        return AuditLogger.log_action(
            action_type="data_access",
            action_category="security",
            user_id=user_id,
            user_type=user_type,
            result_status="success",
            action_details={
                "data_type": data_type,
                "data_id": data_id,
            },
            company_id=company_id,
            ip_address=ip_address,
        )
    
    @staticmethod
    def log_security_event(
        event_type: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        """✅ D2: Log d'événement de sécurité."""
        return AuditLogger.log_action(
            action_type=event_type,
            action_category="security",
            user_id=user_id,
            result_status=severity,
            action_details=details or {},
            ip_address=ip_address,
        )

