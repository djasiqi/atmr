"""✅ Utilitaire pour convertir les erreurs DB en messages clairs pour l'API.

Fournit des helpers pour intercepter IntegrityError et retourner des messages
compréhensibles au lieu de messages techniques PostgreSQL.
"""

import re
from typing import Any, Dict, Tuple

from sqlalchemy.exc import IntegrityError


def format_integrity_error(error: IntegrityError) -> Tuple[Dict[str, Any], int]:
    """Convertit une IntegrityError en message d'erreur clair pour l'API.

    Analyse le code d'erreur PostgreSQL et le message pour déterminer
    le type de contrainte violée et retourner un message utilisateur-friendly.

    Args:
        error: Exception IntegrityError de SQLAlchemy

    Returns:
        Tuple (response_json, status_code) pour Flask

    Exemples:
        - Foreign key violation → "Client inexistant" ou "Chauffeur inexistant"
        - Unique constraint violation → "Cette valeur existe déjà"
        - Check constraint violation → "Valeur invalide pour ce champ"
    """
    # Extraire le code d'erreur PostgreSQL
    error_code = None
    if (
        hasattr(error, "orig")
        and error.orig is not None
        and hasattr(error.orig, "pgcode")
    ):
        error_code = error.orig.pgcode

    # Extraire le message d'erreur
    error_message = str(error)
    error_detail = None
    if (
        hasattr(error, "orig")
        and error.orig is not None
        and hasattr(error.orig, "diag")
    ):
        diag = error.orig.diag
        if diag is not None:
            if hasattr(diag, "message_detail"):
                error_detail = diag.message_detail
            elif hasattr(diag, "message_primary"):
                error_detail = diag.message_primary

    # Codes d'erreur PostgreSQL courants
    # 23503 = foreign_key_violation
    # 23505 = unique_violation
    # 23514 = check_violation
    # 23502 = not_null_violation

    if error_code == "23503":  # Foreign key violation
        return _format_foreign_key_error(error_message, error_detail)
    if error_code == "23505":  # Unique constraint violation
        return _format_unique_constraint_error(error_message, error_detail)
    if error_code == "23514":  # Check constraint violation
        return _format_check_constraint_error(error_message, error_detail)
    if error_code == "23502":  # Not null violation
        return _format_not_null_error(error_message, error_detail)
    # Erreur d'intégrité non reconnue
    return {
        "error": "database_constraint_error",
        "message": "Erreur de contrainte de base de données. Vérifiez vos données.",
    }, 400


def _format_foreign_key_error(
    error_message: str, error_detail: str | None
) -> Tuple[Dict[str, Any], int]:
    """Formate une erreur de foreign key en message clair."""
    # Analyser le message pour extraire la table référencée
    # Exemples de messages PostgreSQL:
    # "insert or update on table "booking" violates foreign key constraint
    # "booking_client_id_fkey""
    # "Key (client_id)=(999) is not present in table "client"."

    message = "Référence invalide dans les données."

    # Détecter la table référencée depuis le message
    if error_detail:
        # Exemple: "Key (client_id)=(999) is not present in table "client"."
        table_match = re.search(r'table "(\w+)"', error_detail, re.IGNORECASE)

        if table_match:
            table_name = table_match.group(1).lower()

            # Mapping des tables vers des messages clairs
            table_messages = {
                "client": "Client inexistant",
                "user": "Utilisateur inexistant",
                "driver": "Chauffeur inexistant",
                "company": "Entreprise inexistante",
                "booking": "Réservation inexistante",
                "vehicle": "Véhicule inexistant",
            }

            # Si on trouve un message spécifique pour cette table
            if table_name in table_messages:
                message = table_messages[table_name]
            else:
                # Message générique avec le nom de la table
                message = f"{table_name.capitalize()} référencé(e) inexistant(e)"
    elif "client" in error_message.lower():
        message = "Client inexistant"
    elif "driver" in error_message.lower() or "chauffeur" in error_message.lower():
        message = "Chauffeur inexistant"
    elif "user" in error_message.lower() or "utilisateur" in error_message.lower():
        message = "Utilisateur inexistant"
    elif "company" in error_message.lower() or "entreprise" in error_message.lower():
        message = "Entreprise inexistante"
    elif "booking" in error_message.lower() or "réservation" in error_message.lower():
        message = "Réservation inexistante"

    return {
        "error": "foreign_key_violation",
        "message": message,
    }, 400


def _format_unique_constraint_error(
    error_message: str, error_detail: str | None
) -> Tuple[Dict[str, Any], int]:
    """Formate une erreur de contrainte unique en message clair."""
    # Analyser le message pour extraire le champ en conflit
    # Exemples:
    # "duplicate key value violates unique constraint "user_email_key""
    # "Key (email)=(test@example.com) already exists."

    message = "Cette valeur existe déjà."

    if error_detail:
        # Exemple: "Key (email)=(test@example.com) already exists."
        column_match = re.search(r"Key \(([^)]+)\)", error_detail, re.IGNORECASE)
        if column_match:
            column_name = column_match.group(1).lower()

            # Mapping des colonnes vers des messages clairs
            column_messages = {
                "email": "Cet email est déjà utilisé",
                "username": "Ce nom d'utilisateur est déjà utilisé",
                "phone": "Ce numéro de téléphone est déjà utilisé",
                "license_plate": "Cette plaque d'immatriculation existe déjà",
            }

            column_lower = column_name.lower()
            if column_lower in column_messages:
                message = column_messages[column_lower]
            else:
                message = f"Cette valeur pour '{column_name}' existe déjà"
    elif "email" in error_message.lower():
        message = "Cet email est déjà utilisé"
    elif "username" in error_message.lower():
        message = "Ce nom d'utilisateur est déjà utilisé"
    elif "phone" in error_message.lower():
        message = "Ce numéro de téléphone est déjà utilisé"

    return {
        "error": "unique_constraint_violation",
        "message": message,
    }, 400


def _format_check_constraint_error(
    error_message: str,  # noqa: ARG001
    error_detail: str | None,  # noqa: ARG001
) -> Tuple[Dict[str, Any], int]:
    """Formate une erreur de contrainte check en message clair."""
    return {
        "error": "check_constraint_violation",
        "message": (
            "Valeur invalide pour ce champ. Vérifiez les contraintes de validation."
        ),
    }, 400


def _format_not_null_error(
    error_message: str, error_detail: str | None
) -> Tuple[Dict[str, Any], int]:
    """Formate une erreur de contrainte NOT NULL en message clair."""
    # Analyser le message pour extraire le champ manquant
    # Exemple: "null value in column "customer_name" violates not-null constraint"

    message = "Un champ obligatoire est manquant."

    if error_detail:
        column_match = re.search(r'column "(\w+)"', error_detail, re.IGNORECASE)
        if column_match:
            column_name = column_match.group(1)
            message = f"Le champ '{column_name}' est obligatoire"
    elif "column" in error_message.lower():
        # Utiliser error_message comme fallback si error_detail n'est pas disponible
        column_match = re.search(r'column "(\w+)"', error_message, re.IGNORECASE)
        if column_match:
            column_name = column_match.group(1)
            message = f"Le champ '{column_name}' est obligatoire"

    return {
        "error": "not_null_violation",
        "message": message,
    }, 400
