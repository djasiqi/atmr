# ruff: noqa: T201, W293, DTZ003, DTZ007
"""
Routes API pour le Shadow Mode du DQN Dispatch.

Ces routes permettent de monitorer et contrôler le mode shadow.
"""
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from ext import role_required
from models.enums import UserRole
from services.rl.shadow_mode_manager import ShadowModeManager

shadow_mode_bp = Blueprint('shadow_mode', __name__, url_prefix='/api/shadow-mode')

# Instance globale du shadow mode manager
# En production, utiliser un singleton ou dependency injection
shadow_manager: ShadowModeManager | None = None


def get_shadow_manager() -> ShadowModeManager:
    """Récupère ou crée l'instance du shadow manager."""
    global shadow_manager
    if shadow_manager is None:
        shadow_manager = ShadowModeManager(
            model_path="data/rl/models/dqn_best.pth",
            log_dir="data/rl/shadow_mode",
            enable_logging=True
        )
    return shadow_manager


@shadow_mode_bp.route('/status', methods=['GET'])
@jwt_required()
@role_required(UserRole.admin)
def get_status():
    """
    Retourne le statut actuel du shadow mode.
    
    Returns:
        JSON avec statistiques et état du shadow mode
    """
    try:
        manager = get_shadow_manager()
        stats = manager.get_stats()

        return jsonify({
            "status": "active",
            "model_loaded": manager.agent is not None,
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@shadow_mode_bp.route('/stats', methods=['GET'])
@jwt_required()
@role_required(UserRole.admin)
def get_stats():
    """
    Retourne les statistiques détaillées du shadow mode.
    
    Query params:
        - period: "today" | "week" | "month" (défaut: today)
    
    Returns:
        JSON avec métriques détaillées
    """
    try:
        manager = get_shadow_manager()
        period = request.args.get('period', 'today')

        # Stats de session courante
        session_stats = manager.get_stats()

        # Rapport quotidien
        daily_report = manager.generate_daily_report()

        return jsonify({
            "period": period,
            "session_stats": session_stats,
            "daily_report": daily_report
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@shadow_mode_bp.route('/report/<date>', methods=['GET'])
@jwt_required()
@role_required(UserRole.admin)
def get_daily_report(date: str):
    """
    Retourne le rapport quotidien pour une date donnée.
    
    Args:
        date: Date au format YYYYMMDD
    
    Returns:
        JSON avec le rapport détaillé
    """
    try:
        manager = get_shadow_manager()

        # Valider le format de date
        datetime.strptime(date, '%Y%m%d')

        report = manager.generate_daily_report(date)

        return jsonify(report), 200

    except ValueError:
        return jsonify({"error": "Format de date invalide (attendu: YYYYMMDD)"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@shadow_mode_bp.route('/predictions', methods=['GET'])
@jwt_required()
@role_required(UserRole.admin)
def get_recent_predictions():
    """
    Retourne les prédictions récentes du shadow mode.
    
    Query params:
        - limit: Nombre de prédictions à retourner (défaut: 50)
        - date: Date au format YYYYMMDD (défaut: aujourd'hui)
    
    Returns:
        JSON avec liste des prédictions récentes
    """
    try:
        manager = get_shadow_manager()
        limit = int(request.args.get('limit', 50))
        date = request.args.get('date', datetime.utcnow().strftime('%Y%m%d'))

        predictions_file = manager.log_dir / f"predictions_{date}.jsonl"

        if not predictions_file.exists():
            return jsonify({
                "date": date,
                "predictions": [],
                "count": 0
            }), 200

        # Lire les dernières prédictions
        predictions = []
        with open(predictions_file, encoding="utf-8") as f:
            lines = f.readlines()
            # Prendre les N dernières lignes
            for line in lines[-limit:]:
                import json
                predictions.append(json.loads(line))

        return jsonify({
            "date": date,
            "predictions": predictions,
            "count": len(predictions),
            "limit": limit
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@shadow_mode_bp.route('/comparisons', methods=['GET'])
@jwt_required()
@role_required(UserRole.admin)
def get_recent_comparisons():
    """
    Retourne les comparaisons récentes (DQN vs Réel).
    
    Query params:
        - limit: Nombre de comparaisons à retourner (défaut: 50)
        - date: Date au format YYYYMMDD (défaut: aujourd'hui)
        - agreement: "true" | "false" | "all" (défaut: all)
    
    Returns:
        JSON avec liste des comparaisons récentes
    """
    try:
        manager = get_shadow_manager()
        limit = int(request.args.get('limit', 50))
        date = request.args.get('date', datetime.utcnow().strftime('%Y%m%d'))
        agreement_filter = request.args.get('agreement', 'all')

        comparisons_file = manager.log_dir / f"comparisons_{date}.jsonl"

        if not comparisons_file.exists():
            return jsonify({
                "date": date,
                "comparisons": [],
                "count": 0
            }), 200

        # Lire les dernières comparaisons
        comparisons = []
        with open(comparisons_file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                import json
                comp = json.loads(line)

                # Filtrer par agreement si demandé
                if agreement_filter == "true" and not comp.get("agreement"):
                    continue
                if agreement_filter == "false" and comp.get("agreement"):
                    continue

                comparisons.append(comp)

        return jsonify({
            "date": date,
            "comparisons": comparisons,
            "count": len(comparisons),
            "limit": limit,
            "agreement_filter": agreement_filter
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@shadow_mode_bp.route('/reload-model', methods=['POST'])
@jwt_required()
@role_required(UserRole.admin)
def reload_model():
    """
    Recharge le modèle DQN (utile après un réentraînement).
    
    Body (optionnel):
        - model_path: Chemin vers le nouveau modèle
    
    Returns:
        JSON confirmant le rechargement
    """
    try:
        global shadow_manager

        data = request.get_json() or {}
        model_path = data.get('model_path', 'data/rl/models/dqn_best.pth')

        # Créer une nouvelle instance avec le nouveau modèle
        shadow_manager = ShadowModeManager(
            model_path=model_path,
            log_dir="data/rl/shadow_mode",
            enable_logging=True
        )

        return jsonify({
            "message": "Modèle rechargé avec succès",
            "model_path": model_path,
            "model_loaded": shadow_manager.agent is not None
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

