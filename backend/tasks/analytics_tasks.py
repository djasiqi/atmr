# backend/tasks/analytics_tasks.py
"""
Tâches Celery pour les analytics et rapports automatiques.
"""

import logging
from datetime import date, timedelta
from celery_app import celery
from models import Company, User

logger = logging.getLogger(__name__)


@celery.task(name='analytics.aggregate_daily_stats')
def aggregate_daily_stats_task(company_id: int = None, day: date = None):
    """
    Tâche quotidienne : Agrège les stats du jour précédent.
    
    Args:
        company_id: ID d'une company spécifique (optionnel, sinon toutes)
        day: Date à agréger (optionnel, par défaut hier)
    
    Planification: Tous les jours à 1h du matin
    """
    
    try:
        from services.analytics.aggregator import aggregate_daily_stats
        
        if day is None:
            day = date.today() - timedelta(days=1)
        
        if company_id:
            companies = [Company.query.get(company_id)]
        else:
            companies = Company.query.filter_by(is_active=True).all()
        
        success_count = 0
        error_count = 0
        
        for company in companies:
            if not company:
                continue
            
            try:
                stats = aggregate_daily_stats(company.id, day)
                if stats:
                    success_count += 1
                    logger.info(
                        f"[Analytics] Daily stats aggregated for company {company.id} ({company.name}) - "
                        f"Quality: {stats.quality_score:.1f}"
                    )
                else:
                    error_count += 1
                    logger.warning(
                        f"[Analytics] No metrics found for company {company.id} on {day}"
                    )
            except Exception as e:
                error_count += 1
                logger.error(
                    f"[Analytics] Failed to aggregate daily stats for company {company.id}: {e}",
                    exc_info=True
                )
        
        logger.info(
            f"[Analytics] Daily aggregation complete: "
            f"{success_count} success, {error_count} errors"
        )
        
        return {
            "success": success_count,
            "errors": error_count,
            "date": day.isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Analytics] Daily aggregation task failed: {e}", exc_info=True)
        raise


@celery.task(name='analytics.send_daily_reports')
def send_daily_reports_task(company_id: int = None, day: date = None):
    """
    Tâche quotidienne : Envoie les rapports quotidiens par email.
    
    Args:
        company_id: ID d'une company spécifique (optionnel, sinon toutes)
        day: Date du rapport (optionnel, par défaut hier)
    
    Planification: Tous les jours à 8h du matin
    """
    
    try:
        from services.analytics.report_generator import generate_daily_report, generate_email_content
        
        if day is None:
            day = date.today() - timedelta(days=1)
        
        if company_id:
            companies = [Company.query.get(company_id)]
        else:
            companies = Company.query.filter_by(is_active=True).all()
        
        success_count = 0
        error_count = 0
        
        for company in companies:
            if not company:
                continue
            
            try:
                # Générer le rapport
                report = generate_daily_report(company.id, day)
                
                if "error" in report:
                    logger.warning(
                        f"[Analytics] Cannot generate daily report for company {company.id}: {report['error']}"
                    )
                    error_count += 1
                    continue
                
                # Générer l'email
                email_content = generate_email_content(report, "daily")
                
                # Envoyer aux admins de la company
                admins = User.query.filter_by(
                    company_id=company.id,
                    is_admin=True,
                    is_active=True
                ).all()
                
                for admin in admins:
                    if admin.email:
                        try:
                            # À adapter selon votre service d'email
                            # send_email(
                            #     to=admin.email,
                            #     subject=email_content["subject"],
                            #     html=email_content["body"]
                            # )
                            logger.info(
                                f"[Analytics] Daily report sent to {admin.email} "
                                f"for company {company.name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[Analytics] Failed to send email to {admin.email}: {e}"
                            )
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(
                    f"[Analytics] Failed to generate/send daily report for company {company.id}: {e}",
                    exc_info=True
                )
        
        logger.info(
            f"[Analytics] Daily reports sent: "
            f"{success_count} success, {error_count} errors"
        )
        
        return {
            "success": success_count,
            "errors": error_count,
            "date": day.isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Analytics] Daily reports task failed: {e}", exc_info=True)
        raise


@celery.task(name='analytics.send_weekly_reports')
def send_weekly_reports_task(company_id: int = None, week_start: date = None):
    """
    Tâche hebdomadaire : Envoie les rapports hebdomadaires par email.
    
    Args:
        company_id: ID d'une company spécifique (optionnel, sinon toutes)
        week_start: Date de début de semaine (optionnel, lundi dernier)
    
    Planification: Tous les lundis à 9h du matin
    """
    
    try:
        from services.analytics.report_generator import generate_weekly_report, generate_email_content
        
        if week_start is None:
            today = date.today()
            # Lundi de la semaine dernière
            week_start = today - timedelta(days=today.weekday() + 7)
        
        if company_id:
            companies = [Company.query.get(company_id)]
        else:
            companies = Company.query.filter_by(is_active=True).all()
        
        success_count = 0
        error_count = 0
        
        for company in companies:
            if not company:
                continue
            
            try:
                # Générer le rapport
                report = generate_weekly_report(company.id, week_start)
                
                if "error" in report:
                    logger.warning(
                        f"[Analytics] Cannot generate weekly report for company {company.id}: {report['error']}"
                    )
                    error_count += 1
                    continue
                
                # Générer l'email
                email_content = generate_email_content(report, "weekly")
                
                # Envoyer aux admins
                admins = User.query.filter_by(
                    company_id=company.id,
                    is_admin=True,
                    is_active=True
                ).all()
                
                for admin in admins:
                    if admin.email:
                        try:
                            # À adapter selon votre service d'email
                            # send_email(
                            #     to=admin.email,
                            #     subject=email_content["subject"],
                            #     html=email_content["body"]
                            # )
                            logger.info(
                                f"[Analytics] Weekly report sent to {admin.email} "
                                f"for company {company.name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[Analytics] Failed to send email to {admin.email}: {e}"
                            )
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(
                    f"[Analytics] Failed to generate/send weekly report for company {company.id}: {e}",
                    exc_info=True
                )
        
        logger.info(
            f"[Analytics] Weekly reports sent: "
            f"{success_count} success, {error_count} errors"
        )
        
        return {
            "success": success_count,
            "errors": error_count,
            "week_start": week_start.isoformat()
        }
        
    except Exception as e:
        logger.error(f"[Analytics] Weekly reports task failed: {e}", exc_info=True)
        raise


# Planification des tâches (à ajouter dans celery_app.py ou config)
"""
from celery.schedules import crontab

celery.conf.beat_schedule = {
    # Agrégation quotidienne à 1h du matin
    'aggregate-daily-stats': {
        'task': 'analytics.aggregate_daily_stats',
        'schedule': crontab(hour=1, minute=0),
    },
    
    # Rapports quotidiens à 8h du matin
    'send-daily-reports': {
        'task': 'analytics.send_daily_reports',
        'schedule': crontab(hour=8, minute=0),
    },
    
    # Rapports hebdomadaires les lundis à 9h
    'send-weekly-reports': {
        'task': 'analytics.send_weekly_reports',
        'schedule': crontab(day_of_week=1, hour=9, minute=0),
    },
}
"""

